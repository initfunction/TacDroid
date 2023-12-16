
import copy
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

from torch_geometric.nn.conv.gcn_conv import MyGCNConv
from torch_geometric.utils import train_test_split_edges, negative_sampling
from torch_geometric.nn import GAE, VGAE
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter


def build_data_loader(x_list, edge_list, y_list=None, batch=16, shuffle=True, input_dim=None):
    graph_num = len(x_list)
    data_list = []
    for i in range(graph_num):
        x = torch.tensor(x_list[i], dtype=torch.float)
        edge_index = torch.tensor(edge_list[i], dtype=torch.long)
        y = torch.tensor(y_list[i], dtype=torch.long) if y_list else None
        if not x.shape[0] and input_dim:
            x = torch.tensor([0 for i in range(input_dim)], dtype=torch.float)
            edge_index = torch.tensor([0], [0], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    dataloader = DataLoader(data_list, batch_size=batch, shuffle=shuffle)
    return dataloader

def negative_sample(train_data):
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
    # print(neg_edge_index.size(1))   
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    return edge_label, edge_label_index

def negative_sample_for_batch(train_data):
    final_label_list = []
    final_label_index_list = []
    num_graph = train_data.num_graphs  
    history_num_node = 0   
    edge_list = train_data.edge_index.tolist()
    for i in range(num_graph):
        num_node = torch.sum(train_data.batch == i)            
        highest_num_node = num_node + history_num_node          
        edge_label_list = [[], []]
        for l in range(len(edge_list[0])):
            src = edge_list[0][l]
            dst = edge_list[1][l]
            if src >= highest_num_node or dst >= highest_num_node:
                continue   
            elif src >= history_num_node and dst >= history_num_node:   # 是对应边
                edge_label_list[0].append(src - history_num_node)
                edge_label_list[1].append(dst - history_num_node)

        edge_index = torch.tensor(edge_label_list)
        edge_label = torch.ones(len(edge_label_list[0]))

        neg_edge_index = negative_sampling(
            edge_index=edge_index, num_nodes=num_node,
            num_neg_samples=len(edge_label_list[0]), method='sparse')
        neg_edge_index = neg_edge_index + history_num_node 
        history_num_node += num_node     
        # print(neg_edge_index.size(1))   
        t_edge_label_index = torch.cat(
            [edge_index, neg_edge_index],
            dim=-1,
        )
        t_edge_index = torch.cat([
            edge_label,
            edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)
        final_label_list.append(t_edge_index)
        final_label_index_list.append(t_edge_label_index)

    edge_index = torch.cat([i for i in final_label_list])
    edge_label_index = torch.cat([i.to(torch.int64) for i in final_label_index_list], dim=-1)

    return edge_index, edge_label_index

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)  # cached only for transductive learning, GCNConv
        self.conv2 = GCNConv(2 * out_channels, out_channels)  # cached = True

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels)
        self.conv_mu = SAGEConv(2 * out_channels, out_channels)
        self.conv_logstd = SAGEConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    

class MyGAE():
    def __init__(self, num_features, out_channels):
        self.num_features = num_features
        self.out_channels = out_channels
        self.model = GAE(GCNEncoder(self.num_features, self.out_channels))
        self.x = None
        self.train_pos_edge_index = None

    def train(self, dataset, epoch=100):
        
        # move to GPU (if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        for e in range(1, epoch + 1):
            for data in dataset:
                try:
                    udata = copy.deepcopy(data)
                    # udata.train_mask = udata.val_mask = udata.test_mask = None
                    # udata = train_test_split_edges(udata, test_ratio=0.2, val_ratio=0)
                    # x = udata.x.to(device)
                    # self.x = x
                    # train_pos_edge_index = udata.train_pos_edge_index.to(device)
                    # self.train_pos_edge_index = train_pos_edge_index
                    
                    transform = T.RandomLinkSplit(num_val=0, num_test=0.2, is_undirected=False, add_negative_train_samples=False)
                    train_data, val_data, test_data = transform(udata)
                    # edge_label, edge_label_index = negative_sample(train_data)
                    edge_label, edge_label_index = negative_sample_for_batch(train_data)
                    self.model.train()
                    optimizer.zero_grad()
                    # model.encode
                    z = self.model.encode(train_data.x, edge_label_index)
                    # recon_loss 为重构损失
                    loss = self.model.recon_loss(z, edge_label_index)
                    # if args.variational:
                    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
                    loss.backward()
                    optimizer.step()
                    los = float(loss)
                    print('los:', los)
                    auc, ap = self.test(test_data)
                    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(e, auc, ap))
                except Exception as exc:
                    print(exc)
                    print('跳过训练/测试步骤')

    def test(self, data):
        udata = copy.deepcopy(data)
        self.model.eval()
        edge_label, edge_label_index = negative_sample_for_batch(udata)
        with torch.no_grad():
            z = self.model.encode(udata.x, edge_label_index)
        indices_1 = torch.nonzero(edge_label == 1).squeeze()
        indices_0 = torch.nonzero(edge_label == 0).squeeze()
        return self.model.test(z, edge_label_index[:, indices_1], edge_label_index[:, indices_0])

    
    def predict(self, data, threshold=0.5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        Z = model.encode(x, edge_index)
        b = model.decode(Z, edge_index)
        print(Z)
        adj_matrix = torch.sigmoid(torch.matmul(Z, Z.t())) 
        print(adj_matrix)
        edge = [[], []]
        for src in range(data.num_nodes):
            for dst in range(data.num_nodes):
                if adj_matrix[src][dst] > threshold:
                    edge[0].append(src)
                    edge[1].append(dst)
        return edge
    
    def save_model(self, path):
        state_dict = self.model.state_dict()
        torch.save(state_dict, path)

    @staticmethod
    def load_model(path, num_features, out_channels):
        new_model = MyGAE(num_features, out_channels)
        new_model.model.load_state_dict(torch.load(path))
        return new_model
    
class MyVGAE():
    def __init__(self, num_features, out_channels):
        self.num_features = num_features
        self.out_channels = out_channels
        self.model = model = VGAE(VariationalGCNEncoder(self.num_features, self.out_channels))
        self.x = None
        self.train_pos_edge_index = None

    def train(self, dataset, epoch=100):

        # move to GPU (if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        for e in range(1, epoch + 1):
            for data in dataset:
                try:
                    # 有时会存在边不足的情况导致无法训练
                    udata = copy.deepcopy(data)
                    transform = T.RandomLinkSplit(num_val=0, num_test=0.2, is_undirected=False, add_negative_train_samples=False)
                    train_data, val_data, test_data = transform(udata)
                    # edge_label, edge_label_index = negative_sample(train_data)
                    edge_label, edge_label_index = negative_sample_for_batch(train_data)
                    self.model.train()
                    optimizer.zero_grad()
                    # model.encode 调用了我们传入的编码器
                    z = self.model.encode(train_data.x, edge_label_index)
                    # recon_loss 为重构损失
                    loss = self.model.recon_loss(z, edge_label_index)
                    # if args.variational:
                    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
                    loss.backward()
                    optimizer.step()
                    los = float(loss)
                    print('los:', los)
                    auc, ap = self.test(test_data)
                    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(e, auc, ap))
                except Exception as exc:
                    print(exc)
                    print('跳过训练/测试步骤')

    def test(self, data):
        udata = copy.deepcopy(data)
        self.model.eval()
        edge_label, edge_label_index = negative_sample_for_batch(udata)
        with torch.no_grad():
            z = self.model.encode(udata.x, edge_label_index)
        # 使用正边和负边来测试模型的准确率
        indices_1 = torch.nonzero(edge_label == 1).squeeze()
        indices_0 = torch.nonzero(edge_label == 0).squeeze()
        return self.model.test(z, edge_label_index[:, indices_1], edge_label_index[:, indices_0])

    def load(self, url):
        self.model = torch.load(url)
        self.model.eval()

    def encode(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        Z = model.encode(x, edge_index)
        return Z
    
    def predict(self, data, threshold=0.5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        Z = model.encode(x, edge_index)
        b = model.decode(Z, edge_index)
        out = model(x, edge_index)          # out有两维数据, 第一维等同于Z
        reconstructed_prob = torch.sigmoid(out[1])  # 重构概率（经过 sigmoid 激活）
        # 构建预测图
        adj_matrix = torch.sigmoid(torch.matmul(Z, Z.t()))  # 计算邻接矩阵
        print(adj_matrix)
        # 根据预测图输出潜在边
        edge = [[], []]
        for src in range(data.num_nodes):
            for dst in range(data.num_nodes):
                if adj_matrix[src][dst] > threshold:
                    edge[0].append(src)
                    edge[1].append(dst)
        return edge
    
    def save_model(self, path):
        state_dict = self.model.state_dict()
        torch.save(state_dict, path)

    @staticmethod
    def load_model(path, num_features, out_channels):
        new_model = MyVGAE(num_features, out_channels)
        new_model.model.load_state_dict(torch.load(path))
        return new_model

class DirectedEdgePredictionModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DirectedEdgePredictionModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.lin = torch.nn.Linear(2 * out_channels, 1)

    def encode(self, x, edge_index):
        normalized_features = F.normalize(x, p=2, dim=1)    # TODO
        x = self.conv1(normalized_features, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        x = torch.cat([src, dst], dim=-1)
        x = self.lin(x)
        return torch.sigmoid(x)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)

class MyDirectGCN():
    def __init__(self, num_features, hidden_channels, out_channels):
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.model = model = DirectedEdgePredictionModel(num_features, hidden_channels, out_channels)
        # for param in self.model.parameters():
        #     param.requires_grad = False
    
    def train(self, dataset, epoch=100):
        # move to GPU (if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()

        # summarywriter
        writer = SummaryWriter('D:\Code\log')
        print(writer.log_dir)
        for e in range(1, epoch + 1):
            e_loss = []
            e_acc = []
            for data in dataset:
                try:
                    # 有时会存在边不足的情况导致无法训练
                    udata = copy.deepcopy(data)
                    udata.train_mask = udata.val_mask = udata.test_mask = None
                    # udata = train_test_split_edges(udata, test_ratio=0.2, val_ratio=0)

                    # 加negative会导致正负边数量不均匀
                    transform = T.RandomLinkSplit(num_val=0, num_test=0.2, is_undirected=False, add_negative_train_samples=False)
                    train_data, val_data, test_data = transform(udata)
                    # edge_label, edge_label_index = negative_sample(train_data)
                    edge_label, edge_label_index = negative_sample_for_batch(train_data)
                    self.model.train()
                    optimizer.zero_grad()
                    # model.encode 调用了我们传入的编码器
                    out = self.model(train_data.x.to(device), edge_label_index.to(device)).view(-1)
                    # recon_loss 为重构损失
                    loss = criterion(out, edge_label.to(device))
                    loss.backward()
                    optimizer.step()
                    los = float(loss)
                    print('los:', los)
                    loss, acc = self.test(test_data)
                    e_loss.append(los)
                    e_acc.append(acc)
                    print('Epoch: {:03d}, LOSS: {:.4f}, ACC: {:.4f}'.format(e, los, acc))
                except Exception as exc:
                    print(exc)
                    print('跳过训练/测试步骤')
            e_loss_num = torch.mean(torch.tensor(e_loss))
            e_acc_num = torch.mean(torch.tensor(e_acc))
            # 记录损失
            writer.add_scalar('Loss', e_loss_num.item(), e)
            # 记录准确率
            writer.add_scalar('Accuracy', e_acc_num.item(), e)
        writer.close()    

    def test(self, test_data, threshold=0.5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        criterion = torch.nn.BCELoss()
        with torch.no_grad():
            edge_label, edge_label_index = negative_sample(test_data)
            out = self.model(test_data.x.to(device), edge_label_index.to(device)).view(-1)
            loss = criterion(out, edge_label.to(device))
            # 使用正边和负边来测试模型的准确率
            out_label = torch.where(out >= threshold, torch.tensor(1), torch.tensor(0))
            acc = torch.sum(out_label == edge_label.to(torch.int64)).item() / out_label.shape[0]
        return loss, acc

    def predict(self, data, threshold=0.5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        z = model.encode(x, edge_index)
        # 构建预测边
        edge_all_index = [[], []]
        for src in range(data.num_nodes):
            for dst in range(data.num_nodes):
                edge_all_index[0].append(src)
                edge_all_index[1].append(dst)
        edge_all_index = torch.tensor(edge_all_index, dtype=torch.long).to(device)
        out = model.decode(z, edge_all_index)
        # 根据预测结果输出潜在边
        edge = [[], []]
        i = 0
        for src in range(data.num_nodes):
            for dst in range(data.num_nodes):
                if out[i] > threshold:
                    edge[0].append(src)
                    edge[1].append(dst)
                i += 1
        return edge
    
    def predict_score(self, data):
        '''预测, 但是没有threshold, 直接返回所有边的分数'''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        z = model.encode(x, edge_index)
        # 构建预测边
        edge_all_index = [[], []]
        for src in range(data.num_nodes):
            for dst in range(data.num_nodes):
                edge_all_index[0].append(src)
                edge_all_index[1].append(dst)
        edge_all_index = torch.tensor(edge_all_index, dtype=torch.long).to(device)
        out = model.decode(z, edge_all_index)
        return edge_all_index, out

    def save_model(self, path):
        state_dict = self.model.state_dict()
        torch.save(state_dict, path)

    @staticmethod
    def load_model(path, num_features, hidden_channels, out_channels):
        new_model = MyDirectGCN(num_features, hidden_channels, out_channels)
        new_model.model.load_state_dict(torch.load(path))
        return new_model
       
class NTNConvModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k):
        super(NTNConvModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        # 此时out_channels的维度就相当于后续输入NTN的输出维度，即公式中的d 
        # W的维度为d*d*k, 其中d为特征维度, k可以理解为多头维度
        # e1We2的维度为k
        # V的维度为k*2d, U为k, b为k 
        self.d = d = out_channels
        self.k = k

        self.W = torch.nn.Parameter(torch.Tensor(d, d, k))
        self.V = torch.nn.Parameter(torch.Tensor(k, 2 * d))
        self.b = torch.nn.Parameter(torch.Tensor(k, 1))
        self.U = torch.nn.Parameter(torch.Tensor(1, k))
        
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.xavier_uniform_(self.V)

        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(0.2)
        self.normalize = torch.nn.BatchNorm1d(in_channels)

    def encode(self, x, edge_index):
        # normalized_features = F.normalize(x, p=2, dim=1)    # TODO 标准化
        # 使用F.normalize效果不大好的, 使用batchnormalization
        normalized_features = self.normalize(x)
        x = self.conv1(normalized_features, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        # 此处的src和dst是由batch条边的起始/结束节点特征(维度为out_channels)组成的列表
        x_out = []
        for i in range(self.k):
            # 假设batch = 25
            # [25, 8] * W[8, 8, 2] * [25, 8] -> [25, 2]
            # 所以对于每个k: [25, 1, 8] * [8, 8] * [25, 8. 1] -> [25, 1]
            k_out = torch.matmul(src.unsqueeze(dim=1), self.W[:, :, i])
            k_out = torch.matmul(k_out, dst.unsqueeze(dim=2))
            x_out.append(k_out)
        # 拼接k个[25, 1]并转置, 形成[k, 25]
        # 与V[k, 2d] * [d + d, 25] = [k, 25]相加, 得到[k, 25]
        # 加上b[k, 1], 得到整体[k, 25]经过tanh与U[1, k]相乘, 得到[25, 1], 每个即为边是否存在的分数
        x = torch.concatenate(x_out, dim=1).T
        x = x + torch.matmul(self.V, torch.cat([src, dst], dim=-1).T) + self.b
        x = self.tanh(x)
        x = self.dropout(x)
        x = torch.matmul(self.U, x).view(x.shape[-1])
        return torch.sigmoid(x)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)

class MyDirectNTNGCN():
    def __init__(self, num_features, hidden_channels, out_channels, k):
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.model = model = NTNConvModel(num_features, hidden_channels, out_channels, k)
        # for param in self.model.parameters():
        #     param.requires_grad = False
    
    def train(self, dataset, epoch=100):
        # move to GPU (if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        # summarywriter
        writer = SummaryWriter('D:\Code\log')
        print(writer.log_dir)
        for e in range(1, epoch + 1):
            e_loss = []
            e_acc = []
            for data in dataset:
                try:
                    # 有时会存在边不足的情况导致无法训练
                    udata = copy.deepcopy(data)
                    udata.train_mask = udata.val_mask = udata.test_mask = None
                    # udata = train_test_split_edges(udata, test_ratio=0.2, val_ratio=0)

                    # 加negative会导致正负边数量不均匀
                    transform = T.RandomLinkSplit(num_val=0, num_test=0.2, is_undirected=False, add_negative_train_samples=False)
                    train_data, val_data, test_data = transform(udata)
                    # edge_label, edge_label_index = negative_sample(train_data)
                    edge_label, edge_label_index = negative_sample_for_batch(train_data)
                    self.model.train()
                    optimizer.zero_grad()
                    # model.encode 调用了我们传入的编码器
                    out = self.model(train_data.x.to(device), edge_label_index.to(device)).view(-1)
                    # recon_loss 为重构损失
                    loss = criterion(out, edge_label.to(device))
                    loss.backward()
                    optimizer.step()
                    los = float(loss)
                    print('los:', los)
                    loss, acc = self.test(test_data)
                    e_loss.append(los)
                    e_acc.append(acc)
                    print('Epoch: {:03d}, LOSS: {:.4f}, ACC: {:.4f}'.format(e, los, acc))
                except Exception as exc:
                    print(exc)
                    print('跳过训练/测试步骤')
            e_loss_num = torch.mean(torch.tensor(e_loss))
            e_acc_num = torch.mean(torch.tensor(e_acc))
            # 记录损失
            writer.add_scalar('Loss', e_loss_num.item(), e)
            # 记录准确率
            writer.add_scalar('Accuracy', e_acc_num.item(), e)
        writer.close()    

    def test(self, test_data, threshold=0.5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            edge_label, edge_label_index = negative_sample(test_data)
            out = self.model(test_data.x.to(device), edge_label_index.to(device)).view(-1)
            loss = criterion(out, edge_label.to(device))
            # 使用正边和负边来测试模型的准确率
            out_label = torch.where(out >= threshold, torch.tensor(1), torch.tensor(0))
            acc = torch.sum(out_label == edge_label.to(torch.int64)).item() / out_label.shape[0]
        return loss, acc

    def predict(self, data, threshold=0.5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        z = model.encode(x, edge_index)
        # 构建预测边
        edge_all_index = [[], []]
        for src in range(data.num_nodes):
            for dst in range(data.num_nodes):
                edge_all_index[0].append(src)
                edge_all_index[1].append(dst)
        edge_all_index = torch.tensor(edge_all_index, dtype=torch.long).to(device)
        out = model.decode(z, edge_all_index)
        # 根据预测结果输出潜在边
        edge = [[], []]
        i = 0
        for src in range(data.num_nodes):
            for dst in range(data.num_nodes):
                if out[i] > threshold:
                    edge[0].append(src)
                    edge[1].append(dst)
                i += 1
        return edge
    
    def predict_score(self, data):
        '''预测, 但是没有threshold, 直接返回所有边的分数'''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        z = model.encode(x, edge_index)
        # 构建预测边
        edge_all_index = [[], []]
        for src in range(data.num_nodes):
            for dst in range(data.num_nodes):
                edge_all_index[0].append(src)
                edge_all_index[1].append(dst)
        edge_all_index = torch.tensor(edge_all_index, dtype=torch.long).to(device)
        out = model.decode(z, edge_all_index)
        return edge_all_index, out

    def save_model(self, path):
        state_dict = self.model.state_dict()
        torch.save(state_dict, path)

    @staticmethod
    def load_model(path, num_features, hidden_channels, out_channels):
        new_model = MyDirectGCN(num_features, hidden_channels, out_channels)
        new_model.model.load_state_dict(torch.load(path))
        return new_model

       
class GravityAEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k):
        super(GravityAEModel, self).__init__()
        self.conv1 = MyGCNConv(in_channels, hidden_channels)
        self.conv2 = MyGCNConv(hidden_channels, out_channels + 1)
        # 在GravityAE中, GCN新增加了最后一维用于表示节点的质量
        # 用于有向图的识别参数，改变decoder的结构
        # (Z, M') = GCN(A, X)
            # Z: (n * d)矩阵, n是节点数量, d是节点特征
            # M': n维向量, m' = logG mi  
    
        self.dropout = torch.nn.Dropout(0.2)
        self.normalize1 = torch.nn.BatchNorm1d(in_channels)
        self.normalize2 = torch.nn.BatchNorm1d(out_channels + 1)


    def encode(self, x, edge_index):
        # normalized_features = F.normalize(x, p=2, dim=1)    # TODO 标准化
        # 使用F.normalize效果不大好的, 使用batchnormalization
        normalized_features = self.normalize1(x)
        x = self.conv1(normalized_features, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x, edge_index)
        # x = self.normalize2(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        return x
    
    def decode(self, z):
        # z为encode的结果, 维度为[n, d + 1], n为节点数量，d为特征维度
        # 则m的维度为[n, 1], z_i和z_j的维度为[n, d]
        # 构造矩阵实现集中运算
        '''
        将源特征在第二维上重复n次, 如假设A, B, C三个点(n=3), 每个点有4维特征(d=4)
            A [1, 2, 3, 4]
            B [2, 3, 4, 5]
            C [0, 0, 0, 0]
        则z[n * d]的结构为, 可做扩展(第0维扩展与第1维扩展):
            [[1, 2, 3, 4],    [[A,    [[A],
             [2, 3, 4, 5],      B,     [B],
             [0, 0, 0, 0]]      C]]    [C]]
        并分别在第0维/第1维上重复n=3次, 构成源矩阵/目标矩阵
            [[A, B, C],     [[A, A, A],
             [A, B, C],      [B, B, B],
             [A, B, C]]      [C, C, C]]
        因此将两个矩阵相减, 第0维的数字代表了源节点, 第1维代表目标节点, 第2维则是特征
            [[A-A, B-A, C-A],
             [A-B, B-B, C-B],
             [A-C, B-C, C-C]]
        之后分别求欧氏距离和log值, 得到n*n的矩阵, 并构建mj矩阵, 通过sigmoid得到结果矩阵:
            [[mA, mA, mA],      [[A11, A21, A31],  
             [mB, mB, mB],       [A12, A22, A32],
             [mC, mC, mC]]       [A13, A23, A33]]
        
        '''
        mj = z[:, -1]
        n, d = z.shape[0], z.shape[1] - 1
        src = z[:, :-1].unsqueeze(0).repeat(n, 1, 1)    
        dst = z[:, :-1].unsqueeze(1).repeat(1, n, 1)
        distances = torch.norm(src - dst, dim=2)
        m =  mj.unsqueeze(0).unsqueeze(1).repeat(1, n, 1)
        A = torch.sigmoid(m - distances)
        return A

    def decode_typical_edge(self, z, edge_index):
        A = self.decode(z)
        res = A[0, edge_index[0], edge_index[1]]
        return torch.Tensor(res)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode_typical_edge(z, edge_index)

class MyGravityAEGCN():
    def __init__(self, num_features, hidden_channels, out_channels, k):
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.model = model = GravityAEModel(num_features, hidden_channels, out_channels, k)
        # for param in self.model.parameters():
        #     param.requires_grad = False
    
    def train(self, dataset, epoch=100):
        # move to GPU (if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)  # 我要对学习率开刀了!
        criterion = torch.nn.MSELoss()
        # summarywriter
        writer = SummaryWriter('D:\Code\log')
        print(writer.log_dir)
        for e in range(1, epoch + 1):
            e_loss = []
            e_acc = []
            for data in dataset:
                try:
                    # 有时会存在边不足的情况导致无法训练
                    udata = copy.deepcopy(data)
                    udata.train_mask = udata.val_mask = udata.test_mask = None
                    # udata = train_test_split_edges(udata, test_ratio=0.2, val_ratio=0)

                    # 加negative会导致正负边数量不均匀
                    transform = T.RandomLinkSplit(num_val=0, num_test=0.2, is_undirected=False, add_negative_train_samples=False)
                    train_data, val_data, test_data = transform(udata)
                    # edge_label, edge_label_index = negative_sample(train_data)
                    edge_label, edge_label_index = negative_sample_for_batch(train_data)
                    self.model.train()
                    optimizer.zero_grad()
                    # model.encode 调用了我们传入的编码器
                    out = self.model(train_data.x.to(device), edge_label_index.to(device)).view(-1)
                    # recon_loss 为重构损失
                    loss = criterion(out, edge_label.to(device))
                    loss.backward()
                    optimizer.step()
                    los = float(loss)
                    print('los:', los)
                    loss, acc = self.test(test_data)
                    e_loss.append(los)
                    e_acc.append(acc)
                    print('Epoch: {:03d}, LOSS: {:.4f}, ACC: {:.4f}'.format(e, los, acc))
                except Exception as exc:
                    print(exc)
                    print('跳过训练/测试步骤')
            e_loss_num = torch.mean(torch.tensor(e_loss))
            e_acc_num = torch.mean(torch.tensor(e_acc))
            # 记录损失
            writer.add_scalar('Loss', e_loss_num.item(), e)
            # 记录准确率
            writer.add_scalar('Accuracy', e_acc_num.item(), e)
        writer.close()    

    def test(self, test_data, threshold=0.5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            edge_label, edge_label_index = negative_sample_for_batch(test_data)
            out = self.model(test_data.x.to(device), edge_label_index.to(device)).view(-1)
            loss = criterion(out, edge_label.to(device))
            # 使用正边和负边来测试模型的准确率
            out_label = torch.where(out >= threshold, torch.tensor(1), torch.tensor(0))
            acc = torch.sum(out_label == edge_label.to(torch.int64)).item() / out_label.shape[0]
        return loss, acc

    def predict(self, data, threshold=0.5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        model = self.model.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        z = model.encode(x, edge_index)
        with torch.no_grad():
        # 构建预测边
            edge_all_index = [[], []]
            for src in range(data.num_nodes):
                for dst in range(data.num_nodes):
                    edge_all_index[0].append(src)
                    edge_all_index[1].append(dst)
            edge_all_index = torch.tensor(edge_all_index, dtype=torch.long).to(device)
            out = model.decode_typical_edge(z, edge_all_index)
            # 根据预测结果输出潜在边
            edge = [[], []]
            i = 0
            for src in range(data.num_nodes):
                for dst in range(data.num_nodes):
                    if out[i] > threshold:
                        edge[0].append(src)
                        edge[1].append(dst)
                    i += 1
        return edge
    
    def predict_score(self, data):
        '''预测, 但是没有threshold, 直接返回所有边的分数'''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        model = self.model.to(device)
        with torch.no_grad():
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            z = model.encode(x, edge_index)
            # 构建预测边
            edge_all_index = [[], []]
            for src in range(data.num_nodes):
                for dst in range(data.num_nodes):
                    edge_all_index[0].append(src)
                    edge_all_index[1].append(dst)
            edge_all_index = torch.tensor(edge_all_index, dtype=torch.long).to(device)
            out = model.decode_typical_edge(z, edge_all_index)
            return edge_all_index, out

    def save_model(self, path):
        state_dict = self.model.state_dict()
        torch.save(state_dict, path)

    @staticmethod
    def load_model(path, num_features, hidden_channels, out_channels):
        new_model = GravityAEModel(num_features, hidden_channels, out_channels, 0)
        new_model.load_state_dict(torch.load(path))
        G = MyGravityAEGCN(num_features, hidden_channels, out_channels, 0)
        G.model = new_model
        return G

if __name__ == '__main__':

    # 20231010 搞一个新的graph AE进行测试
    # x = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5], [0, 0, 0, 0]], dtype=torch.float32)
    # edge_index = torch.tensor([[0, 1, 1],
    #                            [1, 0, 2]], dtype=torch.long)
    # data = Data(x=x, edge_index=edge_index)
    # loader = DataLoader([data])

    # gae = MyGravityAEGCN(4, 3, 4, 1)
    # gae.train(loader, 1)


    V2v = 500
    edge_index = torch.tensor([[0, 1, 1, 2, 0, 2, 7, 4, 8, 2, 3],
                            [1, 0, 2, 1, 2, 0, 9, 5, 7, 5, 4]], dtype=torch.long)
    edge_attr = torch.tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype=torch.float)
    # Y = torch.tensor([[0, 0], [1, 1], [2, 0]], dtype=torch.float)
    data = Data(x=torch.randn(12, V2v), edge_index=torch.randint(low=0, high=11, size=(2, 35))) # y=Y
    print(data)

    datas = []
    for i in range(20):
        data2 = Data(x=torch.randn(9, V2v), edge_index=torch.randint(low=0, high=8, size=(2, 25))) # y=Y
        data3 = Data(x=torch.randn(8, V2v), edge_index=torch.randint(low=0, high=6, size=(2, 7))) # y=Y
        data4 = Data(x=torch.randn(14, V2v), edge_index=torch.randint(low=0, high=14, size=(2, 45))) # y=Y
        data5 = Data(x=torch.randn(13, V2v), edge_index=torch.randint(low=0, high=13, size=(2, 38))) # y=Y
        data6 = Data(x=torch.randn(9, V2v), edge_index=torch.randint(low=0, high=8, size=(2, 25))) # y=Y
        data7 = Data(x=torch.randn(8, V2v), edge_index=torch.randint(low=0, high=6, size=(2, 7))) # y=Y
        data8 = Data(x=torch.randn(14, V2v), edge_index=torch.randint(low=0, high=14, size=(2, 45))) # y=Y
        data9 = Data(x=torch.randn(13, V2v), edge_index=torch.randint(low=0, high=13, size=(2, 38))) # y=Y
        datas.extend([data, data2, data3, data4, data5, data6, data7, data8, data9])


    dataset = datas
    # gae = MyDirectGCN(V2v, 16, 8)
    # gae = MyDirectNTNGCN(V2v, 200, 50, 5)
    gae = MyGravityAEGCN(V2v, 64, 16, 1)
    # gae = MyGAE(20, 8)
    # old_acc = []
    # for data in dataset:
    #     transform = T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=False, add_negative_train_samples=False)
    #     data, val_data, test_data = transform(data)
    #     acc = gae.test(data)
    #     old_acc.append(acc)
    # # gae.train(dataset, 100)

    # o = gae.predict(data3)
    # print(o)

    # new_acc = []
    # for data in dataset:
    #     transform = T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=False, add_negative_train_samples=False)
    #     data, val_data, test_data = transform(data)
    #     acc = gae.test(data)
    #     new_acc.append(acc)

    # print(old_acc)
    # print(new_acc)


    edge_index4 = torch.tensor([[0, 4, 1, 2, 3, 2],
                                [3, 1, 4, 3, 2, 4]], dtype=torch.long)
    data4 = Data(x=torch.randn(5, V2v), edge_index=edge_index4) # y=Y, edge_attr=edge_attr
    o = gae.predict(data4)
    print(o)

    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=15, shuffle=True)
    gae.train(loader, 200)
    o = gae.predict(data4)
    print(o)

    # gae.save_model('model/mymodel/lp_vgae_20_8.pt')
    # e = MyVGAE.load_model('model/mymodel/lp_vgae_20_8.pt', 20, 8)
    # e.train(loader, 20)




'''
# GAE模板代码

import argparse
import os.path as osp
import time

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--variational', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--epochs', type=int, default=400)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=False),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=transform)
train_data, val_data, test_data = dataset[0]


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # in_channels 是特征数量, out_channels * 2 是因为我们有两个GCNConv, 最后我们得到embedding大小的向量
        # cached 因为我们只有一张图
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()


        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


in_channels, out_channels = dataset.num_features, 16

if not args.variational and not args.linear:
    model = GAE(GCNEncoder(in_channels, out_channels))
elif not args.variational and args.linear:
    model = GAE(LinearEncoder(in_channels, out_channels))
elif args.variational and not args.linear:
    model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
elif args.variational and args.linear:
    model = VGAE(VariationalLinearEncoder(in_channels, out_channels))

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if args.variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    auc, ap = test(test_data)
    print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
'''