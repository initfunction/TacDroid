
import copy
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
# enhanced 
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
            elif src >= history_num_node and dst >= history_num_node:  
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
                    transform = T.RandomLinkSplit(num_val=0, num_test=0.2, is_undirected=False, add_negative_train_samples=False)
                    train_data, val_data, test_data = transform(udata)
                    edge_label, edge_label_index = negative_sample_for_batch(train_data)
                    self.model.train()
                    optimizer.zero_grad()
                    z = self.model.encode(train_data.x, edge_label_index)
                    loss = self.model.recon_loss(z, edge_label_index)
                    loss.backward()
                    optimizer.step()
                    los = float(loss)
                    print('los:', los)
                    auc, ap = self.test(test_data)
                    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(e, auc, ap))
                except Exception as exc:
                    print(exc)

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
                    udata = copy.deepcopy(data)
                    transform = T.RandomLinkSplit(num_val=0, num_test=0.2, is_undirected=False, add_negative_train_samples=False)
                    train_data, val_data, test_data = transform(udata)
                    edge_label, edge_label_index = negative_sample_for_batch(train_data)
                    self.model.train()
                    optimizer.zero_grad()
                    z = self.model.encode(train_data.x, edge_label_index)
                    loss = self.model.recon_loss(z, edge_label_index)
                    loss.backward()
                    optimizer.step()
                    los = float(loss)
                    print('los:', los)
                    auc, ap = self.test(test_data)
                    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(e, auc, ap))
                except Exception as exc:
                    print(exc)

    def test(self, data):
        udata = copy.deepcopy(data)
        self.model.eval()
        edge_label, edge_label_index = negative_sample_for_batch(udata)
        with torch.no_grad():
            z = self.model.encode(udata.x, edge_label_index)
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
        out = model(x, edge_index)       
        reconstructed_prob = torch.sigmoid(out[1]) 
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
                    udata = copy.deepcopy(data)
                    udata.train_mask = udata.val_mask = udata.test_mask = None
                    transform = T.RandomLinkSplit(num_val=0, num_test=0.2, is_undirected=False, add_negative_train_samples=False)
                    train_data, val_data, test_data = transform(udata)
                    edge_label, edge_label_index = negative_sample_for_batch(train_data)
                    self.model.train()
                    optimizer.zero_grad()
                    out = self.model(train_data.x.to(device), edge_label_index.to(device)).view(-1)
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
            e_loss_num = torch.mean(torch.tensor(e_loss))
            e_acc_num = torch.mean(torch.tensor(e_acc))
            writer.add_scalar('Loss', e_loss_num.item(), e)
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
            out_label = torch.where(out >= threshold, torch.tensor(1), torch.tensor(0))
            acc = torch.sum(out_label == edge_label.to(torch.int64)).item() / out_label.shape[0]
        return loss, acc

    def predict(self, data, threshold=0.5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        z = model.encode(x, edge_index)
        edge_all_index = [[], []]
        for src in range(data.num_nodes):
            for dst in range(data.num_nodes):
                edge_all_index[0].append(src)
                edge_all_index[1].append(dst)
        edge_all_index = torch.tensor(edge_all_index, dtype=torch.long).to(device)
        out = model.decode(z, edge_all_index)
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        z = model.encode(x, edge_index)
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
        normalized_features = self.normalize(x)
        x = self.conv1(normalized_features, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        x_out = []
        for i in range(self.k):
            k_out = torch.matmul(src.unsqueeze(dim=1), self.W[:, :, i])
            k_out = torch.matmul(k_out, dst.unsqueeze(dim=2))
            x_out.append(k_out)
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
    
    def train(self, dataset, epoch=100):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        writer = SummaryWriter('D:\Code\log')
        print(writer.log_dir)
        for e in range(1, epoch + 1):
            e_loss = []
            e_acc = []
            for data in dataset:
                try:
                    udata = copy.deepcopy(data)
                    udata.train_mask = udata.val_mask = udata.test_mask = None
                    transform = T.RandomLinkSplit(num_val=0, num_test=0.2, is_undirected=False, add_negative_train_samples=False)
                    train_data, val_data, test_data = transform(udata)
                    edge_label, edge_label_index = negative_sample_for_batch(train_data)
                    self.model.train()
                    optimizer.zero_grad()
                    out = self.model(train_data.x.to(device), edge_label_index.to(device)).view(-1)
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
            e_loss_num = torch.mean(torch.tensor(e_loss))
            e_acc_num = torch.mean(torch.tensor(e_acc))
            writer.add_scalar('Loss', e_loss_num.item(), e)
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
            out_label = torch.where(out >= threshold, torch.tensor(1), torch.tensor(0))
            acc = torch.sum(out_label == edge_label.to(torch.int64)).item() / out_label.shape[0]
        return loss, acc

    def predict(self, data, threshold=0.5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        z = model.encode(x, edge_index)
        edge_all_index = [[], []]
        for src in range(data.num_nodes):
            for dst in range(data.num_nodes):
                edge_all_index[0].append(src)
                edge_all_index[1].append(dst)
        edge_all_index = torch.tensor(edge_all_index, dtype=torch.long).to(device)
        out = model.decode(z, edge_all_index)
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        z = model.encode(x, edge_index)
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
    
        self.dropout = torch.nn.Dropout(0.2)
        # self.normalize1 = torch.nn.BatchNorm1d(in_channels)
        self.normalize1 = torch.nn.BatchNorm1d(hidden_channels)
        self.normalize2 = torch.nn.BatchNorm1d(out_channels + 1)


    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.normalize1(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x, edge_index)
        x = self.normalize2(x)
        x = self.dropout(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)        
        return x
    
    def decode(self, z):
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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3) 
        criterion = torch.nn.MSELoss(reduction='sum')
        writer = SummaryWriter('D:\Code\log')
        print(writer.log_dir)
        for e in range(1, epoch + 1):
            e_loss = []
            e_acc = []
            for data in dataset:
                try:
                    udata = copy.deepcopy(data)
                    udata.train_mask = udata.val_mask = udata.test_mask = None
                    transform = T.RandomLinkSplit(num_val=0, num_test=0.2, is_undirected=False, add_negative_train_samples=False)
                    train_data, val_data, test_data = transform(udata)
                    edge_label, edge_label_index = negative_sample_for_batch(train_data)
                    self.model.train()
                    optimizer.zero_grad()
                    out = self.model(train_data.x.to(device), edge_label_index.to(device)).view(-1)
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
            e_loss_num = torch.mean(torch.tensor(e_loss))
            e_acc_num = torch.mean(torch.tensor(e_acc))
            writer.add_scalar('Loss', e_loss_num.item(), e)
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
            edge_all_index = [[], []]
            for src in range(data.num_nodes):
                for dst in range(data.num_nodes):
                    edge_all_index[0].append(src)
                    edge_all_index[1].append(dst)
            edge_all_index = torch.tensor(edge_all_index, dtype=torch.long).to(device)
            out = model.decode_typical_edge(z, edge_all_index)
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        model = self.model.to(device)
        with torch.no_grad():
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            z = model.encode(x, edge_index)
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
    pass