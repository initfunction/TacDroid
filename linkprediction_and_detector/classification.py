import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def build_data_loader(x_list, edge_list, y_list=None, batch=64, shuffle=True):
    graph_num = len(x_list)
    data_list = []
    for i in range(graph_num):
        x = torch.tensor(x_list[i], dtype=torch.float)
        edge_index = torch.tensor(edge_list[i], dtype=torch.long)
        y = torch.tensor(y_list[i], dtype=torch.long) if y_list else None
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    dataloader = DataLoader(data_list, batch_size=batch, shuffle=shuffle)
    return dataloader

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.linear = nn.Linear(output_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]  # global_mean_pool(x, batch)
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.linear(x)
        return x

class Func_GCN():
    def __init__(self, num_features, hidden_channels, out_channels, num_classes):
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.model = model = GCN(num_features, hidden_channels, out_channels, num_classes)

    def train(self, dataset, epochs=100):
        # move to GPU (if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for data in dataset:
                optimizer.zero_grad()
                output = self.model(data.x, data.edge_index, data.batch)
                loss = criterion(output, data.y)
                print(loss)
                loss.backward()
                optimizer.step()
                out_label = torch.argmax(output, dim=-1)
                acc = torch.sum(out_label == data.y) / out_label.shape[0]
                print('Epoch: {:03d}, LOSS: {:.4f}, ACC: {:.4f}'.format(epoch, loss, acc))

    def test(self, test_dataset):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        predict = []
        acc = 0
        tot = 0
        for test_data in test_dataset:
            x = test_data.x.to(device)
            edge_index = test_data.edge_index.to(device)
            batch = test_data.batch.to(device)

            model.eval()
            with torch.no_grad():
                predicted_labels = self.model(x, edge_index, batch).argmax(dim=-1)
                acc = acc + torch.sum(predicted_labels == test_data.y)
                tot = tot + predicted_labels.shape[0]
                predict = predict + predicted_labels.tolist()
        acc = acc / tot
        return predict, acc

    def predict(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        model.eval()
        with torch.no_grad():
            batch = torch.zeros(x.shape[0], dtype=torch.int64)
            predicted_labels = self.model(x, edge_index, batch).argmax(dim=1)
        return predicted_labels
    
    def save_model(self, path):
        state_dict = self.model.state_dict()
        torch.save(state_dict, path)

    @staticmethod
    def load_model(path, num_features, hidden_channels, out_channels, num_classes):
        G = Func_GCN(num_features, hidden_channels, out_channels, num_classes)
        G.model.load_state_dict(torch.load(path))
        return G

