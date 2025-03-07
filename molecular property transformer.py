import torch
import torch.nn as nn
import torch.optim as optim
import deepchem as dc
import torch_geometric as tg
from torch_geometric.data import Data
from torch_geometric.nn import GraphTransformerConv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the ZINC dataset
tasks, datasets, transformers = dc.molnet.load_zinc15(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets


X_train = torch.tensor(train_dataset.X)
y_train = torch.tensor(train_dataset.y) 
X_valid = torch.tensor(valid_dataset.X)
y_valid = torch.tensor(valid_dataset.y)
X_test = torch.tensor(test_dataset.X)
y_test = torch.tensor(test_dataset.y)

def deepchem_to_pyg(dataset):
    graphs = []
    for i in range (len(dataset)):
        x = dataset.X[i]  # Extract molecule graph
        y = dataset.y[i]  # Extract label

        node_features = torch.tensor(x.node_features, dtype=torch.float)
        edge_index = torch.tensor(x.edge_index, dtype=torch.long).T  # Convert to (2, num_edges)
        edge_features = torch.tensor(x.edge_features, dtype=torch.float)
        label = torch.tensor(y, dtype=torch.float)

        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=label)
        graphs.append(graph)

    return graphs

train_graphs = deepchem_to_pyg(train_dataset)
valid_graphs = deepchem_to_pyg(valid_dataset)
test_graphs = deepchem_to_pyg(test_dataset)

train_loader = tg.data.DataLoader(train_graphs, batch_size=64, shuffle=True)
valid_loader = tg.data.DataLoader(valid_graphs, batch_size=64, shuffle=False)
test_loader = tg.data.DataLoader(test_graphs, batch_size=64, shuffle=False)

# We will use graphormer for this 
class Transformer(nn.Module): #inherit from nn.Module
    def __init__(self):
        super(Transformer, self).__init__()
        # Define the layers: Convolutional, 2 linear layers leaky relu in between
        self.conv1 = GraphTransformerConv(in_channels=75, out_channels=64, edge_dim=6)
        self.leaky_relu1 = nn.LeakyReLU()
        self.linear1 = nn.Linear(64, 32)
        self.leaky_relu2 = nn.LeakyReLU()
        self.linear2 = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.leaky_relu1(x)
        x = self.linear1(x)
        x = self.leaky_relu2(x)
        x = self.linear2(x)
        return self
    
model = Transformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

