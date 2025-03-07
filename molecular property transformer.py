import torch
import torch.nn as nn
import torch.optim as optim
import deepchem as dc
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the ZINC dataset
tasks, datasets, transformers = dc.molnet.load_zinc15(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets

print("Dataset loaded!")

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
print("Converted DeepChem dataset to PyG!")

train_graphs = deepchem_to_pyg(train_dataset)
valid_graphs = deepchem_to_pyg(valid_dataset)
test_graphs = deepchem_to_pyg(test_dataset)

train_loader = tg.data.DataLoader(train_graphs, batch_size=64, shuffle=True)
valid_loader = tg.data.DataLoader(valid_graphs, batch_size=64, shuffle=False)
test_loader = tg.data.DataLoader(test_graphs, batch_size=64, shuffle=False)

# We will use graphormer for this 
class GAT(nn.Module): #inherit from nn.Module
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, out_channels, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(out_channels * num_heads, out_channels, heads=1, dropout=0.2)
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x) #as propesed in the original gat paper
        x = self.gat2(x, edge_index, edge_attr)
        x = F.elu(x)
        #make graph level
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x.squeeze()
    
model = GAT(in_channels=75, out_channels=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Model created!")
def train():
    total_loss = 0
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def test():
    total_loss = 0
    model.eval()
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data.y)
        total_loss += loss.item()
    return total_loss / len(test_loader)

print("Training started!")
for epoch in range(300):
    loss = train()
    tloss = test()
    if epoch % 10 == 0:
        print(f'Epoch {epoch:>2} | Loss: {loss:.4f} | Tloss: {tloss:.4f}')