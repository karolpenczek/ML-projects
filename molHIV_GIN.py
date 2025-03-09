import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)

graph = dataset[0]

class GIN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(graph.x.shape[1], 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 64)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 64)))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 64)))
        self.lin1 = nn.Linear(64 * 3, 128)
        self.lin2 = nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #architecture as in the book
        #capture different levels of abstraction and concat them -> richer model
        x1 = self.conv1(x, edge_index) #GNN -> input: nodes, vertices, so we put in node features and edge index
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)

        x1 = global_mean_pool(x1, batch)
        x2 = global_mean_pool(x2, batch)
        x3 = global_mean_pool(x3, batch)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    
model = GIN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCEWithLogitsLoss()

def train():
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.to(torch.float))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return loss.item()/len(train_loader)

def accuracy(pred_y, y):
    pred_y = (pred_y > 0.5).float() 
    correct = (pred_y == y).sum().item()
    return correct / len(y)

@torch.no_grad()
def test():
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    
    #we will use the ROC-AUC score as suggested by OGB Stanford (source of the dataset)
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data.y.to(torch.float))

        total_loss += loss.item()

        all_labels.append(data.y.cpu().numpy())
        all_preds.append(output.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)

    return total_loss / len(test_loader), roc_auc


print("Training started!")
for epoch in range(300):
    loss = train()
    tloss, roc_auc = test()
    if epoch % 1 == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, Mean of gradients: {param.grad.abs().mean()}")
        print(f'Epoch {epoch:>2} | Loss: {loss:.4f} | Tloss: {tloss:.4f} | ROC-AUC: {roc_auc:.4f}')