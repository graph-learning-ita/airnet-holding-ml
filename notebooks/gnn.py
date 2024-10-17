from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dsc2024 import datasets

df = datasets.get_train_dataset()
df_features = df.select_dtypes(include='number').drop("espera", axis=1)
df_features.fillna(0, inplace=True)
df_features.info()

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set a random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False     # Disable CuDNN benchmark to ensure determinism

# Call the seed function at the start of your script
set_seed(42)  # You can use any integer as the seed value

NUM_EDGE_FEATURES = len(df_features.columns)

# GNN model with a single GATConv layer for binary classification
class FlightGNNWithGAT(torch.nn.Module):
    def __init__(self, num_features=1, hidden_size=32, target_size=1):
        super(FlightGNNWithGAT, self).__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size

        # Single GATConv layer
        self.conv = GATConv(self.num_features, self.hidden_size, edge_dim=NUM_EDGE_FEATURES)

        # Linear layer for final edge-level classification output
        self.edge_mlp = nn.Linear(2 * self.hidden_size + NUM_EDGE_FEATURES, self.target_size)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Apply GATConv layer
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Concatenate node embeddings of the source and destination nodes of each edge, along with edge attributes
        row, col = edge_index
        edge_rep = torch.cat([x[row], x[col], edge_attr], dim=1)

        return torch.sigmoid(self.edge_mlp(edge_rep))

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Probabilities
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def train_test_split_data(data, test_size=0.2):
    num_edges = data.edge_index.shape[1]
    edge_indices = list(range(num_edges))

    train_indices, test_indices = train_test_split(edge_indices, test_size=test_size, random_state=42)

    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    return train_mask, test_mask

def train(model, data, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)  # Edge-level output
    loss = criterion(out[train_mask], data.y[train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()

def test(model, data, test_mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        preds = (out[test_mask] > 0.5).float()

        y_true = data.y[test_mask].numpy()
        y_pred = preds.numpy()

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        return acc, precision, recall, f1, cm

def prepare_data(df):
    # Convert your DataFrame into PyG Data format
    node_set = set(df['origem']).union(set(df['destino']))
    node_to_idx = {node: i for i, node in enumerate(node_set)}

    edge_index = torch.tensor([[node_to_idx[o], node_to_idx[d]] for o, d in zip(df['origem'], df['destino'])], dtype=torch.long).t().contiguous()

    # Edge features
    edge_attr = torch.tensor(df_features.values, dtype=torch.float)

    # Target edge feature to predict ('espera')
    y = torch.tensor(df['espera'].values, dtype=torch.float).unsqueeze(1)

    # Dummy node features (e.g., all ones)
    num_nodes = len(node_to_idx)
    node_features = torch.ones((num_nodes, 1))  # Dummy node features (1 for each node)

    # Create Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data

# Prepare the data
data = prepare_data(df)

# Split the data into train and test masks
train_mask, test_mask = train_test_split_data(data, test_size=0.2)

# Initialize the model
model = FlightGNNWithGAT(num_features=1, hidden_size=32, target_size=1)
criterion = FocalLoss(alpha=0.1, gamma=70)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
train_losses = []
for epoch in range(epochs):
    train_loss = train(model, data, train_mask, optimizer, criterion)
    train_losses.append(train_loss)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {train_loss:.4f}')

# Testing
acc, precision, recall, f1, cm = test(model, data, test_mask)
print(f'Test Accuracy: {acc:.4f}')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
print(f'Confusion Matrix:\n{cm}')
