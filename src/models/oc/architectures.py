# src/models/oc/architectures.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

class _ReadoutHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = dropout
    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin(h)

class OCGAT(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 33, layers: int = 4, heads: int = 8, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        # primeira camada
        self.layers.append(GATConv(in_channels, hidden, heads=heads, concat=True, dropout=dropout))
        dim = hidden * heads
        # camadas intermediárias
        for _ in range(layers - 2):
            self.layers.append(GATConv(dim, hidden, heads=heads, concat=True, dropout=dropout))
            dim = hidden * heads
        # última camada (mantém concat para representação rica)
        self.layers.append(GATConv(dim, hidden, heads=heads, concat=True, dropout=dropout))
        dim = hidden * heads
        self.out_dim = dim // 2
        self.head = _ReadoutHead(dim, self.out_dim, dropout)

    def forward(self, x, edge_index):
        h = x
        for conv in self.layers:
            h = conv(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.head(h)  # embedding final (one-class)
        return z

class OCGCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32, layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        # primeira
        self.layers.append(GCNConv(in_channels, hidden))
        # intermediárias
        for _ in range(layers - 2):
            self.layers.append(GCNConv(hidden, hidden))
        # última
        self.layers.append(GCNConv(hidden, hidden))
        self.out_dim = hidden // 2
        self.head = _ReadoutHead(hidden, self.out_dim, dropout)

    def forward(self, x, edge_index):
        h = x
        for conv in self.layers:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.head(h)
        return z

class OCGraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32, layers: int = 2, agg: str = "mean", dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_channels, hidden, aggr=agg))
        for _ in range(layers - 2):
            self.layers.append(SAGEConv(hidden, hidden, aggr=agg))
        self.layers.append(SAGEConv(hidden, hidden, aggr=agg))
        self.out_dim = hidden // 2
        self.head = _ReadoutHead(hidden, self.out_dim, dropout)

    def forward(self, x, edge_index):
        h = x
        for conv in self.layers:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.head(h)
        return z
