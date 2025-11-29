from __future__ import annotations
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

class SkipGCN(nn.Module):
    """
    Duas camadas GCN + skip connection linear da entrada para a saída.
    Retorna log-probabilidades (log_softmax) para NLLLoss.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 100, out_channels: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.skip = nn.Linear(in_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        out = self.conv2(h1, edge_index) + self.skip(x)
        return F.log_softmax(out, dim=1)
