import torch
import torch.nn.functional as F
from torch_geometric.data import Data

def drop_edges(edge_index: torch.Tensor, drop_prob: float):
    E = edge_index.size(1)
    device = edge_index.device
    mask = torch.rand(E, device=device) > drop_prob
    return edge_index[:, mask]

def drop_features(x: torch.Tensor, drop_prob: float):
    if drop_prob <= 0.0:
        return x
    device = x.device
    N, Fdim = x.size()
    mask = torch.rand(N, Fdim, device=device) > drop_prob
    return x * mask

def make_random_views(data: Data, drop_p_edge=0.3, drop_p_feat=0.3):
    x1 = drop_features(data.x, drop_p_feat)
    e1 = drop_edges(data.edge_index, drop_p_edge)
    x2 = drop_features(data.x, drop_p_feat)
    e2 = drop_edges(data.edge_index, drop_p_edge)
    return (x1, e1), (x2, e2)
