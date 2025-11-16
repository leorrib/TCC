from collections import defaultdict
import torch

def build_positive_lists(edge_index: torch.Tensor, num_nodes: int, add_self: bool = True):
    if edge_index.is_cuda:
        edge_index = edge_index.detach().cpu()

    pos = defaultdict(set)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    for s, d in zip(src, dst):
        pos[s].add(d)

    pos_lists = []
    for i in range(num_nodes):
        neigh = pos[i]
        if add_self:
            neigh = set(neigh)
            neigh.add(i)
        pos_lists.append(list(neigh))

    return pos_lists
