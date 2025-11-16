import torch
import torch.nn.functional as F

def build_knn_edge_index(x: torch.Tensor, k: int = 15, batch_size: int = 4096, device=None):
    if device is None:
        device = x.device

    x = F.normalize(x, dim=1)
    N = x.size(0)
    knn_src = []
    knn_dst = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_batch = x[start:end]             # [B, F]
        sim = torch.matmul(x_batch, x.t())  # [B, N]
        vals, idxs = torch.topk(sim, k=k+1, dim=1)
        for i, neighbors in enumerate(idxs):
            src = start + i
            for dst in neighbors.tolist():
                if dst == src:
                    continue
                knn_src.append(src)
                knn_dst.append(dst)

    edge_index_knn = torch.tensor([knn_src, knn_dst], dtype=torch.long, device=device)
    edge_index_knn = torch.cat([edge_index_knn, edge_index_knn.flip(0)], dim=1)
    return edge_index_knn
