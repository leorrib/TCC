import torch
import torch.nn.functional as F

def contrastive_loss_tiled(
    z_anchor,
    z_target,
    pos_lists=None,
    tau: float = 0.5,
    anchor_bs: int = 2048,
    target_bs: int = 32768,
    device=None,
):
    N = z_anchor.size(0)
    if device is None:
        device = z_anchor.device

    z_anchor = F.normalize(z_anchor, dim=1)
    z_target = F.normalize(z_target, dim=1)

    losses = []

    for a_start in range(0, N, anchor_bs):
        a_end = min(a_start + anchor_bs, N)
        idx_a = torch.arange(a_start, a_end, device=device)
        za = z_anchor[idx_a]  # [Ba, D]

        # 1) global max
        global_max = None
        for t_start in range(0, N, target_bs):
            t_end = min(t_start + target_bs, N)
            zt = z_target[t_start:t_end]  # [Bt, D]
            sim_chunk = torch.matmul(za, zt.t()) / tau
            chunk_max, _ = sim_chunk.max(dim=1, keepdim=True)
            global_max = chunk_max if global_max is None else torch.maximum(global_max, chunk_max)

        # 2) denom & num
        denom = torch.zeros((a_end - a_start,), device=device)
        num = torch.zeros((a_end - a_start,), device=device)

        for t_start in range(0, N, target_bs):
            t_end = min(t_start + target_bs, N)
            zt = z_target[t_start:t_end]
            sim_chunk = torch.matmul(za, zt.t()) / tau
            sim_chunk = sim_chunk - global_max
            exp_chunk = torch.exp(sim_chunk)
            denom = denom + exp_chunk.sum(dim=1)

            # positives in this chunk
            for local_i, i in enumerate(idx_a.tolist()):
                if pos_lists is None:
                    if t_start <= i < t_end:
                        pos_local = i - t_start
                        num[local_i] += exp_chunk[local_i, pos_local]
                else:
                    pos_i_global = pos_lists[i]
                    pos_in_this_chunk = [j for j in pos_i_global if t_start <= j < t_end]
                    if not pos_in_this_chunk:
                        continue
                    pos_local_idx = torch.tensor([j - t_start for j in pos_in_this_chunk], device=device, dtype=torch.long)
                    num[local_i] += exp_chunk[local_i, pos_local_idx].sum()

        eps = 1e-8
        loss_batch = -torch.log((num + eps) / (denom + eps))
        losses.append(loss_batch.mean())

    return torch.stack(losses).mean()
