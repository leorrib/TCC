# src/models/oc/trainer.py
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from sklearn.metrics import roc_auc_score, average_precision_score

def _interp_weight_decay(epoch, total_epochs, wd_start, wd_end):
    # decaimento linear simples
    alpha = min(max(epoch / max(total_epochs, 1), 0.0), 1.0)
    return wd_start * (1 - alpha) + wd_end * alpha

class OneClassTrainer:
    """
    Treina OCGNN à la Deep SVDD:
      - aprende centro c (média de z dos normais nas primeiras épocas)
      - loss = ||z - c||^2 (apenas amostras 'normais' da classe normal_label)
      - optimizer: SGD(lr) + (opcional) weight decay manual (constante ou interpolado)
      - early stopping: combina loss (↓) e AUC (↑) no período de validação

    Compatibilidade de argumentos:
      - epochs: define epochs_min = epochs_max = epochs (atalho)
      - OU use epochs_min / epochs_max explicitamente
      - weight_decay: se passado, usa wd_start = wd_end = weight_decay
      - OU use wd_start / wd_end explicitamente
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        # compat: pode passar 'epochs' OU (epochs_min/epochs_max)
        epochs: int | None = None,
        epochs_min: int | None = 100,
        epochs_max: int | None = 1000,
        patience: int = 50,
        # compat: pode passar 'weight_decay' OU (wd_start/wd_end)
        weight_decay: float | None = None,
        wd_start: float = 5e-3,
        wd_end: float = 5e-6,
        dropout: float = 0.5,             # aceito só pra compat; não usado aqui
        center_warmup_epochs: int = 5,
        normal_label: int = 0,
        device: str | None = None,
        seed: int = 42,
        val_period: str = "pre",
    ):
        self.model = model
        self.lr = lr

        # ---- Compatibilidade para épocas ----
        if epochs is not None:
            # se 'epochs' foi fornecido, fixa min=max=epochs
            self.epochs_min = int(epochs)
            self.epochs_max = int(epochs)
        else:
            self.epochs_min = int(epochs_min if epochs_min is not None else 100)
            self.epochs_max = int(epochs_max if epochs_max is not None else 1000)

        self.patience = int(patience)

        # ---- Compatibilidade para weight decay ----
        if weight_decay is not None:
            self.wd_start = float(weight_decay)
            self.wd_end = float(weight_decay)
        else:
            self.wd_start = float(wd_start)
            self.wd_end = float(wd_end)

        self.center_warmup_epochs = int(center_warmup_epochs)
        self.normal_label = int(normal_label)
        self.val_period = val_period

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        np.random.seed(seed)

    @torch.no_grad()
    def _init_center(self, graphs):
        self.model.eval().to(self.device)
        zs = []
        for g in graphs:
            y = g.y.cpu().numpy()
            mask = (y == self.normal_label)
            if mask.sum() == 0:
                continue
            x = g.x.to(self.device)
            z = self.model(x, g.edge_index.to(self.device)).cpu().numpy()
            zs.append(z[mask])
        if len(zs) == 0:
            zdim = getattr(self.model, "out_dim", None)
            if zdim is None:
                # tenta deduzir uma vez
                x0 = graphs[0].x[:1].to(self.device)
                z0 = self.model(x0, graphs[0].edge_index.to(self.device))
                zdim = z0.size(1)
                self.model.out_dim = zdim
            return torch.zeros(int(zdim), dtype=torch.float32, device=self.device)
        c = torch.tensor(np.vstack(zs).mean(axis=0), dtype=torch.float32, device=self.device)
        return c

    def _one_epoch(self, graphs, center, optimizer, weight_decay):
        self.model.train()
        total = 0.0
        for g in graphs:
            y = g.y.cpu().numpy()
            mask = (y == self.normal_label)
            if mask.sum() == 0:
                continue
            x = g.x.to(self.device)
            z = self.model(x, g.edge_index.to(self.device))
            z_n = z[torch.from_numpy(mask).to(self.device)]
            loss = ((z_n - center) ** 2).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # aplica weight decay L2 manualmente (se > 0)
            if weight_decay and weight_decay > 0:
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.requires_grad and p.grad is not None:
                            p.grad.add_(p, alpha=weight_decay)
            optimizer.step()
            total += float(loss.item())
        return total / max(1, len(graphs))

    @torch.no_grad()
    def _eval_auc(self, graphs, center):
        # score = ||z - c|| (maior = mais anômalo). Classe positiva = 1 (fraude).
        self.model.eval()
        y_all, s_all = [], []
        for g in graphs:
            y = g.y.cpu().numpy()
            mask = (y != -1)
            if mask.sum() == 0:
                continue
            x = g.x.to(self.device)
            z = self.model(x, g.edge_index.to(self.device)).cpu().numpy()
            d = ((z - center.cpu().numpy()) ** 2).sum(axis=1) ** 0.5
            y_all.extend(y[mask])
            s_all.extend(d[mask])
        if len(y_all) == 0 or len(set(y_all)) < 2:
            return np.nan, np.nan
        y_arr, s_arr = np.array(y_all), np.array(s_all)
        try:
            auc = roc_auc_score(y_arr, s_arr)
            ap  = average_precision_score(y_arr, s_arr)
        except ValueError:
            auc, ap = np.nan, np.nan
        return auc, ap

    def fit(self, train_graphs, val_graphs):
        self.model.to(self.device)
        center = self._init_center(train_graphs)
        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.0)

        best_score = -np.inf
        best_state = None
        no_improve = 0
        history = []

        for epoch in range(1, self.epochs_max + 1):
            wd = _interp_weight_decay(epoch, self.epochs_max, self.wd_start, self.wd_end)
            train_loss = self._one_epoch(train_graphs, center, optimizer, wd)

            # reatualiza o centro nas primeiras épocas
            if epoch <= self.center_warmup_epochs:
                center = self._init_center(train_graphs)

            val_auc, val_ap = self._eval_auc(val_graphs, center)
            history.append(dict(epoch=epoch, loss=train_loss, val_auc=val_auc, val_ap=val_ap))

            # critério combinado (AUC ↑ e loss ↓)
            score = (0 if np.isnan(val_auc) else val_auc) - 0.1 * train_loss

            if (epoch >= self.epochs_min) and (score > best_score + 1e-9):
                best_score = score
                best_state = dict(
                    model=self.model.state_dict(),
                    center=center.detach().clone(),
                    epoch=epoch,
                    val_auc=val_auc,
                    val_ap=val_ap,
                )
                no_improve = 0
            else:
                no_improve += 1

            if (epoch >= self.epochs_min) and (no_improve >= self.patience):
                break

        # restaura melhor
        if best_state is not None:
            self.model.load_state_dict(best_state["model"])
            center = best_state["center"]

        return center, history
