from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from .model import SkipGCN

def _set_seed_all(seed: int) -> None:
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True

class SkipGCNTrainer:
    """
    Treina e avalia Skip-GCN em regime temporal (train 1..34, test1 35..42, test2 43..49).
    Retorna (modelo_treinado, dict_métricas).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 100,
        out_channels: int = 2,
        dropout: float = 0.5,
        lr: float = 1e-3,
        epochs: int = 1000,
        class_weights: Tuple[float, float] = (0.3, 0.7),
        device: torch.device | None = None,
    ):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.class_weights = class_weights
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _aggregate_metrics(graph_list, model, feats_idx, device) -> Dict[str, float] | None:
        all_y_true, all_y_pred, all_y_prob = [], [], []
        with torch.no_grad():
            for data in graph_list:
                data = data.to(device)
                x = data.x[:, feats_idx]
                out = model(x, data.edge_index)
                y_true = data.y.cpu().numpy()
                y_pred = out.argmax(dim=1).cpu().numpy()
                y_prob = torch.exp(out)[:, 1].cpu().numpy()
                mask = y_true != -1
                if mask.sum() == 0:
                    continue
                all_y_true.extend(y_true[mask])
                all_y_pred.extend(y_pred[mask])
                all_y_prob.extend(y_prob[mask])

        if len(all_y_true) == 0:
            return None

        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)
        y_prob = np.array(all_y_prob)

        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1        = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        roc_auc   = roc_auc_score(y_true, y_prob)
        pr_auc    = average_precision_score(y_true, y_prob)
        gini      = 2 * roc_auc - 1
        prev      = float(np.mean(y_true))
        pr_over_prev = pr_auc / prev if prev > 0 else np.nan

        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "gini": gini,
            "prevalencia": prev,
            "pr_auc_over_prev": pr_over_prev,
        }

    @staticmethod
    def _metrics_by_timestep(all_graphs, model, feats_idx, device) -> List[Dict[str, Any]]:
        rows = []
        with torch.no_grad():
            for data in all_graphs:
                data = data.to(device)
                x = data.x[:, feats_idx]
                out = model(x, data.edge_index)
                y_true = data.y.cpu().numpy()
                y_pred = out.argmax(dim=1).cpu().numpy()
                y_prob = torch.exp(out)[:, 1].cpu().numpy()
                mask = y_true != -1
                if mask.sum() == 0:
                    continue
                y_t, y_p, y_pb = y_true[mask], y_pred[mask], y_prob[mask]
                prec = precision_score(y_t, y_p, pos_label=1, zero_division=0)
                rec  = recall_score(y_t, y_p, pos_label=1, zero_division=0)
                f1   = f1_score(y_t, y_p, pos_label=1, zero_division=0)
                try:
                    roc_auc = roc_auc_score(y_t, y_pb)
                    pr_auc  = average_precision_score(y_t, y_pb)
                    gini    = 2 * roc_auc - 1
                except ValueError:
                    roc_auc = pr_auc = gini = np.nan
                prev = float(np.mean(y_t))
                pr_over_prev = pr_auc / prev if prev > 0 else np.nan
                rows.append({
                    "time_step": int(data.time_step),
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "gini": gini,
                    "prevalencia": prev,
                    "pr_auc_over_prev": pr_over_prev,
                })
        rows.sort(key=lambda r: r["time_step"])
        return rows

    def fit_evaluate(
        self,
        train_graphs: List[torch.Tensor],
        test_graphs_1: List[torch.Tensor],
        test_graphs_2: List[torch.Tensor],
        feats_idx: List[int],
        seed: int,
        print_interval: int = 250,
    ):
        _set_seed_all(seed)
        device = self.device

        model = SkipGCN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            dropout=self.dropout,
        ).to(device)

        class_weights = torch.tensor(self.class_weights, dtype=torch.float, device=device)
        criterion = nn.NLLLoss(weight=class_weights)
        optimizer = Adam(model.parameters(), lr=self.lr)

        model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for data in train_graphs:
                data = data.to(device)
                optimizer.zero_grad()
                x = data.x[:, feats_idx]
                out = model(x, data.edge_index)
                mask = data.y != -1
                loss = criterion(out[mask], data.y[mask])
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            if epoch % print_interval == 0 or epoch == 1 or epoch == self.epochs:
                avg = total_loss / max(1, len(train_graphs))
                print(f"Epoch {epoch:4d}/{self.epochs} | Loss: {avg:.4f}")

        model.eval()
        global_35_42 = self._aggregate_metrics(test_graphs_1, model, feats_idx, device)
        global_43_49 = self._aggregate_metrics(test_graphs_2, model, feats_idx, device)
        global_35_49 = self._aggregate_metrics(test_graphs_1 + test_graphs_2, model, feats_idx, device)
        by_step      = self._metrics_by_timestep(test_graphs_1 + test_graphs_2, model, feats_idx, device)

        result = {
            "Global (35–42)": global_35_42,
            "Global (43–49)": global_43_49,
            "Global (35–49)": global_35_49,
            "por_time_step": by_step,
            "seed": seed,
        }
        return model, result
