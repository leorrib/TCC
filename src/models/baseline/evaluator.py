# src/models/baseline/evaluator.py
from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)


class BaselineEvaluator:
    @staticmethod
    def _metrics_from_scores(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas a partir de probabilidades e rótulos."""
        # Segurança contra arrays vazios
        if y_true.size == 0:
            return {
                "Precision": np.nan, "Recall": np.nan, "F1": np.nan,
                "Gini": np.nan, "PR-AUC": np.nan,
                "Prevalência": np.nan, "PR-AUC/Prev": np.nan,
            }

        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1        = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc  = average_precision_score(y_true, y_prob)
            gini    = 2 * roc_auc - 1
        except ValueError:
            roc_auc = pr_auc = gini = np.nan

        prev = float(np.mean(y_true)) if y_true.size > 0 else np.nan
        pr_over_prev = (pr_auc / prev) if (prev and not np.isnan(prev) and pr_auc is not np.nan and prev > 0) else np.nan

        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Gini": gini,
            "PR-AUC": pr_auc,
            "Prevalência": prev,
            "PR-AUC/Prev": pr_over_prev,
        }

    @staticmethod
    def by_timestep(clf, df: pd.DataFrame, feats: List[str]) -> List[dict]:
        """
        Calcula métricas por time_step usando SOMENTE linhas rotuladas (class >= 0).
        Retorna lista de dicts: [{ time_step, precision, recall, f1, roc_auc, pr_auc, gini, prevalencia, pr_auc_over_prev }, ...]
        """
        out = []
        if "time_step" not in df.columns:
            return out

        # Apenas rotulados
        dfl = df[df["class"] >= 0].copy()
        if dfl.empty:
            return out

        for ts, sub in dfl.groupby("time_step"):
            X = sub[feats]
            y = sub["class"].to_numpy()
            # Alguns modelos (torch wrapper) expõem predict_proba; garantimos isso no factory.
            prob = clf.predict_proba(X)[:, 1]
            yhat = (prob >= 0.5).astype(int)

            m = BaselineEvaluator._metrics_from_scores(y, prob, yhat)
            out.append({
                "time_step": int(ts),
                "precision": m["Precision"],
                "recall": m["Recall"],
                "f1": m["F1"],
                "roc_auc": m["Gini"] / 2 + 0.5 if m["Gini"] is not np.nan else np.nan,  # opcional
                "pr_auc": m["PR-AUC"],
                "gini": m["Gini"],
                "prevalencia": m["Prevalência"],
                "pr_auc_over_prev": m["PR-AUC/Prev"],
            })

        out = sorted(out, key=lambda x: x["time_step"])
        return out

    @staticmethod
    def global_metrics(clf, list_of_dfs: List[pd.DataFrame], feats: List[str]) -> Dict[str, float]:
        """
        Junta os DFs (ex.: pré/pós/global), filtra rotulados, calcula métricas globais.
        """
        if not list_of_dfs:
            return {k: np.nan for k in ["Precision","Recall","F1","Gini","PR-AUC","Prevalência","PR-AUC/Prev"]}

        dfc = pd.concat(list_of_dfs, ignore_index=True)
        dfl = dfc[dfc["class"] >= 0].copy()
        if dfl.empty:
            return {k: np.nan for k in ["Precision","Recall","F1","Gini","PR-AUC","Prevalência","PR-AUC/Prev"]}

        X = dfl[feats]
        y = dfl["class"].to_numpy()
        prob = clf.predict_proba(X)[:, 1]
        yhat = (prob >= 0.5).astype(int)

        return BaselineEvaluator._metrics_from_scores(y, prob, yhat)
