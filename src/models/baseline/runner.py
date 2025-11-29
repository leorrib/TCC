# src/models/baseline/runner.py
from __future__ import annotations

from typing import Dict, List, Any

import numpy as np
import pandas as pd

from .factory import BaselineFactory
from .evaluator import BaselineEvaluator


class BaselineRunner:
    """
    Orquestra os baselines (LogReg, RF, MLP/torch_mlp) sobre múltiplos
    conjuntos de features e várias seeds, retornando um dicionário
    pronto para persistência no ResultsStore.

    Espera que o evaluator exponha:
      - by_timestep(clf, df, feats) -> List[dict]
      - global_metrics(clf, list_of_dfs, feats) -> Dict[str, float]
    """

    def __init__(self, cfg: Dict[str, Any], seeds: List[int]) -> None:
        self.cfg = cfg
        self.seeds = seeds

    # ------------------------------------------------------
    # Público
    # ------------------------------------------------------
    def run_all(
        self,
        feature_sets: Dict[str, List[str]],
        df_train: pd.DataFrame,
        df_test1: pd.DataFrame,
        df_test2: pd.DataFrame,
        include_torch_mlp: bool = False,
    ) -> Dict[str, Dict[str, list]]:
        """
        Executa todas as famílias pedidas. Retorna:
        {
          "logreg": { "Model (FS)": [runs...] },
          "rf":     { "Model (FS)": [runs...] },
          "mlp":    { "Model (FS)": [runs...] },
          "torch_mlp": { ... } (opcional)
        }
        """
        out: Dict[str, Dict[str, list]] = {
            "logreg": self.run_family("logreg", feature_sets, df_train, df_test1, df_test2),
            "rf":     self.run_family("rf",     feature_sets, df_train, df_test1, df_test2),
            "mlp":    self.run_family("mlp",    feature_sets, df_train, df_test1, df_test2),
        }
        if include_torch_mlp:
            out["torch_mlp"] = self.run_family("torch_mlp", feature_sets, df_train, df_test1, df_test2)
        return out

    def run_family(
        self,
        model_key: str,
        feature_sets: Dict[str, List[str]],
        df_train: pd.DataFrame,
        df_test1: pd.DataFrame,
        df_test2: pd.DataFrame,
    ) -> Dict[str, list]:
        """
        Roda uma família (ex.: 'rf') em todos os feature_sets e seeds.

        Retorna: { "RF (Local)": [ {seed, ...}, ... ], "RF (1-hop)": [...], ... }
        """
        family_results: Dict[str, list] = {}

        for fs_name, feats in feature_sets.items():
            model_name = self._model_display_name(model_key, fs_name)
            runs = []
            for sd in self.seeds:
                clf = self._fit_one(model_key, sd, df_train, feats)
                # métricas por time_step (test1+test2)
                by_ts = BaselineEvaluator.by_timestep(clf, pd.concat([df_test1, df_test2], ignore_index=True), feats)
                # métricas globais (pré / pós / global)
                m_pre  = BaselineEvaluator.global_metrics(clf, [df_test1], feats)
                m_post = BaselineEvaluator.global_metrics(clf, [df_test2], feats)
                m_glob = BaselineEvaluator.global_metrics(clf, [df_test1, df_test2], feats)

                runs.append({
                    "seed": sd,
                    "por_time_step": by_ts,                 # lista de dicts (time_step -> métricas)
                    "Global (35–42)": m_pre,
                    "Global (43–49)": m_post,
                    "Global (35–49)": m_glob,
                })

            family_results[model_name] = runs

        return family_results

    # ------------------------------------------------------
    # Privados
    # ------------------------------------------------------
    def _fit_one(
        self,
        model_key: str,
        seed: int,
        df_train: pd.DataFrame,
        feats: List[str],
    ):
        """
        Cria e ajusta um modelo em df_train usando apenas as colunas 'feats'.
        Mantém a ordem e os nomes das colunas para compatibilidade com sklearn.
        """
        clf = BaselineFactory.from_cfg(model_key, self.cfg, seed=seed)

        # garante que só as colunas 'feats' (na ordem correta) sejam usadas no fit
        X_tr = df_train[feats]
        y_tr = df_train["class"].to_numpy()

        clf.fit(X_tr, y_tr)
        return clf

    @staticmethod
    def _model_display_name(model_key: str, fs_name: str) -> str:
        key_map = {
            "logreg": "LogReg",
            "rf": "RF",
            "mlp": "MLP",
            "torch_mlp": "Torch-MLP",
        }
        prefix = key_map.get(model_key.lower(), model_key.upper())
        return f"{prefix} ({fs_name})"
