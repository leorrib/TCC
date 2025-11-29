# src/models/baseline/factory.py
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from .torch_mlp import TorchMLP


class BaselineFactory:
    """
    Constrói modelos baseline a partir de um dicionário de parâmetros
    ou diretamente de um cfg (YAML carregado).

    Suporta:
      - "logreg"     -> LogisticRegression (com StandardScaler)
      - "rf"         -> RandomForestClassifier (sem scaler)
      - "mlp"        -> MLPClassifier (sklearn, com StandardScaler)
      - "torch_mlp"  -> TorchMLP (implementação PyTorch, com StandardScaler na pipeline)
    """

    # --------------------------
    # API principal
    # --------------------------
    @staticmethod
    def from_cfg(model_key: str, cfg: Dict[str, Any], seed: Optional[int] = None):
        """
        Lê hiperparâmetros de cfg["models"]["baseline"][model_key] (se existir)
        e cria o estimador correspondente.

        Exemplo de YAML esperado:
        models:
          baseline:
            logreg:
              C: 1.0
              max_iter: 2000
            rf:
              n_estimators: 50
              max_features: 50
              class_weight: balanced
              n_jobs: -1
            mlp:
              hidden_layer_sizes: [256, 128]
              activation: relu
              alpha: 0.0001
              max_iter: 100
            torch_mlp:
              hidden_channels: 50
              dropout: 0.5
              lr: 0.001
              epochs: 200
              class_weights: [0.3, 0.7]
              batch_size: null
        """
        params = (
            cfg.get("models", {})
               .get("baseline", {})
               .get(model_key, {})
            or {}
        )
        return BaselineFactory.build(model_key, params, seed=seed)

    @staticmethod
    def build(model_key: str, params: Dict[str, Any], seed: Optional[int] = None):
        model_key = model_key.lower().strip()
        if model_key == "logreg":
            return _make_logreg(params, seed)
        elif model_key == "rf":
            return _make_rf(params, seed)
        elif model_key == "mlp":
            return _make_mlp_sklearn(params, seed)
        elif model_key == "torch_mlp":
            return _make_torch_mlp(params, seed)
        else:
            raise ValueError(f"Modelo desconhecido: {model_key}")


# ==========================================================
# Implementações concretas
# ==========================================================
def _make_logreg(params: Dict[str, Any], seed: Optional[int]):
    """
    Logistic Regression + StandardScaler (pipeline).
    """
    C = float(params.get("C", 1.0))
    max_iter = int(params.get("max_iter", 2000))
    penalty = params.get("penalty", "l2")
    solver = params.get("solver", "lbfgs")  # lbfgs lida bem com muitos features
    class_weight = params.get("class_weight", "balanced")

    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        penalty=penalty,
        solver=solver,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=params.get("n_jobs", None),
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", clf),
    ])
    return pipe


def _make_rf(params: Dict[str, Any], seed: Optional[int]):
    """
    Random Forest (sem scaler).
    """
    n_estimators = int(params.get("n_estimators", 50))
    max_features = params.get("max_features", 50)  # pode ser int, float, str
    class_weight = params.get("class_weight", "balanced")
    n_jobs = int(params.get("n_jobs", -1))

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=seed,
    )
    # RF não precisa de scaler
    return clf


def _make_mlp_sklearn(params: Dict[str, Any], seed: Optional[int]):
    """
    MLPClassifier do sklearn + StandardScaler (pipeline).
    Pensado para alto número de colunas.
    """
    hidden_layer_sizes = params.get("hidden_layer_sizes", [256, 128])
    activation = params.get("activation", "relu")
    alpha = float(params.get("alpha", 1e-4))
    batch_size = params.get("batch_size", "auto")
    learning_rate = params.get("learning_rate", "adaptive")
    learning_rate_init = float(params.get("learning_rate_init", 1e-3))
    max_iter = int(params.get("max_iter", 100))
    early_stopping = bool(params.get("early_stopping", True))
    n_iter_no_change = int(params.get("n_iter_no_change", 10))

    clf = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        activation=activation,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        n_iter_no_change=n_iter_no_change,
        random_state=seed,
        verbose=False,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", clf),
    ])
    return pipe


def _make_torch_mlp(params: Dict[str, Any], seed: Optional[int]):
    """
    TorchMLP + StandardScaler (pipeline).
    - Mantém API sklearn-like com .fit/.predict/.predict_proba
    """
    hidden_channels = int(params.get("hidden_channels", 50))
    dropout = float(params.get("dropout", 0.5))
    lr = float(params.get("lr", 1e-3))
    epochs = int(params.get("epochs", 200))
    class_weights = params.get("class_weights", (0.3, 0.7))
    batch_size = params.get("batch_size", None)
    device = params.get("device", None)  # "cuda" | "cpu" | None (auto)
    verbose = bool(params.get("verbose", False))

    torch_est = TorchMLP(
        hidden_channels=hidden_channels,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        class_weights=tuple(class_weights),
        batch_size=batch_size,
        device=device,
        seed=seed,
        verbose=verbose,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", torch_est),
    ])
    return pipe


__all__ = ["BaselineFactory"]