# src/models/baseline/torch_mlp.py
from __future__ import annotations

from typing import Iterable, Optional, Tuple, Dict, Any
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from sklearn.base import BaseEstimator, ClassifierMixin


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _MLPNet(nn.Module):
    """MLP de 2 camadas: in -> hidden -> 2, com dropout=0.5 e log_softmax na saída."""
    def __init__(self, in_channels: int, hidden_channels: int = 50, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 2)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # NLLLoss-ready


class TorchMLP(BaseEstimator, ClassifierMixin):
    """
    Estimador sklearn-like que treina um MLP PyTorch para classificação binária.

    Hiperparâmetros principais (defaults iguais ao que você pediu):
    - hidden_channels=50
    - dropout=0.5
    - lr=1e-3
    - epochs=200
    - class_weights=(0.3, 0.7)
    - batch_size=None (usa full-batch). Pode passar um int, ex: 4096
    - device: "cuda" se disponível, senão "cpu"
    - seed: para reprodutibilidade
    """

    def __init__(
        self,
        hidden_channels: int = 50,
        dropout: float = 0.5,
        lr: float = 1e-3,
        epochs: int = 200,
        class_weights: Tuple[float, float] = (0.3, 0.7),
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        self.hidden_channels = int(hidden_channels)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.class_weights = tuple(class_weights)
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.verbose = verbose

        # Atributos definidos após fit()
        self._net: Optional[_MLPNet] = None
        self._criterion: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    # ----------------------------
    # API sklearn
    # ----------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "hidden_channels": self.hidden_channels,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "class_weights": self.class_weights,
            "batch_size": self.batch_size,
            "device": self.device,
            "seed": self.seed,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "TorchMLP":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ----------------------------
    # Treino
    # ----------------------------
    def fit(self, X: np.ndarray, y: Iterable[int]) -> "TorchMLP":
        _set_seed(self.seed)

        X = self._to_numpy_float32(X)
        y = np.asarray(list(y), dtype=np.int64)

        # Guarda metadados sklearn-like
        self.n_features_in_ = X.shape[1]
        # Garante classes_ = [0, 1] na mesma ordem usada em NLLLoss
        self.classes_ = np.array([0, 1], dtype=np.int64)

        device = torch.device(self.device) if self.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tensors
        X_t = torch.from_numpy(X).to(device)
        y_t = torch.from_numpy(y).to(device)

        # Modelo + otimizador + loss
        self._net = _MLPNet(in_channels=X.shape[1], hidden_channels=self.hidden_channels, dropout=self.dropout).to(device)
        w0, w1 = map(float, self.class_weights)
        self._criterion = nn.NLLLoss(weight=torch.tensor([w0, w1], dtype=torch.float32, device=device))
        self._optimizer = Adam(self._net.parameters(), lr=self.lr)

        # Loop de treino
        self._net.train()
        batch_size = self.batch_size or X_t.shape[0]  # full-batch por default
        n = X_t.shape[0]

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            # mini-batch
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = X_t[start:end]
                yb = y_t[start:end]

                self._optimizer.zero_grad()
                log_probs = self._net(xb)  # N x 2 (log-prob)
                loss = self._criterion(log_probs, yb)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item() * (end - start)

            if self.verbose and (epoch == 1 or epoch % max(1, self.epochs // 4) == 0):
                print(f"[TorchMLP] Epoch {epoch:4d}/{self.epochs} | loss={epoch_loss / n:.4f}")

        return self

    # ----------------------------
    # Predição
    # ----------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._to_numpy_float32(X)

        device = next(self._net.parameters()).device  # type: ignore
        X_t = torch.from_numpy(X).to(device)

        self._net.eval()  # type: ignore
        with torch.no_grad():
            log_probs = self._net(X_t)  # type: ignore
            probs = torch.exp(log_probs).cpu().numpy()
        return probs  # colunas na ordem [class 0, class 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    # ----------------------------
    # Utilidades
    # ----------------------------
    def _check_is_fitted(self) -> None:
        if self._net is None or self.classes_ is None or self.n_features_in_ is None:
            raise RuntimeError("TorchMLP não foi treinado. Chame .fit(X, y) primeiro.")

    @staticmethod
    def _to_numpy_float32(X: Any) -> np.ndarray:
        # aceita DataFrame, array-like, etc.
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Esperado X 2D, recebido shape={X.shape}")
        return X


__all__ = ["TorchMLP"]
