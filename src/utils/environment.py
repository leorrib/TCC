"""
environment.py
---------------
Provides environment setup utilities for reproducibility in PyTorch
and PyTorch Geometric experiments.
"""

import os
import random
import warnings
import numpy as np
import torch


class EnvironmentSetup:
    """
    Handles environment configuration, random seed setup, and device selection (CPU/GPU).

    Example
    -------
    >>> env = EnvironmentSetup(seed=42)
    >>> device = env.device
    """

    def __init__(self, seed: int = 42, verbose: bool = True):
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set all random seeds for reproducibility
        self.set_seed_all(seed)

        # Optional CUDA workspace config (may be ignored on some systems)
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

        # Suppress non-critical warnings
        warnings.filterwarnings("ignore")

        if verbose:
            self._print_env_info()

    def set_seed_all(self, seed: int):
        """Sets random seeds for Python, NumPy, and PyTorch (CPU and GPU)."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Enforce deterministic behavior for reproducibility
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _print_env_info(self):
        """Displays current environment information."""
        print(f"✅ Active device: {self.device}")
        print(f"GPU detected: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
        print(f"Torch version: {torch.__version__}")
