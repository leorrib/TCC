# src/models/gcpal/safe_split_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce


@dataclass
class SafeSplitBuilder:
    """
    Builds safe, self-contained PyG Data objects for a given time_step range,
    using a contiguous local node index (required by PyG) and removing any
    edges that cross the split boundary.

    It replicates exactly the logic you used in your notebook:
      - filter nodes by [lo, hi]
      - local reindex (contiguous)
      - keep only internal edges and reindex them
      - undirect + coalesce
      - add masks/checks and keep txId index for later merges
    """
    df_nodes_with_class: pd.DataFrame
    df_edges: pd.DataFrame
    feature_cols: List[str]

    def make_split_data_safe(self, lo: int, hi: int) -> Data:
        """Create a single PyG Data for time_step in [lo, hi] with local indexing."""
        # 1) filter nodes in range and keep only needed columns
        df_nodes_s = self.df_nodes_with_class[
            (self.df_nodes_with_class["time_step"] >= lo) &
            (self.df_nodes_with_class["time_step"] <= hi)
        ][["txId", *self.feature_cols, "class"]].copy()

        # 2) local contiguous mapping (sorted by txId for determinism)
        df_nodes_s = df_nodes_s.sort_values("txId").reset_index(drop=True)
        txids_np = df_nodes_s["txId"].to_numpy()
        txid_to_local = {int(t): i for i, t in enumerate(txids_np)}

        # 3) keep internal edges only and reindex, no NaN
        e1 = self.df_edges["txId1"].to_numpy()
        e2 = self.df_edges["txId2"].to_numpy()
        mask = np.isin(e1, txids_np) & np.isin(e2, txids_np)
        e1, e2 = e1[mask], e2[mask]

        # vectorized reindex via dict (fast enough for this size)
        src = np.fromiter((txid_to_local[int(x)] for x in e1), dtype=np.int64, count=len(e1))
        dst = np.fromiter((txid_to_local[int(x)] for x in e2), dtype=np.int64, count=len(e2))

        # 4) tensors: features, labels, edges
        x = torch.tensor(df_nodes_s[self.feature_cols].to_numpy(), dtype=torch.float)
        y = torch.tensor(df_nodes_s["class"].to_numpy(), dtype=torch.long)
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

        # 5) undirected + coalesce (remove duplicates, sort)
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        edge_index, _ = coalesce(edge_index, None, num_nodes=x.size(0))

        # 6) sanity checks (will raise early if something is off)
        N = x.size(0)
        assert edge_index.numel() > 0, "edge_index is empty"
        assert int(edge_index.min()) >= 0
        assert int(edge_index.max()) < N
        assert torch.isfinite(x).all(), "x contains NaN/Inf"

        # 7) build Data with mask and original txId index for merges
        data = Data(x=x, edge_index=edge_index, y=y)
        data.mask_labeled = (y != -1)
        data.df_index = df_nodes_s[["txId"]].reset_index(drop=True)  # keep txId for later joins
        data.time_range = (lo, hi)
        return data

    def build_train_test_splits(
        self,
        train_lo: int = 1, train_hi: int = 34,
        test1_lo: int = 35, test1_hi: int = 42,
        test2_lo: int = 43, test2_hi: int = 49,
    ) -> Tuple[Data, Data, Data]:
        """Convenience: build (train, test1, test2) PyG Data using standard ranges."""
        data_train = self.make_split_data_safe(train_lo, train_hi)
        data_test1 = self.make_split_data_safe(test1_lo, test1_hi)
        data_test2 = self.make_split_data_safe(test2_lo, test2_hi)
        return data_train, data_test1, data_test2
