from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import torch
from torch_geometric.data import Data

@dataclass
class TimeSplits:
    train_upper: int
    test1_lower: int
    test1_upper: int
    test2_lower: int

class GraphTimeBuilder:
    """
    Constrói um grafo PyG por time_step e devolve splits temporais.
    Mantém os índices locais contíguos por step.
    """

    def __init__(self, df_nodes_with_class: pd.DataFrame, df_edges: pd.DataFrame, feature_cols: List[str]):
        self.df_nodes = df_nodes_with_class
        self.df_edges = df_edges
        self.feature_cols = feature_cols

    def build_by_timestep(self) -> Dict[int, Data]:
        graphs_by_t: Dict[int, Data] = {}
        for t in sorted(self.df_nodes["time_step"].unique()):
            df_nodes_t = self.df_nodes[self.df_nodes["time_step"] == t]
            txids = df_nodes_t["txId"].values
            id_map = {tx: i for i, tx in enumerate(txids)}

            mask_edges = self.df_edges["txId1"].isin(txids) & self.df_edges["txId2"].isin(txids)
            df_e = self.df_edges[mask_edges]
            src = df_e["txId1"].map(id_map)
            dst = df_e["txId2"].map(id_map)

            edge_index = torch.tensor([src.values, dst.values], dtype=torch.long)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # não-direcionado

            x = torch.tensor(df_nodes_t[self.feature_cols].values, dtype=torch.float)
            y = torch.tensor(df_nodes_t["class"].values, dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            data.mask_labeled = (y != -1)
            data.time_step = int(t)
            graphs_by_t[int(t)] = data
        return graphs_by_t

    @staticmethod
    def split(graphs_by_t: Dict[int, Data], splits: TimeSplits) -> Tuple[List[Data], List[Data], List[Data]]:
        train = [graphs_by_t[t] for t in sorted(graphs_by_t) if t <= splits.train_upper]
        test1 = [graphs_by_t[t] for t in sorted(graphs_by_t) if splits.test1_lower <= t <= splits.test1_upper]
        test2 = [graphs_by_t[t] for t in sorted(graphs_by_t) if t >= splits.test2_lower]
        return train, test1, test2
