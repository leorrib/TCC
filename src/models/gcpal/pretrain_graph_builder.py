from pathlib import Path
import torch
import pandas as pd
from torch_geometric.data import Data

class PretrainGraphBuilder:
    """
    Builds the single global training graph (time_step <= MAX_TRAIN_STEP)
    using exactly the same logic as in your notebook.
    """

    def __init__(self, df_nodes_with_class: pd.DataFrame, df_edges: pd.DataFrame, feature_cols: list[str], max_train_step: int = 34, device: str | torch.device | None = None):
        self.df_nodes_with_class = df_nodes_with_class
        self.df_edges = df_edges
        self.feature_cols = feature_cols
        self.max_train_step = int(max_train_step)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.data_train_global: Data | None = None

    def build(self) -> Data:
        # 1) Filter nodes up to step 34
        df_train_nodes = self.df_nodes_with_class[self.df_nodes_with_class["time_step"] <= self.max_train_step].copy()

        # 2) Node id set
        train_node_ids = set(df_train_nodes["txId"].tolist())

        # 3) Keep edges fully inside the training set
        df_train_edges = self.df_edges[
            self.df_edges["txId1"].isin(train_node_ids) &
            self.df_edges["txId2"].isin(train_node_ids)
        ].copy()

        # 4) Undirected edge_index
        edge_index_train = torch.tensor(
            [df_train_edges["txId1"].values, df_train_edges["txId2"].values],
            dtype=torch.long
        )
        edge_index_train = torch.cat([edge_index_train, edge_index_train.flip(0)], dim=1)

        # 5) Features and labels
        x_train_nodes = torch.tensor(df_train_nodes[self.feature_cols].values, dtype=torch.float)
        y_train_nodes = torch.tensor(df_train_nodes["class"].values, dtype=torch.long)

        # 6) Mask of labeled nodes
        mask_labeled_train = (y_train_nodes != -1)

        # 7) Single Data object (as in notebook)
        data_train_global = Data(x=x_train_nodes, edge_index=edge_index_train, y=y_train_nodes)
        data_train_global.mask_labeled = mask_labeled_train

        self.data_train_global = data_train_global

        # Prints (same as notebook)
        print(data_train_global)
        print(f"Total de nós (≤{self.max_train_step}): {data_train_global.num_nodes}")
        print(f"Total de arestas (bidirecionais): {data_train_global.num_edges}")
        print(f"Nós rotulados: {int(mask_labeled_train.sum())}")
        prop_ilic = float((y_train_nodes[mask_labeled_train] == 1).float().mean())
        print("Proporção de ilícitos nos rotulados:", prop_ilic)

        return data_train_global

    def save(self, path: str | Path) -> None:
        if self.data_train_global is None:
            raise ValueError("Call build() before save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.data_train_global, path)
        print(f"💾 Saved pretrain Data to: {path.resolve()}")
