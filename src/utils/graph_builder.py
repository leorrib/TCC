from pathlib import Path
import torch
from torch_geometric.data import Data
import pandas as pd


class GraphBuilder:
    """
    Builds a sequence of PyTorch Geometric (PyG) Data objects from node and edge DataFrames.
    Each time_step becomes one graph snapshot (as in the Elliptic dataset).
    """

    def __init__(
        self,
        df_nodes: pd.DataFrame,
        df_edges: pd.DataFrame,
        feature_cols: list[str],
        output_dir: str | Path,
        exclude_steps: list[int] | None = None,
    ):
        self.df_nodes = df_nodes
        self.df_edges = df_edges
        self.feature_cols = feature_cols
        self.output_dir = Path(output_dir)
        self.exclude_steps = set(exclude_steps or [])
        self.graphs = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # Main methods
    # ==========================================================

    def build_graphs(self) -> list[Data]:
        """Creates one PyG graph per time_step and stores them in memory."""
        graphs = []

        for t in sorted(self.df_nodes["time_step"].unique()):
            if t in self.exclude_steps:
                print(f"⏭️ Skipping time_step {t} (excluded)")
                continue

            df_t = self.df_nodes[self.df_nodes["time_step"] == t]

            # Restrict edges to nodes within the same time_step
            edges_t = self.df_edges[
                self.df_edges["txId1"].isin(df_t["txId"]) &
                self.df_edges["txId2"].isin(df_t["txId"])
            ].copy()

            # Local mapping (PyG requires contiguous indices)
            local_map = {tx: i for i, tx in enumerate(df_t["txId"])}
            edges_t["txId1"] = edges_t["txId1"].map(local_map)
            edges_t["txId2"] = edges_t["txId2"].map(local_map)
            edges_t = edges_t.dropna(subset=["txId1", "txId2"]).astype(int)

            # Edge index (bidirectional)
            edge_index = torch.tensor(
                [edges_t["txId1"].values, edges_t["txId2"].values],
                dtype=torch.long
            )
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

            # Node features and labels
            x = torch.tensor(df_t[self.feature_cols].values, dtype=torch.float)
            y = torch.tensor(df_t["class"].fillna(-1).astype(int).values, dtype=torch.long)

            # Build PyG Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            data.time_step = t
            data.mask_labeled = (y != -1)

            graphs.append(data)

        self.graphs = graphs
        print(f"✅ Created {len(graphs)} graphs (time_steps {graphs[0].time_step}–{graphs[-1].time_step})")
        return graphs

    def save_graphs(self, filename: str = "elliptic_graphs.pt"):
        """Saves all built graphs to a .pt file."""
        if not self.graphs:
            raise ValueError("No graphs built. Run build_graphs() first.")

        save_path = self.output_dir / filename
        torch.save(self.graphs, save_path)
        print(f"💾 Saved {len(self.graphs)} graphs to {save_path.resolve()}")

    def run(self, filename: str = "elliptic_graphs.pt"):
        """Convenience method: build and save graphs in one call."""
        self.build_graphs()
        self.save_graphs(filename)
