from pathlib import Path
import torch
import pandas as pd


class DataSplitter:
    """
    Splits PyG graphs and/or tabular node data into training and testing subsets,
    using temporal ranges from the configuration file.
    """

    def __init__(self, cfg: dict, df_nodes_with_class: pd.DataFrame | None = None):
        """
        Args:
            cfg (dict): Configuration loaded from YAML.
            df_nodes_with_class (pd.DataFrame, optional): Node dataframe with time_step & class columns.
        """
        self.cfg = cfg
        self.df_nodes_with_class = df_nodes_with_class
        self.paths = cfg["paths"]
        self.splits = cfg["splits"]

        self.data_processed = Path(self.paths["data_processed"])
        self.graphs = None
        self.train_graphs, self.test_graphs_1, self.test_graphs_2 = [], [], []

    # ==========================================================
    # PyG Graph Handling
    # ==========================================================
    def load_graphs(self, filename: str = "elliptic_graphs.pt"):
        """Loads pre-built PyG graphs from disk."""
        path = self.data_processed / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        self.graphs = torch.load(path)
        print(f"✅ Loaded {len(self.graphs)} PyG graphs from {path}")
        return self.graphs

    def split_graphs_by_time(self):
        """Splits PyG graphs into train/test1/test2 based on YAML time ranges."""
        if self.graphs is None:
            raise ValueError("Graphs not loaded. Run load_graphs() first.")

        s = self.splits
        self.train_graphs = [g for g in self.graphs if g.time_step <= s["train_upper"]]
        self.test_graphs_1 = [g for g in self.graphs if s["test1_lower"] <= g.time_step <= s["test1_upper"]]
        self.test_graphs_2 = [g for g in self.graphs if g.time_step >= s["test2_lower"]]

        print(f"Train graphs: {len(self.train_graphs)} ({self.train_graphs[0].time_step}–{self.train_graphs[-1].time_step})")
        print(f"Test1 graphs: {len(self.test_graphs_1)} ({self.test_graphs_1[0].time_step}–{self.test_graphs_1[-1].time_step})")
        print(f"Test2 graphs: {len(self.test_graphs_2)} ({self.test_graphs_2[0].time_step}–{self.test_graphs_2[-1].time_step})")

        return self.train_graphs, self.test_graphs_1, self.test_graphs_2

    # ==========================================================
    # Tabular Data Splitting
    # ==========================================================
    def split_tabular_data(self):
        """Splits node dataframe into train/test subsets according to YAML time ranges."""
        if self.df_nodes_with_class is None:
            raise ValueError("df_nodes_with_class must be provided for tabular splits.")

        s = self.splits
        df = self.df_nodes_with_class[self.df_nodes_with_class["class"] >= 0].copy()

        df_train = df[df["time_step"] <= s["train_upper"]]
        df_test1 = df[(df["time_step"] >= s["test1_lower"]) & (df["time_step"] <= s["test1_upper"])]
        df_test2 = df[df["time_step"] >= s["test2_lower"]]
        df_test = pd.concat([df_test1, df_test2], ignore_index=True)

        print(f"Train nodes: {len(df_train)} | Test1: {len(df_test1)} | Test2: {len(df_test2)} | Total labeled: {len(df)}")
        return df_train, df_test1, df_test2, df_test

    # ==========================================================
    # CSV Export
    # ==========================================================
    def save_splits_to_csv(self, df_train, df_test1, df_test2):
        """Saves tabular splits (train/test1/test2) as CSV files."""
        out_dir = self.data_processed / "splits"
        out_dir.mkdir(parents=True, exist_ok=True)

        df_train.to_csv(out_dir / "train_nodes.csv", index=False)
        df_test1.to_csv(out_dir / "test1_nodes.csv", index=False)
        df_test2.to_csv(out_dir / "test2_nodes.csv", index=False)

        print(f"💾 Saved CSV splits to {out_dir.resolve()}")

    # ==========================================================
    # Combined Pipeline
    # ==========================================================
    def run(self, save_csv: bool = True):
        """Loads graphs, performs splits, and optionally saves CSVs."""
        self.load_graphs()
        self.split_graphs_by_time()

        if self.df_nodes_with_class is not None:
            df_train, df_test1, df_test2, df_test = self.split_tabular_data()
            if save_csv:
                self.save_splits_to_csv(df_train, df_test1, df_test2)
            return df_train, df_test1, df_test2, df_test

        return None
