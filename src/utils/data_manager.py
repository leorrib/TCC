"""
data_manager.py
----------------
Utilities for managing the Elliptic dataset:
- Download via KaggleHub (saved directly in data/raw/)
- Load raw CSVs from data/raw/
- Prepare processed CSVs (preserving column names/order)
- Quick DataFrame inspection
"""

import shutil
from pathlib import Path
import kagglehub
import pandas as pd
import numpy as np


class DataManager:
    """Handles dataset downloading, storage, and inspection."""

    def __init__(self, root: Path, data_raw_path: str = "data/raw", verbose: bool = True):
        self.root = Path(root)
        self.data_raw = self.root / data_raw_path
        self.data_raw.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        if verbose:
            print(f"📂 DataManager initialized. Raw data path: {self.data_raw.resolve()}")

    # ---------------------------------------------------------
    # Download & Load (no subdir — saves directly in data/raw/)
    # ---------------------------------------------------------
    def download_elliptic(self):
        """Downloads the Elliptic dataset from KaggleHub and saves directly into data/raw/."""
        dst = self.data_raw
        dst.mkdir(parents=True, exist_ok=True)

        print("⬇️ Downloading Elliptic dataset from KaggleHub...")
        src_path = Path(kagglehub.dataset_download("ellipticco/elliptic-data-set"))

        # If KaggleHub creates a single nested folder, enter it
        subdirs = [p for p in src_path.iterdir() if p.is_dir()]
        src_root = subdirs[0] if len(subdirs) == 1 else src_path

        for item in src_root.iterdir():
            target = dst / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

        if self.verbose:
            print(f"✅ Dataset saved to: {dst.resolve()}")
        return dst

    def load_elliptic(self):
        """Loads Elliptic dataset CSVs from data/raw/ into DataFrames (raw format)."""
        base = self.data_raw
        df_nodes = pd.read_csv(base / "elliptic_txs_features.csv", header=None)
        df_edges = pd.read_csv(base / "elliptic_txs_edgelist.csv")
        df_target = pd.read_csv(base / "elliptic_txs_classes.csv")

        if self.verbose:
            print(
                f"✅ Elliptic CSVs loaded:\n"
                f" - Nodes:  {df_nodes.shape}\n"
                f" - Edges:  {df_edges.shape}\n"
                f" - Target: {df_target.shape}"
            )
        return df_nodes, df_edges, df_target

    # ---------------------------------------------------------
    # Prepare (KEEP names/order) + Save processed
    # ---------------------------------------------------------
    def prepare_elliptic_dataset(
        self,
        df_nodes: pd.DataFrame,
        df_edges: pd.DataFrame,
        df_target: pd.DataFrame,
        save_processed: bool = True,
    ):
        """
        Prepares the Elliptic dataset while PRESERVING original column names and ordering:
        - df_nodes -> ['txId','time_step'] + feature_1..feature_N-1 (with txId remapped to integers)
        - df_edges -> keeps ['txId1','txId2'] (remapped)
        - df_target -> keeps ['txId','class'] (class mapped to -1/1/0)
        - df_nodes_with_class -> merge on 'txId'
        """
        # 1) Rename node columns exactly like your original notebook
        feature_cols = [f"feature_{idx}" for idx in range(1, df_nodes.shape[1] - 1)]
        df_nodes = df_nodes.copy()
        df_nodes.columns = ["txId", "time_step"] + feature_cols

        # 2) Create mapping and overwrite txId with incremental ints
        id_map = {tid: i for i, tid in enumerate(df_nodes["txId"])}
        df_nodes["txId"] = df_nodes["txId"].map(id_map).astype(int)

        # 3) Remap edges in-place, keep names and order
        df_edges = df_edges.copy()
        if not {"txId1", "txId2"}.issubset(df_edges.columns):
            raise ValueError("df_edges must contain 'txId1' and 'txId2'.")
        df_edges["txId1"] = df_edges["txId1"].map(id_map)
        df_edges["txId2"] = df_edges["txId2"].map(id_map)
        df_edges = df_edges.dropna(subset=["txId1", "txId2"]).astype({"txId1": int, "txId2": int})
        df_edges = df_edges[["txId1", "txId2"]]

        # 4) Remap target, keep names
        df_target = df_target.copy()
        if not {"txId", "class"}.issubset(df_target.columns):
            raise ValueError("df_target must contain 'txId' and 'class'.")
        df_target["txId"] = df_target["txId"].map(id_map)
        df_target["class"] = df_target["class"].map({"unknown": -1, "1": 1, "2": 0}).astype(int)
        df_target = df_target[["txId", "class"]]

        # 5) Merge target into nodes (keep txId)
        df_nodes_with_class = df_nodes.merge(
            df_target.drop_duplicates("txId", keep="last"),
            on="txId",
            how="left",
        )

        if self.verbose:
            print(
                f"✅ Elliptic dataset prepared (names/order preserved):\n"
                f" - Nodes:               {df_nodes.shape}\n"
                f" - Edges:               {df_edges.shape}\n"
                f" - Target:              {df_target.shape}\n"
                f" - Nodes with class:    {df_nodes_with_class.shape}"
            )

        # 6) Save processed CSVs
        if save_processed:
            processed_dir = self.root / "data" / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)

            df_nodes.to_csv(processed_dir / "elliptic_nodes.csv", index=False)
            df_edges.to_csv(processed_dir / "elliptic_edges.csv", index=False)
            df_target.to_csv(processed_dir / "elliptic_target.csv", index=False)
            df_nodes_with_class.to_csv(processed_dir / "elliptic_nodes_with_class.csv", index=False)

            if self.verbose:
                print(f"💾 Saved processed CSVs to: {processed_dir.resolve()}")

        return df_nodes, df_edges, df_target, df_nodes_with_class

    # ---------------------------------------------------------
    # Quick inspection
    # ---------------------------------------------------------
    @staticmethod
    def quick_df_report(
        df: pd.DataFrame,
        top_nulls: int = 10,
        target_col: str | None = None,
        cat_counts_max_unique: int = 20,
    ) -> None:
        print(f"Shape: {df.shape}")
        has_nulls = df.isnull().values.any()
        print(f"Contains null values? {has_nulls}")
        if has_nulls:
            null_pct = (df.isnull().mean().sort_values(ascending=False) * 100)
            print("\nTop columns by % of nulls:")
            print(null_pct.head(top_nulls).round(2).astype(str) + "%")

        print("\nDtypes summary:")
        print(df.dtypes.value_counts())

        dup_rows = df.duplicated().sum()
        print(f"\nDuplicate rows: {dup_rows}")

        nunq = df.nunique(dropna=False)
        const_cols = nunq[nunq <= 1].index.tolist()
        print(f"Constant columns (nunique<=1): {const_cols}")

        num = df.select_dtypes(include=[np.number])
        if not num.empty:
            inf_mask = np.isinf(num)
            if inf_mask.values.any():
                cols_inf = num.columns[inf_mask.any()].tolist()
                print(f"Columns with ±inf: {cols_inf}")
            else:
                print("Columns with ±inf: None")

            minmax = pd.DataFrame({"min": num.min(), "max": num.max()})
            print("\nNumeric ranges (first 10 columns):")
            print(minmax.head(10))

        obj = df.select_dtypes(include=["object", "category"])
        if not obj.empty:
            card = (obj.nunique(dropna=False) / len(df)).sort_values(ascending=False)
            high_card = card[card > 0.9].index.tolist()
            print(f"\nHigh-cardinality categoricals (>90% unique): {high_card[:10]}")
            low_card_cols = [c for c in obj.columns if obj[c].nunique(dropna=False) <= cat_counts_max_unique]
            if low_card_cols:
                print(f"\nValue counts (categoricals with ≤ {cat_counts_max_unique} categories):")
                for c in low_card_cols:
                    vc = df[c].value_counts(dropna=False)
                    pct = (vc / len(df) * 100).round(2).astype(str) + '%'
                    out = pd.DataFrame({"count": vc, "pct": pct})
                    print(f"\n[{c}]")
                    print(out)

        if target_col and target_col in df.columns:
            print(f"\nTarget distribution for '{target_col}':")
            print(df[target_col].value_counts(dropna=False))
