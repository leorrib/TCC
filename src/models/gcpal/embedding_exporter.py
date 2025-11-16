# src/models/gcpal/embedding_exporter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.decomposition import PCA


EMB_PREFIX = "emb_gcpal_"
PCA_PREFIX = "emb_pca_gcpal_"


@dataclass
class EmbeddingExporter:
    """
    Export utilities for *GCPAL* embeddings:
      - fixed column prefixes: emb_gcpal_* and emb_pca_gcpal_*
      - GPU extraction with safe CPU fallback
      - DataFrame builders aligned by txId
      - Optional PCA (fit on train, transform test)
      - CSV saving helpers
    """

    # ----- Extraction -----
    @staticmethod
    def extract_H_safe(encoder: torch.nn.Module, data_split: Data, device: torch.device) -> np.ndarray:
        encoder.eval()
        with torch.no_grad():
            try:
                return encoder(
                    data_split.x.to(device),
                    data_split.edge_index.to(device),
                ).detach().cpu().numpy()
            except RuntimeError as e:
                print("⚠️ GPU failed during embedding extraction; temporary CPU fallback...", str(e)[:160], "...")
                enc_cpu = encoder.to("cpu")
                H = enc_cpu(data_split.x, data_split.edge_index).detach().cpu().numpy()
                encoder.to(device)
                return H

    # ----- DataFrame builders -----
    @staticmethod
    def to_dataframe(E: np.ndarray, data_split: Data) -> pd.DataFrame:
        d = E.shape[1]
        emb_cols = [f"{EMB_PREFIX}{i}" for i in range(d)]
        df = pd.DataFrame(E, columns=emb_cols)
        df["txId"] = data_split.df_index["txId"].values
        return df

    @staticmethod
    def to_dataframes(
        E_train: np.ndarray, E_test1: np.ndarray, E_test2: np.ndarray,
        data_train: Data, data_test1: Data, data_test2: Data,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            EmbeddingExporter.to_dataframe(E_train, data_train),
            EmbeddingExporter.to_dataframe(E_test1, data_test1),
            EmbeddingExporter.to_dataframe(E_test2, data_test2),
        )

    # ----- PCA -----
    @staticmethod
    def pca_fit_transform(
        E_train: np.ndarray,
        n_components: int = 30,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, PCA]:
        pca = PCA(n_components=n_components, random_state=random_state)
        E_train_pca = pca.fit_transform(E_train)
        return E_train_pca, pca

    @staticmethod
    def pca_transform(pca: PCA, E: np.ndarray) -> np.ndarray:
        return pca.transform(E)

    @staticmethod
    def pca_dataframes(
        E_train_pca: np.ndarray, E_test1_pca: np.ndarray, E_test2_pca: np.ndarray,
        data_train: Data, data_test1: Data, data_test2: Data,
        start_index: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        d = E_train_pca.shape[1]
        cols = [f"{PCA_PREFIX}{i}" for i in range(start_index, start_index + d)]

        def _df(E, split):
            df = pd.DataFrame(E, columns=cols)
            df["txId"] = split.df_index["txId"].values
            return df

        return _df(E_train_pca, data_train), _df(E_test1_pca, data_test1), _df(E_test2_pca, data_test2)

    # ----- Merge with labeled splits -----
    @staticmethod
    def merge_with_splits(
        df_train: pd.DataFrame, df_test1: pd.DataFrame, df_test2: pd.DataFrame,
        emb_train_df: pd.DataFrame, emb_test1_df: pd.DataFrame, emb_test2_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_train_emb = df_train.merge(emb_train_df, on="txId", how="inner")
        df_test1_emb = df_test1.merge(emb_test1_df, on="txId", how="inner")
        df_test2_emb = df_test2.merge(emb_test2_df, on="txId", how="inner")
        df_test_emb = pd.concat([df_test1_emb, df_test2_emb], ignore_index=True)
        return df_train_emb, df_test1_emb, df_test2_emb, df_test_emb

    # ----- Saving -----
    @staticmethod
    def save_csvs(
        out_dir,
        df_train_emb: pd.DataFrame,
        df_test1_emb: pd.DataFrame,
        df_test2_emb: pd.DataFrame,
        df_train_pca: Optional[pd.DataFrame] = None,
        df_test1_pca: Optional[pd.DataFrame] = None,
        df_test2_pca: Optional[pd.DataFrame] = None,
        make_parents: bool = True,
    ) -> None:
        # out_dir is expected to be a pathlib.Path
        if make_parents:
            out_dir.mkdir(parents=True, exist_ok=True)

        df_train_emb.to_csv(out_dir / "train_embeddings_gcpal.csv", index=False)
        df_test1_emb.to_csv(out_dir / "test1_embeddings_gcpal.csv", index=False)
        df_test2_emb.to_csv(out_dir / "test2_embeddings_gcpal.csv", index=False)

        if df_train_pca is not None:
            df_train_pca.to_csv(out_dir / "train_embeddings_pca_gcpal.csv", index=False)
        if df_test1_pca is not None:
            df_test1_pca.to_csv(out_dir / "test1_embeddings_pca_gcpal.csv", index=False)
        if df_test2_pca is not None:
            df_test2_pca.to_csv(out_dir / "test2_embeddings_pca_gcpal.csv", index=False)
