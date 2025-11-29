# src/utils/visualization.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Visualization:
    """
    Plot utilities for model metrics across time steps.
    """

    @staticmethod
    def plot_pr_auc_with_fraud_bars(
        results_summary: Dict[str, List[dict]],
        df_test1: pd.DataFrame,
        df_test2: pd.DataFrame,
        model_names: Optional[List[str]] = None,
        shutdown_ts: Optional[float] = 42.5,
        title: str = "PR-AUC by time_step (Local vs 1-hop vs AF) + #frauds (TEST)",
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (12, 4),
        alpha_band: float = 0.15,
        bar_alpha: float = 0.25,
        show: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots PR-AUC curves (mean ± std across seeds) for given feature sets and
        overlays bars with number of frauds per test time_step.

        Parameters
        ----------
        results_summary : dict
            Dict produced by your training loop: {model_name: [run_dict, ...]}.
            Each run contains "por_time_step" (list of dicts with 'time_step', 'pr_auc', ...).
        df_test1, df_test2 : pd.DataFrame
            Test splits containing at least ['time_step', 'class'].
        model_names : list[str] | None
            Which models to plot. Defaults to ["Skip-GCN (Local)", "Skip-GCN (1-hop)", "Skip-GCN (AF)"].
        shutdown_ts : float | None
            If provided, draws a vertical dashed red line at this x position.
        title : str
            Plot title.
        ax : matplotlib.axes.Axes | None
            If provided, draw on this axes. Otherwise create a new figure/axes.
        figsize : (int, int)
            Figure size if creating a new figure.
        alpha_band : float
            Alpha for ±1 std fill area.
        bar_alpha : float
            Alpha for fraud bars.
        show : bool
            If True, calls plt.show() at the end.

        Returns
        -------
        fig, ax : Matplotlib figure and axes.
        """
        if model_names is None:
            model_names = ["Skip-GCN (Local)", "Skip-GCN (1-hop)", "Skip-GCN (AF)"]

        # X-axis: only test time steps
        time_axis = np.array(sorted(set(pd.concat([df_test1["time_step"], df_test2["time_step"]]).unique())), dtype=int)

        # Aggregate PR-AUC per time_step for each model
        agg_by_model = {}
        for model_name in model_names:
            runs = results_summary.get(model_name, [])
            if not runs:
                # silently skip missing model
                continue

            dfs = []
            for r in runs:
                df = pd.DataFrame(r["por_time_step"])
                df["seed"] = r.get("seed", None)
                dfs.append(df)

            df_all = pd.concat(dfs, ignore_index=True)
            df_all = df_all[df_all["time_step"].isin(time_axis)]

            agg = (
                df_all.groupby("time_step", as_index=False)
                      .agg(pr_auc_mean=("pr_auc", "mean"),
                           pr_auc_std =("pr_auc", "std"))
                      .sort_values("time_step")
            )
            # Reindex on full test axis
            agg = pd.DataFrame({"time_step": time_axis}).merge(agg, on="time_step", how="left")
            agg_by_model[model_name] = agg

        # Fraud bars from tests only
        df_all_test = pd.concat([df_test1, df_test2], ignore_index=True)
        frauds = (
            df_all_test[df_all_test["class"] == 1]
              .groupby("time_step", as_index=False)
              .size()
              .rename(columns={"size": "fraud_count"})
        )
        frauds_full = pd.DataFrame({"time_step": time_axis}).merge(frauds, on="time_step", how="left")
        frauds_full["fraud_count"] = frauds_full["fraud_count"].fillna(0).astype(int)

        # Figure / axes
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        ax2 = ax.twinx()

        # Curves
        for model_name, agg in agg_by_model.items():
            ax.plot(agg["time_step"], agg["pr_auc_mean"], label=model_name)
            std = agg["pr_auc_std"].fillna(0).to_numpy()
            y_mean = agg["pr_auc_mean"].interpolate().fillna(method="bfill").fillna(method="ffill")
            y1, y2 = (y_mean - std), (y_mean + std)
            ax.fill_between(agg["time_step"], y1, y2, alpha=alpha_band)

        # Bars
        bar = ax2.bar(frauds_full["time_step"], frauds_full["fraud_count"],
                      alpha=bar_alpha, width=0.8, label="# frauds (tests)")

        # Shutdown line
        if shutdown_ts is not None:
            ax.axvline(shutdown_ts, linestyle="--", linewidth=1.5, color="red",
                       label=f"Shutdown (t={shutdown_ts})")

        # Labels / title / legend
        ax.set_xlabel("time_step (tests only)")
        ax.set_ylabel("PR-AUC")
        ax2.set_ylabel("# frauds (bars)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        handles1, labels1 = ax.get_legend_handles_labels()
        handles, labels = handles1 + [bar], labels1 + ["# frauds (tests)"]
        ax.legend(handles, labels, loc="best")

        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax
