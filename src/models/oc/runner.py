# src/models/oc/runner.py
from __future__ import annotations
import numpy as np, torch
from typing import Dict, List, Tuple
from .factory import OCFactory
from .trainer import OneClassTrainer

def split_by_period(graphs, pre_range=(35,42), post_range=(43,49)):
    train = [g for g in graphs if 1 <= int(g.time_step) <= 34]
    pre   = [g for g in graphs if pre_range[0] <= int(g.time_step) <= pre_range[1]]
    post  = [g for g in graphs if post_range[0] <= int(g.time_step) <= post_range[1]]
    return train, pre, post

def run_ocgnn_family(graphs, in_channels: int, cfg: dict, kinds=("ocgat","ocgcn","ocgraphsage")):
    results = {}
    models  = {}
    centers = {}

    train_graphs, pre_graphs, post_graphs = split_by_period(graphs)
    val_graphs = pre_graphs  # validação no pré (paper)

    seeds = cfg["models"]["ocgnn"]["common"]["seeds"]
    common = cfg["models"]["ocgnn"]["common"]

    for kind in kinds:
        runs = []
        for i in range(seeds):
            seed = cfg.get("general",{}).get("seed",42) + i
            model = OCFactory.build(kind, in_channels, cfg)
            trainer = OneClassTrainer(
                model=model,
                lr=common["lr"],
                epochs_min=common["epochs_min"],
                epochs_max=common["epochs_max"],
                patience=common["patience"],
                wd_start=common["wd_start"],
                wd_end=common["wd_end"],
                dropout=common["dropout"],
                center_warmup_epochs=common["center_warmup_epochs"],
                normal_label=common["normal_label"],
                device=None,
                seed=seed,
                val_period=common["val_period"],
            )
            center, hist = trainer.fit(train_graphs, val_graphs)

            # Avaliação final (pré, pós, global)
            auc_pre, ap_pre   = trainer._eval_auc(pre_graphs, center)
            auc_post, ap_post = trainer._eval_auc(post_graphs, center)
            auc_glob, ap_glob = trainer._eval_auc(pre_graphs + post_graphs, center)

            runs.append({
                "seed": seed,
                "history": hist,
                "pre":   {"ROC-AUC": float(auc_pre),  "PR-AUC": float(ap_pre)},
                "post":  {"ROC-AUC": float(auc_post), "PR-AUC": float(ap_post)},
                "global":{"ROC-AUC": float(auc_glob), "PR-AUC": float(ap_glob)},
            })
            models[f"{kind}_seed{seed}"] = model
            centers[f"{kind}_seed{seed}"] = center.detach().cpu()

        results[kind] = runs

    return results, models, centers
