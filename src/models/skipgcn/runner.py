# src/models/skipgcn/runner.py
from .trainer import SkipGCNTrainer

def run_skipgcn_across_feature_sets(
    feature_sets: dict[str, list[int]],
    seeds: list[int],
    in_channels: int,  # mantemos na assinatura só por compatibilidade, mas não usamos
    train_graphs,
    test_graphs_1,
    test_graphs_2,
    trainer_kwargs: dict,
):
    results_summary = {}
    models_map = {}
    feats_map = {}

    for model_name, feats in feature_sets.items():
        runs = []
        for seed in seeds:
            # >>> FIX AQUI: modelo com a quantidade de features do subset <<<
            trainer = SkipGCNTrainer(in_channels=len(feats), **trainer_kwargs)

            mdl, result = trainer.fit_evaluate(
                train_graphs=train_graphs,
                test_graphs_1=test_graphs_1,
                test_graphs_2=test_graphs_2,
                feats_idx=feats,
                seed=seed,
            )
            runs.append(result)
            models_map[f"{model_name}_seed{seed}"] = mdl
            feats_map[f"{model_name}_seed{seed}"] = feats
        results_summary[model_name] = runs

    return results_summary, models_map, feats_map
