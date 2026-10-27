"""Parameter search and final model comparison."""
import gc
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from data_utils import interaction_matrix, positive_matrix
from evaluation import evaluate
from models import ItemKNN, IALS, EASE, CustomALS
from models.fm import FM

SEED = 42
GRIDS = {
    'ItemKNN': [dict(k=k, shrinkage=s) for k in (50, 150) for s in (0, 20)],
    'iALS': [dict(factors=d, reg=r, alpha=20, iterations=15, seed=SEED)
             for d in (32, 64) for r in (1, 10)],
    'EASE': [dict(reg=r) for r in (50, 200, 800)],
    'FM': [dict(factors=d, lr=.01, weight_decay=w, epochs=12,
                batch_size=4096, negative_ratio=1, seed=SEED)
           for d in (16, 32) for w in (1e-5, 1e-4)],
    'CustomALS': [dict(factors=d, reg=r, alpha=20, iterations=15,
                       half_life_days=365, seed=SEED)
                  for d in (32, 64) for r in (1, 10)],
}
GRIDS['FM IDs'] = [dict(params, use_metadata=False) for params in GRIDS['FM']]


def fit_model(name, params, ratings, timestamps, positive, features, cutoff):
    classes = {'ItemKNN': ItemKNN, 'iALS': IALS, 'EASE': EASE,
               'FM': FM, 'FM IDs': FM, 'FM IDs matched': FM,
               'CustomALS': CustomALS, 'CustomALS matched': CustomALS}
    model = classes[name](**params)
    start = perf_counter()
    if name in ('ItemKNN', 'EASE'):
        model.fit(positive)
    elif name.startswith('FM'):
        model.fit(ratings, features)
    else:
        model.fit(ratings, timestamps, cutoff)
    return model, perf_counter() - start


def training_matrices(frame, shape):
    return (interaction_matrix(frame, shape), interaction_matrix(frame, shape, 'timestamp'),
            positive_matrix(frame, shape))


def tune(train, shape, features, cutoff, users, relevant, history, output='results'):
    output = Path(output)
    output.mkdir(exist_ok=True)
    ratings, timestamps, positive = training_matrices(train, shape)
    rows, selected = [], {}
    for name, grid in GRIDS.items():
        trials = []
        for number, params in enumerate(grid):
            model, seconds = fit_model(name, params, ratings, timestamps, positive, features, cutoff)
            metrics, _ = evaluate(model, users, relevant, history)
            row = dict(model=name, trial=number, params=json.dumps(params, sort_keys=True),
                       fit_seconds=seconds, **metrics)
            rows.append(row)
            trials.append(row)
            print(f'{name:10} {number + 1}/{len(grid)}  NDCG={metrics["NDCG@10"]:.5f}  '
                  f'Recall={metrics["Recall@10"]:.5f}  fit={seconds:.1f}s', flush=True)
            del model
            gc.collect()
        best = max(trials, key=lambda row: (row['NDCG@10'], row['Recall@10'], -row['trial']))
        selected[name] = json.loads(best['params'])
        pd.DataFrame(rows).to_csv(output / 'validation.csv', index=False)
    # Matched variants keep the original model's parameters.
    selected['CustomALS matched'] = dict(selected['iALS'], half_life_days=365)
    selected['FM IDs matched'] = dict(selected['FM'], use_metadata=False)
    (output / 'selected_params.json').write_text(json.dumps(selected, indent=2) + '\n')
    return pd.DataFrame(rows), selected


def final_comparison(selected, train_valid, shape, features, cutoff,
                     users, relevant, history, output='results'):
    output = Path(output)
    ratings, timestamps, positive = training_matrices(train_valid, shape)
    rows, per_user, losses = [], {}, {}
    for name, params in selected.items():
        model, seconds = fit_model(name, params, ratings, timestamps, positive, features, cutoff)
        start = perf_counter()
        metrics, individual = evaluate(model, users, relevant, history)
        rows.append(dict(model=name, **metrics, fit_seconds=seconds,
                         evaluation_seconds=perf_counter() - start,
                         params=json.dumps(params, sort_keys=True)))
        per_user[name] = individual
        # Keep per-user metrics local and out of git.
        private = Path('data/processed')
        private.mkdir(parents=True, exist_ok=True)
        individual.to_csv(private / f'{name.replace(" ", "_")}_test.csv', index=False)
        if hasattr(model, 'history_'):
            losses[name] = model.history_
        print(f'{name:18} NDCG={metrics["NDCG@10"]:.5f}  Recall={metrics["Recall@10"]:.5f}', flush=True)
        del model
        gc.collect()
    results = pd.DataFrame(rows).set_index('model')
    results.to_csv(output / 'test.csv')
    (output / 'fm_losses.json').write_text(json.dumps(losses, indent=2) + '\n')
    return results, per_user, losses
