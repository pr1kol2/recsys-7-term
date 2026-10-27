"""Top-k over the full catalog, with metrics averaged across users."""
import numpy as np
import pandas as pd
from scipy import sparse


def evaluation_set(future, history, shape, k=10):
    relevant = sparse.csr_matrix(
        (np.ones(int((future.rating >= 4).sum())),
         (future.loc[future.rating >= 4, 'user'], future.loc[future.rating >= 4, 'item'])),
        shape=shape)
    relevant = relevant - relevant.multiply(history.astype(bool))
    relevant.eliminate_zeros()
    enough_candidates = shape[1] - history.getnnz(axis=1) >= k
    users = np.flatnonzero((relevant.getnnz(axis=1) > 0) & enough_candidates)
    return users, relevant


def top_k(scores, seen, k=10):
    scores = np.asarray(scores).copy()
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite model scores')
    rows, columns = seen.nonzero()
    scores[rows, columns] = -np.inf
    # Break score ties by placing the smaller item index first.
    ranking = np.argsort(-scores, axis=1, kind='stable')[:, :k]
    if not np.isfinite(np.take_along_axis(scores, ranking, axis=1)).all():
        raise ValueError('Fewer than k unseen candidates')
    return ranking


def metric_rows(ranking, relevant, users):
    k = ranking.shape[1]
    relevance = relevant[users[:, None], ranking].toarray() > 0
    totals = np.asarray(relevant[users].getnnz(axis=1))
    discounts = 1 / np.log2(np.arange(2, k + 2))
    dcg = (relevance * discounts).sum(axis=1)
    ideal = np.cumsum(discounts)[np.minimum(totals, k) - 1]
    return pd.DataFrame({'user': users,
                         'NDCG@10': dcg / ideal,
                         'Recall@10': relevance.sum(axis=1) / totals,
                         'HitRate@10': relevance.any(axis=1).astype(float)})


def evaluate(model, users, relevant, history, batch_size=256, k=10):
    frames, recommendations = [], []
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        ranking = top_k(model.score(batch), history[batch], k=k)
        frames.append(metric_rows(ranking, relevant, batch))
        recommendations.append(ranking)
    if not frames:
        raise ValueError('No users with evaluable positives')
    per_user = pd.concat(frames, ignore_index=True)
    ranking = np.concatenate(recommendations)
    summary = per_user.drop(columns='user').mean().to_dict()
    summary['Coverage@10'] = np.unique(ranking).size / history.shape[1]
    summary['users'] = len(users)
    return summary, per_user


def paired_bootstrap(first, second, metric='NDCG@10', seed=42, repeats=2000):
    paired = first[['user', metric]].merge(second[['user', metric]], on='user',
                                           suffixes=('_a', '_b'), validate='one_to_one')
    assert len(paired) == len(first) == len(second)
    delta = (paired[f'{metric}_a'] - paired[f'{metric}_b']).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)])
    return {'difference': delta.mean(), 'ci_low': np.quantile(means, .025),
            'ci_high': np.quantile(means, .975)}
