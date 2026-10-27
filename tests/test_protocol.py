"""Check metrics, candidates, and selection by time."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from data_utils import TRAIN_END, VALID_END, select_cohort
from evaluation import evaluate, evaluation_set, metric_rows, top_k


def test_metrics_on_hand_computed_rankings():
    ranking = np.tile(np.arange(10), (2, 1))
    relevant = sparse.csr_matrix((np.ones(5), ([0, 0, 0, 0, 1], [0, 2, 10, 11, 12])), shape=(2, 14))
    actual = metric_rows(ranking, relevant, np.array([0, 1]))
    expected_ndcg = (1 + 1 / np.log2(4)) / sum(1 / np.log2(np.arange(2, 6)))
    assert_allclose(actual['NDCG@10'], [expected_ndcg, 0])
    assert_allclose(actual['Recall@10'], [.5, 0])
    assert_allclose(actual['HitRate@10'], [1, 0])


def test_top_k_masks_every_known_rating_and_breaks_ties_by_item_id():
    history = sparse.csr_matrix(([5., 1.], ([0, 0], [0, 2])), shape=(1, 15))
    scores = np.ones((1, 15))
    assert_array_equal(top_k(scores, history), [[1, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
    assert_array_equal(scores, 1)
    scores[0, 14] = 100
    assert top_k(scores, history)[0, 0] == 14
    with pytest.raises(ValueError, match='Fewer than k'):
        top_k(np.ones((1, 11)), sparse.csr_matrix(np.ones((1, 11))), k=10)


def test_evaluation_users_require_future_positives_and_ten_unseen_items():
    history = sparse.csr_matrix((np.ones(9),
                                ([0] * 4 + [2] * 5, [0, 1, 2, 3, 0, 1, 2, 3, 4])),
                               shape=(4, 14))
    future = pd.DataFrame({'user': [0, 0, 0, 1, 2, 3],
                           'item': [0, 8, 9, 10, 10, 13],
                           'rating': [5, 4, 2, 3, 5, 5]})
    users, relevant = evaluation_set(future, history, history.shape)
    assert_array_equal(users, [0, 3])
    assert relevant[0, 0] == 0  # Exclude repeated interactions.
    assert relevant[0, 8] == 1 and relevant[0, 9] == 0


def test_evaluation_uses_full_catalog_and_macro_average():
    history = sparse.csr_matrix(([1., 5.], ([0, 1], [0, 1])), shape=(2, 20))
    relevant = sparse.csr_matrix(([1., 1., 1.], ([0, 1, 1], [19, 18, 2])), shape=(2, 20))

    class FixedScores:
        def score(self, users):
            return np.tile(np.arange(20, dtype=float), (len(users), 1))

    summary, rows = evaluate(FixedScores(), np.array([0, 1]), relevant, history, batch_size=1)
    assert summary['users'] == 2
    assert_allclose(rows['Recall@10'], [1., .5])
    assert_allclose(summary['Recall@10'], .75)
    assert_allclose(summary['NDCG@10'], rows['NDCG@10'].mean())
    assert_allclose(summary['Coverage@10'], .5)


def test_future_rows_do_not_change_training_cohort_or_mapping():
    users = np.repeat(np.arange(1000), 100)
    items = (users + np.tile(np.arange(100), 1000)) % 1000
    clean = pd.DataFrame({
        'user_id': [f'u{i:04d}' for i in users],
        'item_id': [f'i{i:04d}' for i in items],
        'rating': np.full(len(users), 5.),
        'timestamp': np.full(len(users), TRAIN_END - 86400),
    })
    base = select_cohort(clean)
    future = pd.DataFrame({
        'user_id': ['u0000', 'u0000', 'future-user', 'u0000'],
        'item_id': ['i0999', 'i0998', 'i0000', 'future-item'],
        'rating': [4., 5., 5., 5.],
        'timestamp': [TRAIN_END, VALID_END, VALID_END, VALID_END],
    })
    changed = select_cohort(pd.concat([clean, future], ignore_index=True))
    pd.testing.assert_frame_equal(base[1], changed[1])
    assert base[4] == changed[4] and base[5] == changed[5]
    assert len(changed[2]) == len(changed[3]) == 1
    assert changed[2].timestamp.iloc[0] == TRAIN_END
    assert changed[3].timestamp.iloc[0] == VALID_END
    assert changed[0].user.min() >= 0 and changed[0].item.min() >= 0
