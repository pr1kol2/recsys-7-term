"""Check formulas on small matrices with known answers."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
import torch

from models.custom_als import CustomALS
from models.ease import EASE
from models.fm import FM, _FM, _sample_negatives
from models.ials import IALS
from models.item_knn import ItemKNN


def test_als_update_matches_dense_weighted_least_squares():
    ratings = sparse.csr_matrix([[5., 1., 0., 0.], [0., 4., 2., 5.], [0., 0., 0., 0.]])
    model = IALS(factors=2, alpha=7., reg=0.3, iterations=1)
    preferences = ratings.copy()
    preferences.data = model._targets(ratings)
    fixed = np.array([[0.1, 0.7], [0.5, -0.2], [1.1, 0.3], [-0.4, 0.6]])
    actual = model._update(preferences, fixed)
    expected = []
    for row in ratings.toarray():
        confidence = 1 + model.alpha * (row > 0)
        target = (row >= 4).astype(float)
        lhs = fixed.T @ (confidence[:, None] * fixed) + model.reg * np.eye(2)
        rhs = fixed.T @ (confidence * target)
        expected.append(np.linalg.solve(lhs, rhs))
    assert_allclose(actual, expected, atol=1e-12)
    # A low rating still has confidence > 1 when p=0.
    assert preferences.nnz == ratings.nnz
    assert np.count_nonzero(preferences.data) < preferences.nnz


def test_als_full_sweep_matches_dense_reference():
    ratings = sparse.csr_matrix([[5., 1., 0.], [0., 4., 2.], [3., 0., 5.]])
    model = IALS(factors=2, alpha=3., reg=0.7, iterations=1, seed=9)
    initial = np.random.default_rng(9).normal(0, .1, (3, 2))
    confidence = 1 + model.alpha * (ratings.toarray() > 0)
    target = (ratings.toarray() >= 4).astype(float)

    def solve_rows(c, p, fixed):
        return np.array([
            np.linalg.solve(fixed.T @ (row[:, None] * fixed) + .7 * np.eye(2),
                            fixed.T @ (row * wanted))
            for row, wanted in zip(c, p)
        ])

    expected_users = solve_rows(confidence, target, initial)
    expected_items = solve_rows(confidence.T, target.T, expected_users)
    model.fit(ratings)
    assert_allclose(model.user_factors, expected_users, atol=1e-12)
    assert_allclose(model.item_factors, expected_items, atol=1e-12)
    assert_allclose(model.score([2, 0]), expected_users[[2, 0]] @ expected_items.T)


def test_custom_target_changes_preference_and_preserves_observations():
    cutoff = 10 * 86400.
    ratings = sparse.csr_matrix([[1., 3., 5., 4.]])
    timestamps = sparse.csr_matrix([[cutoff, cutoff, cutoff - 86400., cutoff - 2 * 86400.]])
    model = CustomALS(half_life_days=1.)
    target = model._targets(ratings, timestamps, cutoff)
    # Age multiplier: 1 now, .75 after one day, .625 after two days.
    assert_allclose(target, [-1., 0., .75, .3125])
    assert ratings.nnz == len(target)


def test_ease_matches_constrained_ridge_regressions():
    matrix = sparse.csr_matrix([[1., 0., 1., 0.], [1., 1., 0., 0.],
                                [0., 1., 1., 1.], [1., 0., 0., 1.]])
    model = EASE(reg=0.7).fit(matrix)
    gram = (matrix.T @ matrix).toarray()
    expected = np.zeros((4, 4))
    for target in range(4):
        sources = np.arange(4) != target
        expected[sources, target] = np.linalg.solve(
            gram[np.ix_(sources, sources)] + .7 * np.eye(3), gram[sources, target])
    assert_allclose(model.weights, expected, rtol=1e-6, atol=1e-7)
    assert_array_equal(model.weights.diagonal(), 0)
    assert_allclose(model.score([0, 3]), matrix[[0, 3]] @ expected, rtol=1e-6, atol=1e-7)
    gradient = (gram + .7 * np.eye(4)) @ model.weights - gram
    assert_allclose(gradient[~np.eye(4, dtype=bool)], 0, atol=2e-7)


def test_item_knn_keeps_nearest_sources_for_each_candidate():
    matrix = sparse.csr_matrix([[1., 1., 0., 0.], [1., 0., 1., 0.],
                                [1., 1., 1., 0.], [0., 0., 1., 1.]])
    model = ItemKNN(k=1, shrinkage=2.).fit(matrix)
    overlap = (matrix.T @ matrix).toarray()
    counts = np.diag(overlap)
    similarities = overlap / np.sqrt(counts[:, None] * counts[None, :])
    similarities *= overlap / (overlap + 2.)
    np.fill_diagonal(similarities, 0)
    expected = np.zeros_like(similarities)
    for target in range(4):
        source = np.lexsort((np.arange(4), -similarities[:, target]))[0]
        expected[source, target] = similarities[source, target]
    assert_allclose(model.similarity.toarray(), expected, atol=1e-7)
    assert_allclose(model.score([1]), matrix[[1]] @ expected, atol=1e-7)


def test_fm_forward_equals_explicit_pairwise_sum():
    features = sparse.csr_matrix([[.5, 1., 0.], [0., 0., 2.]], dtype=np.float32)
    torch.manual_seed(7)
    module = _FM(2, features, factors=3)
    with torch.no_grad():
        module.bias.fill_(.2)
        module.user_linear.weight.copy_(torch.tensor([[.3], [-.1]]))
        module.item_linear.weight.copy_(torch.tensor([[.1], [.4]]))
        module.meta_linear.weight[:3].copy_(torch.tensor([[.2], [.5], [-.3]]))
    expected = []
    for user, item in [(0, 0), (1, 1)]:
        vectors = [module.user_factors.weight[user], module.item_factors.weight[item]]
        linear = module.bias + module.user_linear.weight[user, 0] + module.item_linear.weight[item, 0]
        for feature in features[item].indices:
            value = float(features[item, feature])
            vectors.append(value * module.meta_factors.weight[feature])
            linear = linear + value * module.meta_linear.weight[feature, 0]
        interactions = sum(torch.dot(vectors[i], vectors[j])
                           for i in range(len(vectors)) for j in range(i + 1, len(vectors)))
        expected.append((linear + interactions).item())
    with torch.no_grad():
        actual = module(torch.tensor([0, 1]), torch.tensor([0, 1])).numpy()
    assert_allclose(actual, expected, atol=1e-7)


def test_fm_cached_scores_equal_pair_predictions_after_training():
    torch.set_num_threads(1)
    ratings = sparse.csr_matrix([[5., 1., 0., 0.], [0., 4., 2., 0.], [3., 0., 0., 5.]])
    features = sparse.csr_matrix([[1., 0.], [.5, .5], [0., 1.], [1., 0.]])
    for use_metadata in (True, False):
        model = FM(factors=3, epochs=3, batch_size=4, seed=3,
                   use_metadata=use_metadata).fit(ratings, features)
        user_ids = np.array([2, 0])
        users = torch.from_numpy(np.repeat(user_ids, 4))
        items = torch.arange(4).repeat(2)
        with torch.no_grad():
            expected = model.module_(users, items).numpy().reshape(2, 4)
        assert_allclose(model.score(user_ids), expected, atol=1e-7)
        assert np.isfinite(model.history_).all()


def test_negative_sampler_excludes_low_ratings_and_full_rows():
    ratings = sparse.csr_matrix([[5., 1., 2., 0.], [1., 2., 3., 4.], [0., 4., 0., 0.]])
    users, items = _sample_negatives(ratings, ratio=4, rng=np.random.default_rng(5))
    assert len(users) == 4 * (3 + 1)
    assert 1 not in users  # This user has rated every item.
    assert_array_equal(items[users == 0], 3)
    assert not np.asarray(ratings[users, items]).any()
