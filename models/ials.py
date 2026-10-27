"""Implicit ALS with fixed confidence on every observed rating."""

import numpy as np
from scipy import sparse


class IALS:
    def __init__(self, factors=32, reg=10.0, alpha=20.0, iterations=12, seed=42):
        if factors < 1 or iterations < 1 or reg <= 0 or alpha < 0:
            raise ValueError("invalid ALS parameters")
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def _targets(self, ratings, timestamps=None, cutoff=None):
        return (ratings.data >= 4).astype(np.float64)

    def _update(self, preferences, fixed):
        base = fixed.T @ fixed + self.reg * np.eye(self.factors)
        updated = np.zeros((preferences.shape[0], self.factors))
        for row in range(preferences.shape[0]):
            start, stop = preferences.indptr[row : row + 2]
            if start == stop:
                continue
            observed = fixed[preferences.indices[start:stop]]
            lhs = base + self.alpha * (observed.T @ observed)
            rhs = (1.0 + self.alpha) * (observed.T @ preferences.data[start:stop])
            updated[row] = np.linalg.solve(lhs, rhs)
        return updated

    def fit(self, ratings, timestamps=None, cutoff=None):
        ratings = sparse.csr_matrix(ratings, dtype=np.float64).copy()
        ratings.sort_indices()
        if np.any(~np.isfinite(ratings.data)) or np.any(
            (ratings.data < 1) | (ratings.data > 5)
        ):
            raise ValueError("ratings must contain observed scores from 1 to 5")
        preferences = ratings.copy()
        preferences.data = self._targets(ratings, timestamps, cutoff)
        # Keep explicit zeros: disliked observations still have confidence 1 + alpha.
        by_item = preferences.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.1, (ratings.shape[1], self.factors))
        for _ in range(self.iterations):
            self.user_factors = self._update(preferences, self.item_factors)
            self.item_factors = self._update(by_item, self.user_factors)
        return self

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
