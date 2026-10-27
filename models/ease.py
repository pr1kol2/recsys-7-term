"""EASE: closed-form linear reconstruction with a zero diagonal."""

import numpy as np
from scipy import linalg, sparse


class EASE:
    def __init__(self, reg=100.0):
        if reg <= 0:
            raise ValueError("reg must be positive")
        self.reg = reg

    def fit(self, positive):
        self.interactions = sparse.csr_matrix(positive, dtype=np.float32, copy=True)
        self.interactions.eliminate_zeros()
        self.interactions.data[:] = 1.0
        gram = (self.interactions.T @ self.interactions).toarray().astype(np.float64)
        gram.flat[:: gram.shape[0] + 1] += self.reg
        inverse = linalg.cho_solve(
            linalg.cho_factor(gram, lower=True, check_finite=False),
            np.eye(gram.shape[0]),
            check_finite=False,
        )
        weights = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(weights, 0.0)
        self.weights = weights.astype(np.float32)
        return self

    def score(self, users):
        return self.interactions[users] @ self.weights
