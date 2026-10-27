"""Item-based CF: cosine similarity with overlap shrinkage."""

import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, k=100, shrinkage=10.0):
        if k < 1 or shrinkage < 0:
            raise ValueError("k must be positive and shrinkage nonnegative")
        self.k = k
        self.shrinkage = shrinkage

    def fit(self, positive):
        self.interactions = sparse.csr_matrix(positive, dtype=np.float32, copy=True)
        self.interactions.eliminate_zeros()
        self.interactions.data[:] = 1.0
        overlap = (self.interactions.T @ self.interactions).tocsr()
        counts = np.asarray(self.interactions.sum(axis=0)).ravel()
        overlap.setdiag(0)
        overlap.eliminate_zeros()

        values, indices, indptr = [], [], [0]
        for item in range(overlap.shape[0]):
            start, stop = overlap.indptr[item : item + 2]
            neighbors = overlap.indices[start:stop]
            common = overlap.data[start:stop]
            similarities = common / np.sqrt(counts[item] * counts[neighbors])
            similarities *= common / (common + self.shrinkage)
            # Equal similarities are resolved by the item index.
            best = np.lexsort((neighbors, -similarities))[: self.k]
            values.extend(similarities[best])
            indices.extend(neighbors[best])
            indptr.append(len(indices))

        nearest = sparse.csr_matrix(
            (np.asarray(values, dtype=np.float32), indices, indptr),
            shape=overlap.shape,
        )
        # Each target column contains its k nearest source items.
        self.similarity = nearest.T.tocsr()
        return self

    def score(self, users):
        return (self.interactions[users] @ self.similarity).toarray()
