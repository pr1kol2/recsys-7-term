"""Second-order factorization machine for binary relevance."""

import numpy as np
from scipy import sparse
import torch
from torch import nn


def _sample_negatives(ratings, ratio, rng):
    """Sample uniformly from items the user has not rated.

    Exclude all rated items, regardless of the rating.
    """
    n_users, n_items = ratings.shape
    counts = np.diff(ratings.indptr)
    users = np.repeat(np.arange(n_users), counts)
    users = np.repeat(users[counts[users] < n_items], ratio)
    items = rng.integers(n_items, size=len(users))
    known = np.repeat(np.arange(n_users), counts) * n_items + ratings.indices

    # If most items are rated, sample directly from the remaining ones.
    dense_users = np.flatnonzero((counts > n_items / 2) & (counts < n_items))
    for user in dense_users:
        positions = np.flatnonzero(users == user)
        seen = ratings.indices[ratings.indptr[user] : ratings.indptr[user + 1]]
        missing = np.setdiff1d(np.arange(n_items), seen, assume_unique=True)
        items[positions] = rng.choice(missing, len(positions))

    pending = np.arange(len(users))
    while len(pending):
        keys = users[pending] * n_items + items[pending]
        positions = np.searchsorted(known, keys)
        present = positions < len(known)
        present[present] = known[positions[present]] == keys[present]
        pending = pending[present]
        items[pending] = rng.integers(n_items, size=len(pending))
    return users, items


class _FM(nn.Module):
    def __init__(self, n_users, item_features, factors):
        super().__init__()
        n_items, n_metadata = item_features.shape
        self.bias = nn.Parameter(torch.zeros(()))
        self.user_linear = nn.Embedding(n_users, 1)
        self.item_linear = nn.Embedding(n_items, 1)
        self.user_factors = nn.Embedding(n_users, factors)
        self.item_factors = nn.Embedding(n_items, factors)
        nn.init.zeros_(self.user_linear.weight)
        nn.init.zeros_(self.item_linear.weight)
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.item_factors.weight, std=0.05)

        self.has_metadata = n_metadata > 0
        if self.has_metadata:
            self.meta_linear = nn.Embedding(n_metadata + 1, 1, padding_idx=n_metadata)
            self.meta_factors = nn.Embedding(n_metadata + 1, factors, padding_idx=n_metadata)
            nn.init.zeros_(self.meta_linear.weight)
            nn.init.normal_(self.meta_factors.weight, std=0.05)
            with torch.no_grad():
                self.meta_factors.weight[n_metadata].zero_()
            width = int(np.diff(item_features.indptr).max(initial=0))
            ids = np.full((n_items, width), n_metadata, dtype=np.int64)
            values = np.zeros((n_items, width), dtype=np.float32)
            for item in range(n_items):
                start, end = item_features.indptr[item : item + 2]
                ids[item, : end - start] = item_features.indices[start:end]
                values[item, : end - start] = item_features.data[start:end]
            self.register_buffer("meta_ids", torch.from_numpy(ids))
            self.register_buffer("meta_values", torch.from_numpy(values))

    def item_terms(self, items):
        vectors = self.item_factors(items)
        linear = self.item_linear(items).squeeze(-1)
        if not self.has_metadata:
            return linear, vectors
        ids, values = self.meta_ids[items], self.meta_values[items]
        metadata = self.meta_factors(ids) * values.unsqueeze(-1)
        summed = vectors + metadata.sum(dim=1)
        squared = vectors.square() + metadata.square().sum(dim=1)
        within_item = 0.5 * (summed.square() - squared).sum(dim=1)
        linear = linear + (self.meta_linear(ids).squeeze(-1) * values).sum(dim=1)
        return linear + within_item, summed

    def forward(self, users, items):
        item_bias, item_vectors = self.item_terms(items)
        return (
            self.bias
            + self.user_linear(users).squeeze(-1)
            + item_bias
            + (self.user_factors(users) * item_vectors).sum(dim=1)
        )


class FM:
    """FM with user IDs, item IDs, and item metadata.

    Ratings >= 4 get label 1. Lower ratings and sampled unrated pairs
    get label 0. Train with BCE and L2 regularization.
    Rank by logits: sigmoid preserves their order.
    """

    def __init__(
        self, factors=16, lr=0.01, weight_decay=1e-5, epochs=8,
        batch_size=4096, negative_ratio=1, seed=42, use_metadata=True,
    ):
        if not isinstance(negative_ratio, int) or negative_ratio < 0:
            raise ValueError("negative_ratio must be a nonnegative integer")
        self.factors = factors
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.seed = seed
        self.use_metadata = use_metadata

    def fit(self, ratings, item_features=None):
        ratings = sparse.csr_matrix(ratings, dtype=np.float32, copy=True)
        ratings.eliminate_zeros()
        ratings.sort_indices()
        n_users, n_items = ratings.shape
        if not ratings.nnz or not np.isfinite(ratings.data).all():
            raise ValueError("ratings must contain finite observed ratings")
        if np.any((ratings.data < 1) | (ratings.data > 5)):
            raise ValueError("observed ratings must be between 1 and 5")
        if self.use_metadata:
            if item_features is None:
                raise ValueError("item_features are required when use_metadata=True")
            features = sparse.csr_matrix(item_features, dtype=np.float32, copy=True)
            features.sum_duplicates()
            features.eliminate_zeros()
            features.sort_indices()
            if features.shape[0] != n_items or not np.isfinite(features.data).all():
                raise ValueError("item_features must have one finite row per item")
        else:
            features = sparse.csr_matrix((n_items, 0), dtype=np.float32)

        rng = np.random.default_rng(self.seed)
        positive_users = np.repeat(np.arange(n_users), np.diff(ratings.indptr))
        negative_users, negative_items = _sample_negatives(ratings, self.negative_ratio, rng)
        users = np.concatenate([positive_users, negative_users])
        items = np.concatenate([ratings.indices, negative_items]).astype(np.int64)
        labels = np.concatenate([(ratings.data >= 4).astype(np.float32),
                                 np.zeros(len(negative_users), dtype=np.float32)])
        self.n_negative_ = len(negative_users)
        self.n_training_examples_ = len(labels)

        # Start with the same ID embeddings with and without metadata.
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            self.module_ = _FM(n_users, features, self.factors)
        optimizer = torch.optim.Adam(self.module_.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        self.history_ = []
        self.module_.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(labels))
            total_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                logits = self.module_(torch.from_numpy(users[batch]),
                                      torch.from_numpy(items[batch]))
                loss = criterion(logits, torch.from_numpy(labels[batch]))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            self.history_.append(total_loss / len(labels))

        self.module_.eval()
        with torch.no_grad():
            item_bias, item_vectors = self.module_.item_terms(torch.arange(n_items))
            self.item_bias_ = item_bias.numpy().copy()
            self.item_vectors_ = item_vectors.numpy().copy()
            self.user_bias_ = self.module_.user_linear.weight[:, 0].numpy().copy()
            self.user_vectors_ = self.module_.user_factors.weight.numpy().copy()
            self.bias_ = self.module_.bias.item()
        return self

    def score(self, users):
        users = np.asarray(users, dtype=np.int64)
        return (self.user_vectors_[users] @ self.item_vectors_.T
                + self.user_bias_[users, None] + self.item_bias_[None, :] + self.bias_)
