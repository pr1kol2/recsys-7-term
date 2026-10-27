"""ALS with signed, age-discounted rating targets and unchanged confidence."""

import numpy as np
from scipy import sparse

from .ials import IALS


class CustomALS(IALS):
    def __init__(self, half_life_days=365.0, **kwargs):
        super().__init__(**kwargs)
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        self.half_life_days = half_life_days

    def _targets(self, ratings, timestamps=None, cutoff=None):
        if timestamps is None or cutoff is None:
            raise ValueError("CustomALS requires timestamps and a training cutoff")
        timestamps = sparse.csr_matrix(timestamps, dtype=np.float64).copy()
        timestamps.sort_indices()
        if (
            timestamps.shape != ratings.shape
            or not np.array_equal(timestamps.indptr, ratings.indptr)
            or not np.array_equal(timestamps.indices, ratings.indices)
        ):
            raise ValueError("timestamps must describe exactly the observed ratings")
        age_days = (float(cutoff) - timestamps.data) / 86400.0
        if np.any(~np.isfinite(age_days)) or np.any(age_days < 0):
            raise ValueError("timestamps must be finite and no later than cutoff")
        recency = 0.5 + 0.5 * np.exp2(-age_days / self.half_life_days)
        return ((ratings.data - 3.0) / 2.0) * recency
