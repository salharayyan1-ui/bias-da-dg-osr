"""u_Mah(x) = min_c (f(x)-mu_c)^T Sigma^-1 (f(x)-mu_c), with per-class means
mu_c and ONE shared diagonal covariance Sigma estimated from unaugmented
CIFAR-10 TRAINING features (not validation/test), + 1e-6 on every diagonal
entry. Diagonal Sigma makes the quadratic form a simple weighted squared
distance -- no matrix inverse needed."""
from __future__ import annotations

import numpy as np


class MahalanobisScorer:
    def __init__(self):
        self.class_means: np.ndarray | None = None   # [C, D]
        self.inv_diag_var: np.ndarray | None = None   # [D]

    def fit(self, train_features: np.ndarray, train_labels: np.ndarray, n_classes: int, eps: float = 1e-6):
        d = train_features.shape[1]
        means = np.zeros((n_classes, d))
        sq_dev_sum = np.zeros(d)
        n_total = len(train_features)
        for c in range(n_classes):
            feats_c = train_features[train_labels == c]
            means[c] = feats_c.mean(axis=0)
            sq_dev_sum += ((feats_c - means[c]) ** 2).sum(axis=0)
        shared_var = sq_dev_sum / n_total + eps
        self.class_means = means
        self.inv_diag_var = 1.0 / shared_var
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Returns [N] unknownness scores: min over classes of the (diagonal)
        Mahalanobis distance."""
        assert self.class_means is not None, "call .fit() first"
        # dists[n, c] = sum_d (f[n,d]-mu[c,d])^2 * inv_var[d]
        diffs = features[:, None, :] - self.class_means[None, :, :]     # [N, C, D]
        dists = np.einsum("ncd,d->nc", diffs ** 2, self.inv_diag_var)   # [N, C]
        return dists.min(axis=1)
