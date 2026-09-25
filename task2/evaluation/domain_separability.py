"""Task 2 Step 5: freeze each trained backbone, collect equal numbers of
source-validation and target features, and train a balanced logistic
regression (70/30 split, seed 6304, C=1) to distinguish source from target.
Held-out accuracy = domain separability score; 50% = chance."""
from __future__ import annotations

import numpy as np

from common.metrics import domain_separability_score


def compute_domain_separability(source_val_features: np.ndarray, target_features: np.ndarray, seed: int = 6304) -> float:
    """Balances the two classes by subsampling the larger one down to the
    smaller one's size (deterministically, seed 6304) before the 70/30 probe
    split, per "collect equal numbers of source-validation and target
    features."""
    rng = np.random.RandomState(seed)
    n = min(len(source_val_features), len(target_features))
    src_idx = rng.choice(len(source_val_features), n, replace=False)
    tgt_idx = rng.choice(len(target_features), n, replace=False)
    features = np.concatenate([source_val_features[src_idx], target_features[tgt_idx]], axis=0)
    domain_labels = np.concatenate([np.zeros(n), np.ones(n)])
    return domain_separability_score(features, domain_labels, seed=seed, C=1.0)
