"""Task 3 Step 4: freeze the backbone, collect balanced features from the
three source validation sets, 70/30 split (seed 6304), multinomial logistic
regression (C=1) predicting {Photo, Art Painting, Cartoon}. Chance = 33.3%;
lower = stronger invariance across the OBSERVED sources (not necessarily
better class separability -- see What to Watch For in the PDF)."""
from __future__ import annotations

import numpy as np

from common.metrics import domain_separability_score


def compute_source_domain_separability(features_by_domain: dict[str, np.ndarray], seed: int = 6304) -> float:
    """features_by_domain: {"photo": [...], "art_painting": [...], "cartoon": [...]}.
    Balances all three to the smallest domain's count before the probe."""
    rng = np.random.RandomState(seed)
    n = min(len(v) for v in features_by_domain.values())
    feats, labels = [], []
    for domain_idx, (name, feat) in enumerate(sorted(features_by_domain.items())):
        idx = rng.choice(len(feat), n, replace=False)
        feats.append(feat[idx])
        labels.append(np.full(n, domain_idx))
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    return domain_separability_score(X, y, seed=seed, C=1.0)
