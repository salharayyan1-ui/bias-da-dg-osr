"""u_Energy(x) = -log sum_k exp(z_k(x)), the free-energy score of Liu et al. (2020).
Larger u = more novel. Computed with a log-sum-exp for numerical stability."""
import numpy as np
from scipy.special import logsumexp


def energy_score(logits: np.ndarray) -> np.ndarray:
    return -logsumexp(logits, axis=1)
