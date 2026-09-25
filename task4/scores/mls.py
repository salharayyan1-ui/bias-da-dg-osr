"""u_MLS(x) = -max_k z_k(x): negative of the largest raw logit. Larger u = more novel."""
import numpy as np


def mls_score(logits: np.ndarray) -> np.ndarray:
    return -logits.max(axis=1)
