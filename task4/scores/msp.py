"""u_MSP(x) = 1 - max_k p_k(x), p = softmax(logits). Larger u = more novel."""
import numpy as np


def msp_score(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    probs = e / e.sum(axis=1, keepdims=True)
    return 1.0 - probs.max(axis=1)
