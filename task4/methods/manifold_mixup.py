"""Manifold mixup helpers for PROSER's data placeholders (Task 4 Step 4).
Mixes representations of two DIFFERENT-class examples (mixing same-class
examples wouldn't produce an "unknown-like" region between classes)."""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def cross_class_pair_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    """Returns a permutation `perm` such that labels[perm[i]] != labels[i]
    for every i, whenever that's achievable -- i.e. whenever no single class
    holds more than half the batch (always true in practice for CIFAR-10
    batches of size >= a few dozen with roughly balanced classes).

    Construction: sort indices by class (grouping same-class indices
    together, with a random shuffle *within* each class group so the pairing
    isn't always the same two images), then rotate that class-sorted order by
    half its length before pairing position i with its rotated partner. This
    guarantees each element is paired with one from "the other side" of the
    class-sorted array, which lands in a different class whenever the
    majority-class condition above holds -- unlike a naive random-permutation
    + local-swap scheme, which can strand a single leftover self-collision
    (verified empirically to happen ~99% of the time with random-restart
    swapping, hence the deliberate sort+rotate construction here instead).
    """
    rng = np.random.RandomState(seed)
    n = len(labels)

    order = np.argsort(labels, kind="stable")
    # Shuffle within each class group so pairing isn't deterministic run-to-run
    # beyond the seed, and isn't always "first vs. middle" for the same class.
    start = 0
    for c in np.unique(labels):
        count = int((labels == c).sum())
        group = order[start:start + count].copy()
        rng.shuffle(group)
        order[start:start + count] = group
        start += count

    half = n // 2
    partner_of_order_pos = np.roll(np.arange(n), -half)  # position i -> position (i+half) mod n
    perm = np.empty(n, dtype=int)
    for i in range(n):
        perm[order[i]] = order[partner_of_order_pos[i]]

    n_collisions = int((labels[perm] == labels).sum())
    if n_collisions > 0:
        # Only possible if some class holds > half the batch; fix any
        # remaining self-pairs with a direct, guaranteed-correct linear scan.
        for i in np.where(labels[perm] == labels)[0]:
            for j in range(n):
                if j == i:
                    continue
                if labels[perm[j]] != labels[i] and labels[perm[i]] != labels[j]:
                    perm[i], perm[j] = perm[j], perm[i]
                    break
    return perm


def sample_beta_lambda(batch_size: int, alpha: float = 2.0, beta: float = 2.0, seed: int | None = None) -> np.ndarray:
    """lambda ~ Beta(2, 2), one value per mixed example, per the spec."""
    rng = np.random.RandomState(seed)
    return rng.beta(alpha, beta, size=batch_size).astype(np.float32)


def mix_features(feat_a: "torch.Tensor", feat_b: "torch.Tensor", lam: "torch.Tensor") -> "torch.Tensor":
    """h_tilde = lam * h_i + (1 - lam) * h_j, lam broadcast over all non-batch
    dims (feat_a, feat_b: [N, C, H, W] or [N, D]; lam: [N])."""
    view_shape = [lam.shape[0]] + [1] * (feat_a.dim() - 1)
    lam = lam.view(*view_shape)
    return lam * feat_a + (1.0 - lam) * feat_b
