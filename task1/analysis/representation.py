"""Task 1 Step 6: representation stability and 2-D projections."""
from __future__ import annotations

import numpy as np

from common.metrics import cosine_stability


def paired_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a * b, axis=1)


def random_pair_baseline(clean: np.ndarray, seed: int = 6304) -> float:
    """Mean cosine between each clean image and a *different* random clean
    image. Raw cosine levels differ by backbone (e.g. ResNet features are
    non-negative after ReLU+pooling, so even unrelated images have high
    cosine); this floor makes I_T interpretable across backbones."""
    rng = np.random.RandomState(seed)
    n = len(clean)
    perm = rng.permutation(n)
    fix = perm == np.arange(n)
    perm[fix] = (perm[fix] + 1) % n
    return float(paired_cosines(clean, clean[perm]).mean())


def stability_rows(backbone: str, intervention: str, clean: np.ndarray, transformed: np.ndarray,
                   flips_by_model: dict | None = None, seed: int = 6304) -> list[dict]:
    """I_T for one (backbone, intervention) plus interpretation aids:
      * random-pair floor and a normalized stability
        (I_T - floor) / (1 - floor), 1 = unchanged, 0 = as dissimilar as an
        unrelated image;
      * I_T split by whether the model's prediction flipped, for each model
        that uses this backbone (prediction vs. representation agreement)."""
    cos = paired_cosines(clean, transformed)
    floor = random_pair_baseline(clean, seed)
    it = cosine_stability(clean, transformed)
    row = {"backbone": backbone, "intervention": intervention, "n": len(cos), "I_T": it,
           "I_T_std": float(cos.std()), "random_pair_cosine": floor,
           "normalized_stability": (it - floor) / (1 - floor) if floor < 1 else float("nan")}
    rows = [row]
    for model, flipped in (flips_by_model or {}).items():
        flipped = np.asarray(flipped, dtype=bool)
        rows.append({"backbone": backbone, "intervention": intervention, "model": model,
                     "frac_prediction_changed": float(flipped.mean()),
                     "I_T_prediction_unchanged": float(cos[~flipped].mean()) if (~flipped).any() else float("nan"),
                     "I_T_prediction_changed": float(cos[flipped].mean()) if flipped.any() else float("nan")})
    return rows


def fit_2d_projection(features: np.ndarray, method: str = "tsne", seed: int = 6304, perplexity: float = 30.0,
                      n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """One 2-D projection fitted to the combined clean+transformed features.
    t-SNE uses cosine distance (the same geometry as I_T) and PCA init."""
    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, perplexity=perplexity, metric="cosine", init="pca",
                    learning_rate="auto", random_state=seed).fit_transform(features)
    if method == "umap":
        import umap

        return umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine",
                         random_state=seed).fit_transform(features)
    raise ValueError(method)
