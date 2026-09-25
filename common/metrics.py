"""Metric helpers shared by Tasks 1-4.

Kept torch-free / numpy+sklearn-only where possible so they're easy to unit-test
without a full deep-learning environment, and easy to reuse whether predictions
came from a torch model, a cached numpy array, or a notebook cell.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int] | None = None) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))


def mean_max_confidence(probs: np.ndarray) -> float:
    """probs: [N, C] softmax (or scaled-similarity softmax for zero-shot CLIP)."""
    return float(np.mean(np.max(probs, axis=1)))


def prediction_consistency(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """Fraction of examples whose predicted class is unchanged, e.g. clean vs.
    transformed, as used for the color/translation/patch-shuffle interventions."""
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    return float(np.mean(pred_a == pred_b))


def cosine_stability(feat_clean: np.ndarray, feat_transformed: np.ndarray) -> float:
    """I_T from Task 1 Step 6: mean cosine similarity between paired clean and
    transformed representations. feat_*: [N, D]."""
    a = np.asarray(feat_clean, dtype=np.float64)
    b = np.asarray(feat_transformed, dtype=np.float64)
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(a_n * b_n, axis=1)))


def shape_bias_and_coverage(pred_labels: np.ndarray, shape_labels: np.ndarray, texture_labels: np.ndarray):
    """Task 1 Step 3 cue-conflict scoring.

    pred_labels, shape_labels, texture_labels: arrays of class ids, one entry per
    cue-conflict image. Returns (shape_bias_pct, coverage_pct, n_shape, n_texture,
    n_other, n_total).
    """
    pred_labels = np.asarray(pred_labels)
    shape_labels = np.asarray(shape_labels)
    texture_labels = np.asarray(texture_labels)
    n_total = len(pred_labels)
    is_shape = pred_labels == shape_labels
    is_texture = (pred_labels == texture_labels) & (~is_shape)
    n_shape = int(is_shape.sum())
    n_texture = int(is_texture.sum())
    n_other = n_total - n_shape - n_texture
    denom = n_shape + n_texture
    shape_bias = 100.0 * n_shape / denom if denom > 0 else float("nan")
    coverage = 100.0 * denom / n_total if n_total > 0 else float("nan")
    return shape_bias, coverage, n_shape, n_texture, n_other, n_total


def auroc(unknownness_scores_known: np.ndarray, unknownness_scores_unknown: np.ndarray) -> float:
    """Task 4: larger score = more novel/unknown. AUROC of separating known (label
    0) from unknown (label 1) using the unknownness score directly as the ranking
    statistic."""
    y = np.concatenate([np.zeros_like(unknownness_scores_known), np.ones_like(unknownness_scores_unknown)])
    s = np.concatenate([unknownness_scores_known, unknownness_scores_unknown])
    return float(roc_auc_score(y, s))


def threshold_at_known_percentile(unknownness_scores_val_known: np.ndarray, percentile: float = 95.0) -> float:
    """Task 4 Step 6: tau = 95th percentile of unknownness on the CIFAR-10
    validation set, calibrated on known data only. Accept x when u(x) <= tau."""
    return float(np.percentile(np.asarray(unknownness_scores_val_known), percentile))


def acceptance_and_fpr_at_95tpr(scores_known_test: np.ndarray, scores_unknown: np.ndarray, tau: float):
    """Given a threshold tau calibrated to accept ~95% of known validation
    examples, report:
      - achieved known (test) acceptance rate: fraction with score <= tau
      - unknown rejection rate: fraction with score > tau
      - FPR@95TPR: fraction of *unknown* examples incorrectly accepted (score <= tau)
    """
    known_accept_rate = float(np.mean(np.asarray(scores_known_test) <= tau))
    unknown_reject_rate = float(np.mean(np.asarray(scores_unknown) > tau))
    fpr_at_95tpr = float(np.mean(np.asarray(scores_unknown) <= tau))
    return known_accept_rate, unknown_reject_rate, fpr_at_95tpr


def domain_separability_score(features: np.ndarray, domain_labels: np.ndarray, seed: int = 6304, C: float = 1.0):
    """Balanced logistic regression (C=1) trained on a stratified 70/30 split
    (seed 6304) to predict the domain of a frozen feature (source-vs-target in
    Task 2, Photo/Art/Cartoon in Task 3). Returns held-out accuracy; chance is
    50% (Task 2) or 33.3% (Task 3).

    Features are standardized (scaler fit on the 70% probe-train split only)
    before the probe. Without this, C=1 acts as a different amount of
    regularization for methods that change the feature scale (e.g. MMD
    alignment tends to shrink feature norms), which would confound the
    comparison across methods."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_tr, X_te, y_tr, y_te = train_test_split(
        features, domain_labels, test_size=0.30, random_state=seed, stratify=domain_labels
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=seed),
    )
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))
