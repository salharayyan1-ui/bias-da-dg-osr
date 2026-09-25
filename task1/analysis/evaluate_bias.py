"""Task 1 scoring for Steps 1-5 (torch-free; works on numpy outputs of
task1.analysis.feature_similarity.ModelSuite.run)."""
from __future__ import annotations

import numpy as np

from common.metrics import (
    macro_f1,
    mean_max_confidence,
    prediction_consistency,
    shape_bias_and_coverage,
    top1_accuracy,
)


def clean_baseline_rows(outputs: dict, labels: np.ndarray) -> list[dict]:
    return [{
        "model": name,
        "top1_accuracy": top1_accuracy(labels, out["preds"]),
        "macro_f1": macro_f1(labels, out["preds"]),
        "mean_max_confidence": mean_max_confidence(out["probs"]),
    } for name, out in outputs.items()]


def intervention_rows(clean_outputs: dict, interv_outputs: dict, labels: np.ndarray, intervention: str) -> list[dict]:
    """Absolute accuracy/F1 under the intervention, change relative to the
    model's own clean baseline, prediction consistency vs. clean predictions,
    and mean max confidence (to see whether a model stays confident)."""
    rows = []
    for name in clean_outputs:
        c, t = clean_outputs[name], interv_outputs[name]
        clean_acc, acc = top1_accuracy(labels, c["preds"]), top1_accuracy(labels, t["preds"])
        rows.append({
            "model": name, "intervention": intervention,
            "clean_accuracy": clean_acc, "accuracy": acc, "accuracy_change": acc - clean_acc,
            "relative_accuracy": acc / clean_acc if clean_acc > 0 else float("nan"),
            "macro_f1": macro_f1(labels, t["preds"]),
            "prediction_consistency": prediction_consistency(c["preds"], t["preds"]),
            "mean_max_confidence": mean_max_confidence(t["probs"]),
            "clean_mean_max_confidence": mean_max_confidence(c["probs"]),
        })
    return rows


def translation_rows(clean_outputs: dict, shifted: dict, labels: np.ndarray) -> list[dict]:
    """shifted[(pixels, direction)] = outputs dict. Accuracy and consistency are
    computed per direction and then averaged across the four directions (the
    spec's "average the results across directions"); per-direction values are
    kept too. Pixel 0 is the clean image (consistency 1 by definition)."""
    rows = []
    for name in clean_outputs:
        clean_acc = top1_accuracy(labels, clean_outputs[name]["preds"])
        rows.append({"model": name, "pixels": 0, "direction": "mean", "accuracy": clean_acc,
                     "accuracy_change": 0.0, "consistency": 1.0})
        pixels = sorted({p for p, _ in shifted})
        for p in pixels:
            accs, cons = [], []
            for (pp, d), outs in shifted.items():
                if pp != p:
                    continue
                a = top1_accuracy(labels, outs[name]["preds"])
                k = prediction_consistency(clean_outputs[name]["preds"], outs[name]["preds"])
                accs.append(a)
                cons.append(k)
                rows.append({"model": name, "pixels": p, "direction": d, "accuracy": a,
                             "accuracy_change": a - clean_acc, "consistency": k})
            rows.append({"model": name, "pixels": p, "direction": "mean", "accuracy": float(np.mean(accs)),
                         "accuracy_change": float(np.mean(accs)) - clean_acc,
                         "consistency": float(np.mean(cons)),
                         "accuracy_std_over_directions": float(np.std(accs)),
                         "consistency_std_over_directions": float(np.std(cons))})
    return rows


def shape_texture_rows(conflict_outputs: dict, shape_labels: np.ndarray, texture_labels: np.ndarray) -> list[dict]:
    rows = []
    for name, out in conflict_outputs.items():
        sb, cov, ns, nt, no, ntot = shape_bias_and_coverage(out["preds"], shape_labels, texture_labels)
        rows.append({"model": name, "n_shape": ns, "n_texture": nt, "n_other": no, "n_total": ntot,
                     "shape_bias_pct": sb, "coverage_pct": cov,
                     "mean_max_confidence": float(out["probs"].max(1).mean())})
    return rows


def shape_texture_by_group(conflict_outputs: dict, shape_labels, texture_labels, groups) -> list[dict]:
    """Same decision counts broken down by an arbitrary grouping (e.g. class
    pair or direction) so that one easy pair cannot drive the aggregate."""
    groups = np.asarray(groups)
    rows = []
    for g in sorted(set(groups.tolist())):
        m = groups == g
        for name, out in conflict_outputs.items():
            sb, cov, ns, nt, no, ntot = shape_bias_and_coverage(out["preds"][m], np.asarray(shape_labels)[m],
                                                                np.asarray(texture_labels)[m])
            rows.append({"group": g, "model": name, "n_shape": ns, "n_texture": nt, "n_other": no,
                         "n_total": ntot, "shape_bias_pct": sb, "coverage_pct": cov})
    return rows


def decision_category(preds, shape_labels, texture_labels) -> np.ndarray:
    preds = np.asarray(preds)
    cat = np.full(len(preds), "other", dtype=object)
    cat[preds == np.asarray(texture_labels)] = "texture"
    cat[preds == np.asarray(shape_labels)] = "shape"
    return cat
