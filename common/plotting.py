"""Small matplotlib helpers so every task produces consistent-looking figures
into report/figures/ without repeating boilerplate in each notebook."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, out_path: str | Path):
    """Saves a PNG (for quick viewing) and a vector PDF (for the LaTeX report)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_figure(fig, out_path: str | Path):
    """Public alias of _save for figures built directly in notebooks."""
    return _save(fig, out_path)


def bar_comparison(labels: Sequence[str], values: Sequence[float], title: str, ylabel: str, out_path: str | Path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    return _save(fig, out_path)


def curve_vs_x(x: Sequence[float], series: Dict[str, Sequence[float]], title: str, xlabel: str, ylabel: str, out_path: str | Path):
    """e.g. Task 1 translation curve: accuracy/consistency vs pixel displacement,
    one line per model."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, y in series.items():
        ax.plot(x, y, marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def score_distributions(scores_known: Dict[str, np.ndarray], scores_unknown: Dict[str, np.ndarray], out_path: str | Path):
    """Multi-panel histogram of unknownness scores, one panel per score name, as
    required by Task 4 Step 6 (MSP, MLS, Mahalanobis)."""
    names = list(scores_known.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        ax.hist(scores_known[name], bins=40, alpha=0.6, label="known", density=True)
        ax.hist(scores_unknown[name], bins=40, alpha=0.6, label="unknown", density=True)
        ax.set_title(name)
        ax.legend(fontsize=8)
    return _save(fig, out_path)


def roc_curves(known_scores: Dict[str, np.ndarray], unknown_scores: Dict[str, np.ndarray], out_path: str | Path):
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name in known_scores:
        y = np.concatenate([np.zeros_like(known_scores[name]), np.ones_like(unknown_scores[name])])
        s = np.concatenate([known_scores[name], unknown_scores[name]])
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize=8)
    return _save(fig, out_path)


def embedding_scatter(
    coords: np.ndarray,
    class_labels: Iterable,
    condition_labels: Iterable,
    title: str,
    out_path: str | Path,
):
    """t-SNE/UMAP scatter for Task 1 Step 6: color = ground-truth class, marker =
    clean vs transformed."""
    coords = np.asarray(coords)
    class_labels = np.asarray(class_labels)
    condition_labels = np.asarray(condition_labels)
    unique_classes = np.unique(class_labels)
    unique_conditions = np.unique(condition_labels)
    markers = ["o", "x", "^", "s", "D"]
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(6, 5))
    for ci, cls in enumerate(unique_classes):
        for mi, cond in enumerate(unique_conditions):
            mask = (class_labels == cls) & (condition_labels == cond)
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[cmap(ci % 10)], marker=markers[mi % len(markers)],
                s=14, alpha=0.7,
                label=f"{cls}-{cond}" if False else None,
            )
    # Compact legend: classes by color, conditions by marker shape.
    from matplotlib.lines import Line2D
    color_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(i % 10), label=str(c))
                      for i, c in enumerate(unique_classes)]
    marker_handles = [Line2D([0], [0], marker=markers[i % len(markers)], color="gray", linestyle="",
                              label=str(c)) for i, c in enumerate(unique_conditions)]
    leg1 = ax.legend(handles=color_handles, title="class", loc="upper left", fontsize=7, bbox_to_anchor=(1.02, 1))
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, title="condition", loc="lower left", fontsize=7, bbox_to_anchor=(1.02, 0))
    ax.set_title(title)
    return _save(fig, out_path)


def training_curves(metrics_jsonl_path: str, keys: list[str], title: str, xlabel: str, ylabel: str, out_path: str):
    """Plot training/validation curves from a RunLogger .jsonl file.
    keys: list of metric names to plot (e.g. ['cls_loss', 'mean_val_macro_f1'])."""
    import json
    data = {k: {"steps": [], "values": []} for k in keys}
    with open(metrics_jsonl_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            if "step" not in row: continue
            for k in keys:
                if k in row:
                    data[k]["steps"].append(row["step"])
                    data[k]["values"].append(row[k])
    
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in keys:
        if data[k]["steps"]:
            ax.plot(data[k]["steps"], data[k]["values"], marker="o", label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def confusion_matrix_heatmap(y_true, y_pred, class_names, title, out_path):
    """Plot a confusion matrix heatmap using sklearn.metrics.confusion_matrix."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return _save(fig, out_path)


def separability_vs_accuracy_scatter(method_names, separabilities, accuracies, title, out_path):
    """Scatter plot: one point per method, x=domain_separability, y=target_accuracy."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, sep, acc in zip(method_names, separabilities, accuracies):
        ax.scatter(sep, acc, label=name, s=50)
        ax.text(sep, acc, f" {name}", va="bottom", ha="left", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Domain Separability")
    ax.set_ylabel("Target Accuracy")
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def controlled_study_plot(swept_values, metric_values: dict, title, xlabel, out_path):
    """Line/scatter plot for Task 3's controlled study: swept_values on x-axis,
    metric_values is a dict of {metric_name: [values]} for multiple y-series."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for metric_name, values in metric_values.items():
        ax.plot(swept_values, values, marker="o", label=metric_name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Metric Value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def contact_sheet(images_accepted, images_rejected, labels_accepted, labels_rejected, out_path, n_show=4):
    """Small figure showing a few accepted vs rejected cue-conflict images side by side."""
    n_show = min(n_show, len(images_accepted), len(images_rejected))
    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4.5))
    
    if n_show == 1:
        axes = np.expand_dims(axes, axis=1)
        
    for i in range(n_show):
        # Top row: accepted
        axes[0, i].imshow(images_accepted[i])
        axes[0, i].set_title(f"Acc: {labels_accepted[i]}", fontsize=9)
        axes[0, i].axis("off")
        # Bottom row: rejected
        axes[1, i].imshow(images_rejected[i])
        axes[1, i].set_title(f"Rej: {labels_rejected[i]}", fontsize=9)
        axes[1, i].axis("off")
    
    fig.suptitle("Cue Conflict: Accepted (Top) vs Rejected (Bottom)")
    plt.tight_layout()
    return _save(fig, out_path)


def grouped_bar_near_far(method_names, near_rejection_rates, far_rejection_rates, title, out_path):
    """Grouped bar chart: near vs far unknown rejection rate by method."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(method_names))
    width = 0.35
    
    ax.bar(x - width/2, near_rejection_rates, width, label='Near OOD', color="#4C72B0")
    ax.bar(x + width/2, far_rejection_rates, width, label='Far OOD', color="#DD8452")
    
    ax.set_ylabel("Rejection Rate")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=20, ha="right")
    ax.legend()
    
    return _save(fig, out_path)
