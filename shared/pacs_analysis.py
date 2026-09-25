"""Tables and figures shared by the Task 2 and Task 3 evaluations."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from common.metrics import macro_f1, top1_accuracy
from common.plotting import save_figure
from shared.pacs import PACS_CLASSES

DOMAIN_LABELS = {"photo": "Photo", "art_painting": "Art", "cartoon": "Cartoon", "sketch": "Sketch"}


def per_class_accuracy(preds, labels, n_classes=7):
    return np.array([(preds[labels == c] == c).mean() if (labels == c).any() else np.nan for c in range(n_classes)])


def source_rows(run_preds: dict, source_domains) -> dict:
    """run_preds[domain] = predict() output. Returns per-domain acc/F1 plus
    mean and worst across the source domains."""
    row = {}
    accs, f1s = [], []
    for d in source_domains:
        p = run_preds[d]
        a, f = top1_accuracy(p["labels"], p["preds"]), macro_f1(p["labels"], p["preds"])
        row[f"{d}_acc"], row[f"{d}_f1"] = a, f
        accs.append(a); f1s.append(f)
    row.update({"mean_source_acc": float(np.mean(accs)), "mean_source_f1": float(np.mean(f1s)),
                "worst_source_acc": float(np.min(accs)), "worst_source_f1": float(np.min(f1s)),
                "worst_source_domain_by_f1": source_domains[int(np.argmin(f1s))]})
    return row


def per_class_delta_rows(target_preds: dict, baseline: str) -> list[dict]:
    """target_preds[run] = predict() output on the target domain."""
    base = target_preds[baseline]
    base_acc = per_class_accuracy(base["preds"], base["labels"])
    rows = []
    for run, p in target_preds.items():
        acc = per_class_accuracy(p["preds"], p["labels"])
        for c, name in enumerate(PACS_CLASSES):
            rows.append({"run": run, "class": name, "n": int((p["labels"] == c).sum()), "accuracy": acc[c],
                         "baseline_accuracy": base_acc[c], "delta_vs_baseline": acc[c] - base_acc[c]})
    return rows


def dominant_confusions(p: dict, class_idx: int, top_k: int = 3) -> list[dict]:
    m = (p["labels"] == class_idx) & (p["preds"] != class_idx)
    n_class = int((p["labels"] == class_idx).sum())
    vals, counts = np.unique(p["preds"][m], return_counts=True)
    order = np.argsort(-counts)[:top_k]
    return [{"predicted_as": PACS_CLASSES[int(vals[i])], "count": int(counts[i]),
             "fraction_of_class": counts[i] / max(n_class, 1)} for i in order]


def largest_change_rows(target_preds: dict, baseline: str) -> list[dict]:
    """For each non-baseline run: the most improved and most degraded class
    relative to the baseline and their dominant confusions under both models."""
    rows = []
    base = target_preds[baseline]
    base_acc = per_class_accuracy(base["preds"], base["labels"])
    for run, p in target_preds.items():
        if run == baseline:
            continue
        delta = per_class_accuracy(p["preds"], p["labels"]) - base_acc
        for kind, c in (("most_improved", int(np.nanargmax(delta))), ("most_degraded", int(np.nanargmin(delta)))):
            rows.append({"run": run, "kind": kind, "class": PACS_CLASSES[c], "delta": float(delta[c]),
                         "confusions_run": dominant_confusions(p, c), "confusions_baseline": dominant_confusions(base, c)})
    return rows


def transition_counts(target_preds: dict, baseline: str) -> list[dict]:
    """How many target images each run fixes (baseline wrong -> run right) and
    breaks (baseline right -> run wrong): an aggregate gain can hide breaks."""
    base = target_preds[baseline]
    bc = base["preds"] == base["labels"]
    rows = []
    for run, p in target_preds.items():
        if run == baseline:
            continue
        rc = p["preds"] == p["labels"]
        rows.append({"run": run, "fixed": int((~bc & rc).sum()), "broken": int((bc & ~rc).sum()),
                     "net": int((~bc & rc).sum() - (bc & ~rc).sum()), "n": len(rc)})
    return rows


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def training_curves(histories: dict, out_path, align_keys=("mmd2", "mean_pairwise_mmd2", "domain_loss")):
    """histories[run] = list of per-epoch rows. Two rows of panels per run:
    classification loss + source-val macro-F1; alignment term (MMD^2 or domain
    loss) + discriminator accuracy where applicable."""
    runs = list(histories)
    fig, axes = plt.subplots(2, len(runs), figsize=(3.1 * len(runs), 4.8), squeeze=False)
    for j, run in enumerate(runs):
        h = histories[run]
        ep = [r["epoch"] for r in h]
        ax = axes[0, j]
        ax.plot(ep, [r["cls_loss"] for r in h], "-o", ms=3, label="train cls loss")
        ax.set_title(run, fontsize=9); ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy", fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(ep, [r["mean_val_macro_f1"] for r in h], "-s", ms=3, color="C2", label="mean src-val F1")
        ax2.set_ylim(0, 1)
        best = int(np.argmax([r["mean_val_macro_f1"] for r in h]))
        ax2.axvline(ep[best], color="gray", ls=":", lw=1)
        if j == len(runs) - 1:
            ax2.set_ylabel("macro-F1", fontsize=8)
        if j == 0:
            l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, fontsize=6, loc="center right")
        ax = axes[1, j]
        key = next((k for k in align_keys if k in h[0]), None)
        if key is None:
            ax.text(0.5, 0.5, "no alignment term", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ax.plot(ep, [r[key] for r in h], "-o", ms=3, color="C3", label=key)
        ax.set_xlabel("epoch"); ax.set_ylabel(key, fontsize=8)
        if "disc_acc" in h[0]:
            ax2 = ax.twinx()
            ax2.plot(ep, [r["disc_acc"] for r in h], "-^", ms=3, color="C4", label="disc. acc")
            ax2.plot(ep, [r["grl_alpha"] for r in h], "--", color="C7", label="GRL alpha")
            ax2.axhline(0.5, color="C4", ls=":", lw=0.8)
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("disc. acc / GRL α", fontsize=7)
            l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, fontsize=6)
        else:
            ax.legend(fontsize=6)
    fig.tight_layout()
    return save_figure(fig, out_path)


def confusion_grid(target_preds: dict, out_path, title_prefix=""):
    runs = list(target_preds)
    fig, axes = plt.subplots(1, len(runs), figsize=(3.0 * len(runs), 3.1), squeeze=False)
    short = [c[:5] for c in PACS_CLASSES]
    for ax, run in zip(axes[0], runs):
        p = target_preds[run]
        cm = confusion_matrix(p["labels"], p["preds"], labels=range(7)).astype(float)
        cm /= cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        for i in range(7):
            for k in range(7):
                ax.text(k, i, f"{cm[i, k]:.2f}"[1:] if cm[i, k] < 1 else "1", ha="center", va="center",
                        fontsize=5, color="white" if cm[i, k] > 0.5 else "black")
        ax.set_xticks(range(7), short, rotation=90, fontsize=6); ax.set_yticks(range(7), short, fontsize=6)
        acc = top1_accuracy(p["labels"], p["preds"])
        ax.set_title(f"{title_prefix}{run} (acc {acc:.3f})", fontsize=8)
        ax.set_xlabel("predicted", fontsize=7)
    axes[0, 0].set_ylabel("true", fontsize=7)
    fig.tight_layout()
    return save_figure(fig, out_path)


def per_class_delta_plot(rows, baseline: str, out_path, ylabel="accuracy change vs. baseline"):
    import pandas as pd

    df = pd.DataFrame(rows)
    df = df[df.run != baseline]
    runs = list(dict.fromkeys(df.run))
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    w = 0.8 / len(runs)
    x = np.arange(7)
    for k, r in enumerate(runs):
        v = df[df.run == r].set_index("class").loc[PACS_CLASSES, "delta_vs_baseline"].values
        ax.bar(x + (k - (len(runs) - 1) / 2) * w, v, w, label=r)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, PACS_CLASSES, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8); ax.legend(fontsize=7, ncol=len(runs)); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, out_path)


def domain_tsne(features_by_run: dict, domains_by_run: dict, out_path, seed=6304, max_per_domain=400):
    """One t-SNE per run on (a seeded subsample of) its source-val and target
    features, coloured by domain."""
    from sklearn.manifold import TSNE

    runs = list(features_by_run)
    fig, axes = plt.subplots(1, len(runs), figsize=(3.0 * len(runs), 3.1), squeeze=False)
    for ax, run in zip(axes[0], runs):
        X, dom = features_by_run[run], np.asarray(domains_by_run[run])
        rng = np.random.RandomState(seed)
        keep = np.concatenate([rng.permutation(np.where(dom == d)[0])[:max_per_domain] for d in dict.fromkeys(dom)])
        xy = TSNE(2, perplexity=30, init="pca", learning_rate="auto", random_state=seed).fit_transform(X[keep])
        for k, d in enumerate(dict.fromkeys(dom)):
            m = dom[keep] == d
            ax.scatter(xy[m, 0], xy[m, 1], s=3, alpha=0.6, label=DOMAIN_LABELS.get(d, d), color=f"C{k}")
        ax.set_title(run, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    return save_figure(fig, out_path)


def failure_grid(paths, titles, out_path, n_cols=8):
    from PIL import Image

    n = len(paths)
    n_rows = max(1, int(np.ceil(n / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.4 * n_cols, 1.75 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, p, t in zip(axes.ravel(), paths, titles):
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title(t, fontsize=5.5)
    fig.tight_layout()
    return save_figure(fig, out_path)


def study_plot(df, x_col, series: dict, out_path, xlabel, log_x=True):
    """series = {column: label}; one panel per column group given as dict of
    dicts {panel_title: {column: label}}."""
    panels = series
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 2.9), squeeze=False)
    for ax, (title, cols) in zip(axes[0], panels.items()):
        d = df.sort_values(x_col)
        for col, lab in cols.items():
            ax.plot(d[x_col], d[col], "-o", ms=4, label=lab)
        if log_x:
            ax.set_xscale("log")
            ax.minorticks_off()  # otherwise log minor-tick labels clash with the explicit ticks within a decade
        ax.set_xticks(list(d[x_col])); ax.set_xticklabels([f"{v:g}" for v in d[x_col]])
        ax.set_xlabel(xlabel); ax.set_title(title, fontsize=9); ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    fig.tight_layout()
    return save_figure(fig, out_path)
