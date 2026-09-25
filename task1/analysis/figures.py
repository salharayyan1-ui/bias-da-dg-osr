"""Report figures for Task 1 (kept out of the notebook so cells stay short)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common.plotting import save_figure

MODEL_LABELS = {"resnet50": "ResNet-50", "vit_b_16": "ViT-B/16", "clip_head": "CLIP head",
                "clip_zeroshot": "CLIP zero-shot"}
BACKBONE_LABELS = {"resnet50": "ResNet-50", "vit_b_16": "ViT-B/16", "clip_vit_b_32": "CLIP ViT-B/32"}
MODEL_COLORS = {"resnet50": "#1f77b4", "vit_b_16": "#d62728", "clip_head": "#2ca02c", "clip_zeroshot": "#9467bd"}


def intervention_examples(rows_of_images, row_titles, col_titles, out_path, cell=1.6):
    """rows_of_images[r][c] = HxWx3 uint8."""
    nr, nc = len(rows_of_images), len(rows_of_images[0])
    fig, axes = plt.subplots(nr, nc, figsize=(cell * nc, cell * nr), squeeze=False)
    for r in range(nr):
        for c in range(nc):
            ax = axes[r, c]
            ax.imshow(rows_of_images[r][c])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=8)
    fig.tight_layout()
    return save_figure(fig, out_path)


def curves(x, series: dict, ylabel: str, xlabel: str, out_path, errors: dict | None = None, title=None, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for name, y in series.items():
        e = None if errors is None else errors.get(name)
        ax.errorbar(x, y, yerr=e, marker="o", ms=4, capsize=2, label=MODEL_LABELS.get(name, name),
                    color=MODEL_COLORS.get(name))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xticks(list(x))
    if own:
        ax.legend(fontsize=7)
        return save_figure(fig, out_path)


def translation_figure(df_mean, out_path):
    """df_mean: rows with model, pixels, accuracy, consistency and std columns."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    for metric, ax in zip(["accuracy", "consistency"], axes):
        series, errs = {}, {}
        for m, g in df_mean.groupby("model", sort=False):
            g = g.sort_values("pixels")
            series[m] = g[metric].values
            errs[m] = g.get(f"{metric}_std_over_directions", 0 * g[metric]).fillna(0).values
        x = sorted(df_mean["pixels"].unique())
        curves(x, series, metric.capitalize(), "Displacement (pixels)", None, errs, ax=ax)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return save_figure(fig, out_path)


def decision_bars(rows, out_path):
    """Stacked shape / texture / other fractions per model, annotated with
    shape bias and coverage."""
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    names = [r["model"] for r in rows]
    x = np.arange(len(rows))
    tot = np.array([r["n_total"] for r in rows], dtype=float)
    parts = [("n_shape", "shape (content)", "#4c72b0"), ("n_texture", "texture (style)", "#dd8452"),
             ("n_other", "other", "#bbbbbb")]
    bottom = np.zeros(len(rows))
    for key, lab, col in parts:
        v = np.array([r[key] for r in rows]) / tot
        ax.bar(x, v, bottom=bottom, color=col, label=lab, width=0.6)
        bottom += v
    for i, r in enumerate(rows):
        ax.text(i, 1.01, f"SB={r['shape_bias_pct']:.0f}%\ncov={r['coverage_pct']:.0f}%", ha="center",
                va="bottom", fontsize=7)
    ax.set_xticks(x, [MODEL_LABELS.get(n, n) for n in names], fontsize=8)
    ax.set_ylim(0, 1.22); ax.set_ylabel("Fraction of cue-conflict images")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    return save_figure(fig, out_path)


def cue_conflict_gallery(items, out_path, n_cols=4):
    """items: dicts with content, style, stylized (uint8 images), title (str)
    and caption (str, e.g. model predictions). Each item takes three
    thumbnails: content | style | stylized."""
    n = len(items)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, 3 * n_cols, figsize=(1.15 * 3 * n_cols, 1.55 * n_rows), squeeze=False,
                             gridspec_kw={"wspace": 0.04, "hspace": 0.6})
    for ax in axes.ravel():
        ax.axis("off")
    for k, it in enumerate(items):
        r, c0 = divmod(k, n_cols)
        for j, key in enumerate(["content", "style", "stylized"]):
            ax = axes[r, 3 * c0 + j]
            ax.imshow(it[key])
            if j == 0:
                ax.set_title(it.get("title", ""), fontsize=6, loc="left")
            if j == 1 and it.get("caption"):
                ax.text(0.5, -0.04, it["caption"], transform=ax.transAxes, ha="center", va="top", fontsize=5.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.04)
    return save_figure(fig, out_path)


def image_grid(images, titles, out_path, n_cols=8, cell=1.35, fontsize=6):
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell * n_cols, (cell + 0.35) * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (im, t) in enumerate(zip(images, titles)):
        ax = axes.ravel()[k]
        ax.imshow(im)
        ax.set_title(t, fontsize=fontsize)
    fig.tight_layout()
    return save_figure(fig, out_path)


def tsne_grid(panels, class_names, out_path):
    """panels[(row_name, col_name)] = (coords [2N,2], class_labels [2N], is_transformed [2N])."""
    rows = list(dict.fromkeys(k[0] for k in panels))
    cols = list(dict.fromkeys(k[1] for k in panels))
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.0 * len(cols), 2.9 * len(rows)), squeeze=False)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            if (r, c) not in panels:
                ax.axis("off"); continue
            xy, cls, tr = panels[(r, c)]
            for k in range(len(class_names)):
                m0 = (cls == k) & (~tr)
                m1 = (cls == k) & tr
                ax.scatter(xy[m0, 0], xy[m0, 1], s=5, color=cmap(k), marker="o", alpha=0.6, linewidths=0)
                ax.scatter(xy[m1, 0], xy[m1, 1], s=9, color=cmap(k), marker="x", alpha=0.7, linewidths=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(c, fontsize=9)
            if j == 0:
                ax.set_ylabel(BACKBONE_LABELS.get(r, r), fontsize=9)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(k), label=n, markersize=6)
               for k, n in enumerate(class_names)]
    handles += [Line2D([0], [0], marker="o", color="gray", linestyle="", label="clean", markersize=5),
                Line2D([0], [0], marker="x", color="gray", linestyle="", label="transformed", markersize=5)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return save_figure(fig, out_path)


def stability_vs_consistency(points, out_path):
    """points: list of dicts(model, intervention, I_T_normalized, consistency)."""
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    markers = {"grayscale": "o", "hue_rot_180": "s", "cue_conflict": "*", "translation_32": "^",
               "patch_shuffle_4x4": "D", "translation_8": "v", "translation_16": ">"}
    for p in points:
        ax.scatter(p["normalized_stability"], p["consistency"], color=MODEL_COLORS.get(p["model"]),
                   marker=markers.get(p["intervention"], "o"), s=40, alpha=0.85)
    h1 = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=MODEL_LABELS[m], markersize=7)
          for m, c in MODEL_COLORS.items()]
    h2 = [Line2D([0], [0], marker=mk, color="gray", linestyle="", label=k, markersize=6)
          for k, mk in markers.items() if any(p["intervention"] == k for p in points)]
    ax.legend(handles=h1 + h2, fontsize=6, loc="lower right")
    ax.set_xlabel("Normalized representation stability")
    ax.set_ylabel("Prediction consistency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, out_path)


def grouped_bars(df, index_col, group_col, value_col, out_path, ylabel, title=None, labels=None):
    piv = df.pivot(index=index_col, columns=group_col, values=value_col)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    n = len(piv.columns)
    w = 0.8 / n
    x = np.arange(len(piv.index))
    for k, col in enumerate(piv.columns):
        ax.bar(x + (k - (n - 1) / 2) * w, piv[col].values, w, label=(labels or {}).get(col, col),
               color=MODEL_COLORS.get(col))
    ax.set_xticks(x, piv.index, fontsize=8, rotation=15)
    ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, ncol=min(n, 4))
    fig.tight_layout()
    return save_figure(fig, out_path)
