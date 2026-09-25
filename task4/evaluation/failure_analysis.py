"""Task 4 Step 6 failure analysis under the Vanilla MLS threshold: which
unknown classes are accepted, which CIFAR-10 labels absorb them, and the most
confidently accepted individual examples (unknown class, predicted known
class, score, threshold)."""
from __future__ import annotations

import numpy as np

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def acceptance_by_unknown_class(u, tau, preds, fine_labels, fine_names, group):
    rows = []
    for fid in np.unique(fine_labels):
        m = fine_labels == fid
        acc = m & (u <= tau)
        absorb = np.bincount(preds[acc], minlength=10)
        top = np.argsort(-absorb)[:3]
        rows.append({"group": group, "unknown_class": fine_names[fid], "n": int(m.sum()),
                     "accepted_frac": float(acc.sum() / m.sum()),
                     "top_absorbing_known": ", ".join(f"{CIFAR10_CLASSES[k]} ({absorb[k]})" for k in top if absorb[k] > 0),
                     **{f"to_{c}": int(absorb[k]) for k, c in enumerate(CIFAR10_CLASSES)}})
    return rows


def most_confident_accepted(u, tau, preds, fine_labels, fine_names, group, n=8, one_per_class=True):
    """Lowest-unknownness (most confidently accepted) unknowns. With
    one_per_class, at most one example per unknown class is taken first so the
    listing covers several classes."""
    idx = np.where(u <= tau)[0]
    idx = idx[np.argsort(u[idx])]
    chosen, seen = [], set()
    for i in idx:
        if one_per_class and fine_labels[i] in seen:
            continue
        chosen.append(i); seen.add(fine_labels[i])
        if len(chosen) == n:
            break
    return [{"group": group, "index": int(i), "unknown_class": fine_names[fine_labels[i]],
             "predicted_known_class": CIFAR10_CLASSES[preds[i]], "score_mls": float(u[i]), "threshold": float(tau)}
            for i in chosen]
