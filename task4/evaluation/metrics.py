"""Task 4 Step 6 metrics. Every method is an unknownness score u(x) (larger =
more novel). For each (model, score):
  * AUROC for Known-vs-Near, Known-vs-Far and Known-vs-All (known = CIFAR-10 test);
  * tau = 95th percentile of u on the CIFAR-10 *validation* split; accept iff u <= tau;
  * achieved CIFAR-10 test acceptance, near/far/all rejection rates, and
    FPR@95TPR = fraction of unknowns incorrectly accepted (= 1 - rejection).
"""
from __future__ import annotations

import numpy as np

from common.metrics import auroc, threshold_at_known_percentile


def osr_row(name: str, u_val, u_test, u_near, u_far, csa: float | None = None) -> dict:
    u_all = np.concatenate([u_near, u_far])
    tau = threshold_at_known_percentile(u_val, 95.0)
    row = {"name": name}
    if csa is not None:
        row["closed_set_accuracy"] = csa
    row.update({
        "auroc_near": auroc(u_test, u_near), "auroc_far": auroc(u_test, u_far), "auroc_all": auroc(u_test, u_all),
        "tau": tau,
        "val_acceptance": float(np.mean(u_val <= tau)),
        "test_known_acceptance": float(np.mean(u_test <= tau)),
        "near_rejection": float(np.mean(u_near > tau)), "far_rejection": float(np.mean(u_far > tau)),
        "all_rejection": float(np.mean(u_all > tau)),
        "fpr95_near": float(np.mean(u_near <= tau)), "fpr95_far": float(np.mean(u_far <= tau)),
        "fpr95_all": float(np.mean(u_all <= tau)),
    })
    return row


def score_group_auroc_table(known_scores, near_scores, far_scores):
    """Kept for backwards compatibility: AUROC-only rows."""
    rows = []
    for name in known_scores:
        rows.append({"score": name, "auroc_known_vs_near": auroc(known_scores[name], near_scores[name]),
                     "auroc_known_vs_far": auroc(known_scores[name], far_scores[name]),
                     "auroc_known_vs_all": auroc(known_scores[name], np.concatenate([near_scores[name], far_scores[name]]))})
    return rows
