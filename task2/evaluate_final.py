"""Task 2 Step 5 (+ Step 6 study): final evaluation after every checkpoint
and setting is fixed. This is the only Task 2 code that reads Sketch labels.

    python task2/evaluate_final.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.pacs_engine import (SOURCE_DOMAINS, PACSTrainConfig, predict, source_val_loaders,  # noqa: E402
                                target_eval_loader)
from task2.evaluation.domain_separability import compute_domain_separability  # noqa: E402


def evaluate_models(models: dict, splits: dict, cfg: PACSTrainConfig, seed: int = 6304) -> dict:
    """models[name] = trained PACSNet. Returns, per run, predictions on every
    source validation split and on all Sketch images, plus the source-vs-target
    domain separability of the frozen features."""
    val_loaders = source_val_loaders(splits, cfg)
    tgt_loader = target_eval_loader(splits, cfg)
    out = {}
    for name, model in models.items():
        res = {d: predict(model, val_loaders[d], cfg.device) for d in SOURCE_DOMAINS}
        res["sketch"] = predict(model, tgt_loader, cfg.device)
        src_feats = np.concatenate([res[d]["features"] for d in SOURCE_DOMAINS])
        res["domain_separability"] = compute_domain_separability(src_feats, res["sketch"]["features"], seed=seed)
        out[name] = res
    return out


if __name__ == "__main__":
    print("Run task2/task2.ipynb from the 'Final evaluation' section; it calls evaluate_models() "
          "and writes every table/figure. (Kept as a function so the notebook and scripts share it.)")
