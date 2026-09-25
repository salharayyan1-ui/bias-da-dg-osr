"""Task 3 evaluation, split in two phases so the ordering required by the
spec is explicit:

1. `source_diagnostics` - source validation accuracy/F1 (mean and worst),
   3-way source-domain separability and the sharpness proxy. Uses only the
   three source domains.
2. `sketch_predictions` - loads Sketch for the first time. Call it only after
   every Task 3 checkpoint and setting has been fixed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.pacs_engine import (SOURCE_DOMAINS, PACSTrainConfig, eval_transform, predict,  # noqa: E402
                                source_val_loaders, target_eval_loader)
from task3.evaluation.sharpness import build_fixed_sharpness_batch, compute_sharpness  # noqa: E402
from task3.evaluation.source_domain_separability import compute_source_domain_separability  # noqa: E402


def source_diagnostics(models: dict, source_splits: dict, cfg: PACSTrainConfig, radii=(0.05,), seed: int = 6304):
    assert "sketch" not in source_splits
    val_loaders = source_val_loaders(source_splits, cfg)
    sx, sy, chosen = build_fixed_sharpness_batch(source_splits, SOURCE_DOMAINS, eval_transform(), 32, seed)
    out = {}
    for name, model in models.items():
        res = {d: predict(model, val_loaders[d], cfg.device) for d in SOURCE_DOMAINS}
        res["source_domain_separability"] = compute_source_domain_separability(
            {d: res[d]["features"] for d in SOURCE_DOMAINS}, seed=seed)
        res["sharpness"] = {r: compute_sharpness(model, sx, sy, rho=r, device=cfg.device) for r in radii}
        out[name] = res
    return out, chosen


def sketch_predictions(models: dict, full_splits: dict, cfg: PACSTrainConfig):
    loader = target_eval_loader(full_splits, cfg)
    return {name: predict(model, loader, cfg.device) for name, model in models.items()}
