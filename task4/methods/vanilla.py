"""Task 4 Step 1: Vanilla closed-set ResNet-18, plain crop+flip augmentation,
cross-entropy only."""
from __future__ import annotations

from task4.methods._common import OSRTrainConfig, train_classifier


def run(model, train_loader, val_loader, logger, cfg: OSRTrainConfig | None = None):
    cfg = cfg or OSRTrainConfig()
    return train_classifier(model, train_loader, val_loader, cfg, logger, run_name="vanilla")
