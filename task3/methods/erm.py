"""Task 3 Step 1: ERM baseline = the Task 2 Source-only checkpoint, loaded
unchanged (never retrained)."""
from pathlib import Path

import torch

from task2.models.backbone import PACSNet


def load_erm_baseline(checkpoint_path, device="cpu"):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run Task 2 (Source-only) first.")
    payload = torch.load(path, map_location="cpu")
    model = PACSNet(n_classes=7, pretrained=False)
    model.load_state_dict(payload["model"])
    return model.to(device), payload["summary"]
