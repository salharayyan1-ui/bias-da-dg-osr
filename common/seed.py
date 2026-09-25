"""Central seeding utility. Call set_seed(6304) at the top of every entry-point
script so every stochastic operation in the assignment (splits, subset selection,
patch shuffles, model init, dataloader shuffling, mixup lambda, SAM's random
tie-breaks, etc.) is reproducible from the same seed, as required throughout the PDF.
"""
from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch not required to import this module
    torch = None


def set_seed(seed: int = 6304, deterministic: bool = True) -> None:
    """Seed python, numpy, and (if available) torch CPU/CUDA RNGs.

    Args:
        seed: the assignment specifies seed 6304 everywhere.
        deterministic: if True and torch is available, also request
            deterministic cuDNN algorithms. This can slow training down; set
            False for the controlled-design-study sweeps if you need speed and
            don't need bitwise reproducibility across those runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def seeded_generator(seed: int = 6304):
    """A torch.Generator seeded independently, for DataLoader(generator=...)."""
    if torch is None:
        raise ImportError("torch is required for seeded_generator")
    g = torch.Generator()
    g.manual_seed(seed)
    return g
