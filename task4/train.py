"""Task 4 training entry point (the notebook calls the same function).

    python task4/train.py --method vanilla
    python task4/train.py --method gcsc
    python task4/train.py --method proser   # needs the Vanilla checkpoint
    python task4/train.py --method rpl      # optional extension
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.paths import DATA_ROOT, checkpoints_dir, results_dir  # noqa: E402
from task4.pipeline import train_or_load  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["vanilla", "gcsc", "proser", "rpl"], required=True)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, summary = train_or_load(a.method, DATA_ROOT, checkpoints_dir(4), results_dir(4), device, a.num_workers, a.retrain)
    print(summary)
