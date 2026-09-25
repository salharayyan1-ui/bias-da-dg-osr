"""Saves penultimate features, known-class logits (and PROSER dummy logits /
RPL distances) for CIFAR-10 train (unaugmented) / val / test and the CIFAR-100
near / far unknowns to task4/cache/full/<method>_<split>.npz. Every score is
computed from these saved arrays.

    python task4/extract_outputs.py --method vanilla
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.paths import DATA_ROOT, cache_dir, checkpoints_dir, results_dir  # noqa: E402
from task4.pipeline import eval_loaders, extract_all, train_or_load  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["vanilla", "gcsc", "proser", "rpl"], required=True)
    ap.add_argument("--num-workers", type=int, default=4)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = checkpoints_dir(4) / f"{a.method}.pth"
    if not ckpt.exists():
        sys.exit(f"{ckpt} missing: train first")
    ext, extra, _ = train_or_load(a.method, DATA_ROOT, checkpoints_dir(4), results_dir(4), device, a.num_workers)
    out_dir = cache_dir(4) / "full"
    out_dir.mkdir(exist_ok=True)
    outs = extract_all(a.method, ext, extra, eval_loaders(DATA_ROOT, a.num_workers), device, out_dir, overwrite=True)
    print({k: v["logits"].shape for k, v in outs.items()})
