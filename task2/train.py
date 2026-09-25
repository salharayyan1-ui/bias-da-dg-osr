"""Task 2 training entry point (the notebook calls the same functions).

    python task2/train.py --run source_only
    python task2/train.py --run dan            # lambda_MMD = 1 (main comparison)
    python task2/train.py --run dann
    python task2/train.py --run cdan
    python task2/train.py --run dan_lambda0.1  # controlled design study
    python task2/train.py --run dan_lambda10
    python task2/train.py --all

Source-only must be trained first: it is also the Task 3 ERM checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import DATA_ROOT, checkpoints_dir, results_dir  # noqa: E402
from shared.pacs import ensure_pacs, load_or_create_splits  # noqa: E402
from shared.pacs_engine import PACSTrainConfig, run_or_load  # noqa: E402
from task2.methods import cdan, dan, dann, source_only  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SPLIT_MANIFEST = REPO / "shared" / "splits" / "pacs_sketch_seed6304.json"

MAIN_RUNS = {
    "source_only": source_only.make_method,
    "dan": partial(dan.make_method, lambda_mmd=1.0),
    "dann": partial(dann.make_method, max_alpha=1.0),
    "cdan": partial(cdan.make_method, max_alpha=1.0),
}
# Controlled design study: lambda_MMD in {0.1, 1, 10}; lambda = 1 is the main DAN run.
STUDY_LAMBDAS = (0.1, 1.0, 10.0)
STUDY_RUNS = {f"dan_lambda{lam:g}": partial(dan.make_method, lambda_mmd=lam) for lam in STUDY_LAMBDAS if lam != 1.0}
# Supplementary second study (the assignment's other option): maximum GRL strength
# for DANN in {0.25, 0.5, 1}, same schedule shape; max_alpha = 1 is the main DANN run.
DANN_ALPHAS = (0.25, 0.5, 1.0)
DANN_STUDY_RUNS = {f"dann_alpha{a:g}": partial(dann.make_method, max_alpha=a) for a in DANN_ALPHAS if a != 1.0}
ALL_RUNS = {**MAIN_RUNS, **STUDY_RUNS, **DANN_STUDY_RUNS}


def dann_study_run_name(a: float) -> str:
    return "dann" if a == 1.0 else f"dann_alpha{a:g}"


def study_run_name(lam: float) -> str:
    return "dan" if lam == 1.0 else f"dan_lambda{lam:g}"


def get_splits(data_root=None):
    root = ensure_pacs(Path(data_root) if data_root else DATA_ROOT / "pacs")
    return load_or_create_splits(root, SPLIT_MANIFEST)


def train_run(name: str, splits, cfg: PACSTrainConfig, retrain: bool = False):
    return run_or_load(ALL_RUNS[name], name, splits, cfg, results_dir(2), checkpoints_dir(2), retrain=retrain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(ALL_RUNS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    splits = get_splits(args.data_root)
    cfg = PACSTrainConfig(num_workers=args.num_workers)
    for name in (ALL_RUNS if args.all else [args.run]):
        _, summary = train_run(name, splits, cfg, args.retrain)
        print(name, "best epoch", summary["best_epoch"], "mean source-val F1", round(summary["best_mean_val_macro_f1"], 4))


if __name__ == "__main__":
    main()
