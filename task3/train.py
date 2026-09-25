"""Task 3 training entry point (the notebook calls the same functions).
ERM is NOT trained here: it is Task 2's Source-only checkpoint.

    python task3/train.py --run dan_dg          # lambda_DG = 1 (main comparison)
    python task3/train.py --run sam             # rho = 0.05 (main comparison)
    python task3/train.py --run dan_dg_lambda0.1   # controlled design study
    python task3/train.py --run dan_dg_lambda10
    python task3/train.py --all

No Sketch image is ever loaded: training receives splits restricted to the
three source domains, so any accidental access to Sketch raises a KeyError.
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import DATA_ROOT, checkpoints_dir, results_dir  # noqa: E402
from shared.pacs import ensure_pacs, load_or_create_splits  # noqa: E402
from shared.pacs_engine import SOURCE_DOMAINS, PACSTrainConfig, run_or_load  # noqa: E402
from task3.methods import dan_dg, sam  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SPLIT_MANIFEST = REPO / "shared" / "splits" / "pacs_sketch_seed6304.json"
TASK2_ERM_CHECKPOINT = checkpoints_dir(2) / "source_only.pth"

MAIN_RUNS = {"dan_dg": partial(dan_dg.make_method, lambda_dg=1.0), "sam": partial(sam.make_method, rho=0.05)}
STUDY_LAMBDAS = (0.1, 1.0, 10.0)
STUDY_RUNS = {f"dan_dg_lambda{l:g}": partial(dan_dg.make_method, lambda_dg=l) for l in STUDY_LAMBDAS if l != 1.0}
# Optional second study (off by default; the assignment asks for one): SAM rho in {0.01, 0.05, 0.1}
SAM_RHOS = (0.01, 0.05, 0.1)
SAM_STUDY_RUNS = {f"sam_rho{r:g}": partial(sam.make_method, rho=r) for r in SAM_RHOS if r != 0.05}
ALL_RUNS = {**MAIN_RUNS, **STUDY_RUNS, **SAM_STUDY_RUNS}


def study_run_name(lam: float) -> str:
    return "dan_dg" if lam == 1.0 else f"dan_dg_lambda{lam:g}"


def sam_run_name(rho: float) -> str:
    return "sam" if rho == 0.05 else f"sam_rho{rho:g}"


def get_source_splits(data_root=None):
    """Loads the shared manifest and drops Sketch entirely."""
    root = ensure_pacs(Path(data_root) if data_root else DATA_ROOT / "pacs")
    splits = load_or_create_splits(root, SPLIT_MANIFEST)
    return {d: splits[d] for d in SOURCE_DOMAINS}


def train_run(name: str, source_splits, cfg: PACSTrainConfig, retrain: bool = False):
    assert "sketch" not in source_splits
    return run_or_load(ALL_RUNS[name], name, source_splits, cfg, results_dir(3), checkpoints_dir(3), retrain=retrain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(ALL_RUNS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    splits = get_source_splits(args.data_root)
    cfg = PACSTrainConfig(num_workers=args.num_workers)
    names = list(MAIN_RUNS) + list(STUDY_RUNS) if args.all else [args.run]
    for name in names:
        _, summary = train_run(name, splits, cfg, args.retrain)
        print(name, "best epoch", summary["best_epoch"], "mean source-val F1", round(summary["best_mean_val_macro_f1"], 4))


if __name__ == "__main__":
    main()
