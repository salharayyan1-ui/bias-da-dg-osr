"""Repo-relative paths so notebooks and scripts write to the same places no
matter which directory they are launched from.

Outputs (results, checkpoints, caches, figures) go under the repo by default.
Setting the environment variable PA1_OUTPUT_ROOT redirects all of them to
another directory; this is used for quick pipeline checks so they can never
overwrite (or be mistaken for) the real runs' checkpoints and results.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PA1_DATA_ROOT", REPO_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("PA1_OUTPUT_ROOT", REPO_ROOT))
FIGURES_DIR = OUTPUT_ROOT / "report" / "figures"


def task_dir(task: int) -> Path:
    return REPO_ROOT / f"task{task}"


def _out(task: int, name: str) -> Path:
    p = OUTPUT_ROOT / f"task{task}" / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(task: int) -> Path:
    return _out(task, "results")


def checkpoints_dir(task: int) -> Path:
    return _out(task, "checkpoints")


def cache_dir(task: int) -> Path:
    return _out(task, "cache")


def figure_path(name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / name
