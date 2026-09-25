"""Minimal, dependency-free run logging: JSON-lines metrics + a final summary JSON.
Keeping this trivial (rather than pulling in wandb/tensorboard) means every number
that ends up in the report traces back to a plain-text file that's easy to diff,
grep, and commit, per the "Before You Submit" checklist.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


class RunLogger:
    def __init__(self, out_dir: str | Path, run_name: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.metrics_path = self.out_dir / f"{run_name}.metrics.jsonl"
        self.summary_path = self.out_dir / f"{run_name}.summary.json"
        self._t0 = time.time()
        # Truncate any previous log for this run name so re-runs don't append stale rows.
        self.metrics_path.write_text("")

    def log(self, step: int, **metrics: Any) -> None:
        row: Dict[str, Any] = {"step": step, "t": round(time.time() - self._t0, 2)}
        row.update(metrics)
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def save_summary(self, **summary: Any) -> None:
        summary = dict(summary)
        summary["run_name"] = self.run_name
        summary["wall_clock_seconds"] = round(time.time() - self._t0, 2)
        self.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    def save_json(self, name: str, obj: Any) -> Path:
        path = self.out_dir / name
        path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
        return path


def save_table(rows, out_path: str | Path, float_fmt: str = "%.4f"):
    """Writes a list-of-dicts table as both JSON and CSV (same stem) so every
    number in the report traces to a machine-readable file. Returns a pandas
    DataFrame for display in notebooks."""
    import pandas as pd

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path.with_suffix(".json").write_text(json.dumps(rows, indent=2, default=_json_default))
    df.to_csv(out_path.with_suffix(".csv"), index=False, float_format=float_fmt)
    return df


def save_json(obj: Any, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return out_path


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
