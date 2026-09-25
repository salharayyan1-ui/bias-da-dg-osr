"""Executes task notebooks headless.

    python scripts/run_notebook.py task1 task2        # full run, outputs saved into the .ipynb
    python scripts/run_notebook.py --quick task2      # QUICK_RUN pipeline check

--quick flips QUICK_RUN to True in memory, requires PA1_OUTPUT_ROOT to be set
(so results/checkpoints go to a separate directory) and saves the executed
copy there instead of overwriting the notebook.
"""
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
args = sys.argv[1:]
quick = "--quick" in args
tasks = [a for a in args if not a.startswith("--")] or ["task1", "task2", "task3", "task4"]
if quick and not os.environ.get("PA1_OUTPUT_ROOT"):
    sys.exit("--quick requires PA1_OUTPUT_ROOT so quick outputs cannot overwrite real ones")

for task in tasks:
    path = REPO / task / f"{task}.ipynb"
    nb = nbformat.read(path, as_version=4)
    if quick:
        for c in nb.cells:
            if c.cell_type == "code":
                c.source = c.source.replace("QUICK_RUN = False", "QUICK_RUN = True")
        out = Path(os.environ["PA1_OUTPUT_ROOT"]) / f"{task}_quick.ipynb"
    else:
        out = path
    print(f"[{time.strftime('%H:%M:%S')}] running {task} ({'quick' if quick else 'full'}) -> {out}", flush=True)
    t0 = time.time()
    status = "FAILED"
    try:
        NotebookClient(nb, timeout=None, kernel_name="python3",
                       resources={"metadata": {"path": str(path.parent)}}).execute()
        status = "OK"
    finally:
        nbformat.write(nb, out)
        print(f"[{time.strftime('%H:%M:%S')}] {task} {status} after {(time.time() - t0) / 60:.1f} min", flush=True)
