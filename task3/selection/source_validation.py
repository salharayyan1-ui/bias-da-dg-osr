"""Task 3 checkpoint selection: mean macro-F1 over the three SOURCE validation
splits, computed every epoch inside shared/pacs_engine.py::train_method (the
same rule as Task 2). No Sketch image is involved."""
from shared.pacs_engine import evaluate_source_val, source_val_loaders  # noqa: F401
