"""Task 2 Step 1: Source-only ERM (cross-entropy on domain-balanced source
batches, no alignment term). Its checkpoint is reused unchanged as the Task 3
ERM baseline."""
from shared.pacs_engine import MethodSpec


def make_method() -> MethodSpec:
    return MethodSpec(name="source_only", uses_target=False)
