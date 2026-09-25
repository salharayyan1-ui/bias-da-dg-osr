"""PACS dataset loading, shared by Task 2 (UDA) and Task 3 (DG).

Expects the standard PACS directory layout:
    <root>/<domain>/<class>/<image>.jpg
with domain in {photo, art_painting, cartoon, sketch} (case-insensitive, we
normalize to lowercase-with-underscore below) and the 7 canonical PACS classes.

Both Task 2 and Task 3 MUST reuse the same source splits (per the PDF), which
is why this lives in shared/ rather than being duplicated per task.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

# Canonical class order — fixed so label indices are identical across every
# domain, every task, and every run.
PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
PACS_DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]

try:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object
    Image = None


def _normalize_domain_dir(root: Path, domain: str) -> Path:
    """PACS archives ship the art-painting folder under a few different
    spellings ('art_painting', 'art painting', 'ArtPainting', ...) -- try the
    common variants rather than failing on a naming mismatch."""
    candidates = [domain, domain.replace("_", " "), domain.replace("_", "-"),
                  "".join(w.capitalize() for w in domain.split("_"))]
    for c in candidates:
        p = root / c
        if p.is_dir():
            return p
    raise FileNotFoundError(f"could not find a directory for domain '{domain}' under {root} "
                             f"(tried: {candidates})")


def list_domain_files(root: str | Path, domain: str) -> tuple[list[str], np.ndarray]:
    """Returns (file_paths, labels) for one PACS domain, labels indexed into
    PACS_CLASSES."""
    domain_dir = _normalize_domain_dir(Path(root), domain)
    paths, labels = [], []
    for class_idx, class_name in enumerate(PACS_CLASSES):
        class_dir = domain_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"missing class dir {class_dir}")
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(img_path))
                labels.append(class_idx)
    return paths, np.array(labels)


def stratified_split_indices(labels: np.ndarray, val_fraction: float = 0.2, seed: int = 6304):
    """Same recipe as task1/data/make_subset.stratified_train_val_split,
    duplicated here (rather than imported) so shared/ has no dependency on
    task1/, per "only genuinely shared utilities" in the repo layout guidance."""
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        n_val = max(1, int(round(len(idx_c) * val_fraction)))
        val_idx.append(idx_c[:n_val])
        train_idx.append(idx_c[n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


class PACSImageDataset(Dataset):
    """A plain (paths, labels) image dataset for one PACS domain (or a subset
    of it, via `indices`)."""

    def __init__(self, paths: Sequence[str], labels: np.ndarray, indices: np.ndarray | None = None, transform=None):
        self.paths = list(paths)
        self.labels = np.asarray(labels)
        self.indices = np.arange(len(self.paths)) if indices is None else np.asarray(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def build_pacs_splits(root: str | Path, seed: int = 6304):
    """For every domain: (paths, labels, train_idx, val_idx) with a stratified
    80/20 split (seed 6304). Sketch also gets a split, but Tasks 2/3 only ever
    use Sketch as a whole (unlabeled adaptation set / final evaluation)."""
    splits = {}
    for domain in PACS_DOMAINS:
        paths, labels = list_domain_files(root, domain)
        train_idx, val_idx = stratified_split_indices(labels, 0.2, seed)
        splits[domain] = {"paths": paths, "labels": labels, "train_idx": train_idx, "val_idx": val_idx}
    return splits


def save_splits_manifest(splits: dict, out_path: str | Path, root: str | Path) -> None:
    """Persists paths (relative to the PACS root) and indices as JSON so the
    exact split is reproducible and diffable in git."""
    import json

    root = Path(root).resolve()
    payload = {}
    for domain, d in splits.items():
        payload[domain] = {
            "paths": [Path(p).resolve().relative_to(root).as_posix() for p in d["paths"]],
            "labels": d["labels"].tolist(),
            "train_idx": d["train_idx"].tolist(),
            "val_idx": d["val_idx"].tolist(),
        }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload))


def load_or_create_splits(root: str | Path, manifest_path: str | Path, seed: int = 6304):
    """Single source of truth for Tasks 2 and 3: the first call writes
    shared/splits/pacs_sketch_seed6304.json, every later call (either task)
    reads it back, so both tasks use byte-identical splits."""
    import json

    root, manifest_path = Path(root), Path(manifest_path)
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        splits = {}
        for domain, d in payload.items():
            splits[domain] = {
                "paths": [str(root / p) for p in d["paths"]],
                "labels": np.asarray(d["labels"]),
                "train_idx": np.asarray(d["train_idx"]),
                "val_idx": np.asarray(d["val_idx"]),
            }
        missing = [p for d in splits.values() for p in d["paths"][:5] if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"split manifest paths not found under {root}: {missing[:3]}")
        return splits
    splits = build_pacs_splits(root, seed)
    save_splits_manifest(splits, manifest_path, root)
    return splits


PACS_GDRIVE_ID = "1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd"  # the PACS archive used by DomainBed's download script
PACS_HF = "https://huggingface.co/datasets/flwrlabs/pacs/resolve/main/data/train-00000-of-00001.parquet"


def _pacs_from_huggingface(root: Path) -> None:
    """Writes the HuggingFace copy of PACS (`flwrlabs/pacs`, all 9,991 images
    with domain and class labels) into the standard <domain>/<class>/ layout,
    keeping each image's original encoded bytes."""
    import io
    import urllib.request

    import pyarrow.parquet as pq

    local = root.parent / "pacs_hf.parquet"  # e.g. fetched beforehand with: curl -L -C - -o data/pacs_hf.parquet <PACS_HF>
    if local.exists():
        table = pq.read_table(local)
    else:
        print(f"reading {PACS_HF}")
        table = pq.read_table(io.BytesIO(urllib.request.urlopen(PACS_HF, timeout=1200).read()))
    images = table.column("image").to_pylist()
    domains = table.column("domain").to_pylist()
    labels = table.column("label").to_pylist()
    counters = {}
    for img, dom, lab in zip(images, domains, labels):
        d = dom.strip().lower().replace(" ", "_").replace("-", "_")
        c = PACS_CLASSES[int(lab)]
        k = counters.get((d, c), 0)
        counters[(d, c)] = k + 1
        out = root / d / c
        out.mkdir(parents=True, exist_ok=True)
        fmt = (Image.open(io.BytesIO(img["bytes"])).format or "PNG").lower()
        ext = {"jpeg": ".jpg"}.get(fmt, "." + fmt)
        (out / f"{k:05d}{ext}").write_bytes(img["bytes"])


def ensure_pacs(root: str | Path) -> Path:
    """Returns the PACS root (containing photo/, art_painting/, cartoon/,
    sketch/). If missing, downloads the DomainBed archive with `gdown`; if
    Google Drive is unreachable, falls back to the HuggingFace copy."""
    root = Path(root)
    try:
        for d in PACS_DOMAINS:
            _normalize_domain_dir(root, d)
        return root
    except FileNotFoundError:
        pass
    try:
        import zipfile

        import gdown

        root.parent.mkdir(parents=True, exist_ok=True)
        zip_path = root.parent / "PACS.zip"
        if not zip_path.exists():
            if gdown.download(id=PACS_GDRIVE_ID, output=str(zip_path), quiet=False) is None:
                raise RuntimeError("gdown download failed")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root.parent)
        extracted = root.parent / "kfold"
        if extracted.is_dir() and not root.exists():
            extracted.rename(root)
    except Exception as e:  # Google Drive quota / network
        print(f"Google Drive download failed ({type(e).__name__}: {e}); using the HuggingFace copy")
        _pacs_from_huggingface(root)
    for d in PACS_DOMAINS:
        _normalize_domain_dir(root, d)
    return root
