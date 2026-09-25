"""Downloads every dataset and pretrained weight the four notebooks need, so the
long runs never stall on a download:

    python scripts/prepare_data.py            # everything
    python scripts/prepare_data.py stl10 pacs # a subset
    python scripts/prepare_data.py --hf       # force the HuggingFace copies

Targets: stl10, cifar, pacs, weights (torchvision + OpenCLIP), adain.

--hf skips the primary hosts (Toronto CIFAR archives, DomainBed's Google Drive
PACS archive) and builds the datasets from the HuggingFace copies directly. The
reported results were produced this way, and the committed PACS split manifest
(shared/splits/pacs_sketch_seed6304.json) refers to the file names this path
writes, so use --hf to reproduce them.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from common.paths import DATA_ROOT  # noqa: E402

force_hf = "--hf" in sys.argv
targets = [a for a in sys.argv[1:] if not a.startswith("--")] or ["stl10", "cifar", "pacs", "weights", "adain"]
DATA_ROOT.mkdir(parents=True, exist_ok=True)

if "stl10" in targets:
    from task1.data.stl10 import load_stl10

    tr, te, classes = load_stl10(DATA_ROOT)
    print("STL-10", len(tr.labels), len(te.labels), classes)

if "cifar" in targets:
    from task4.data.cifar_source import _from_huggingface, load_cifar

    for name, train in [("cifar10", True), ("cifar10", False), ("cifar100", False)]:
        if force_hf:
            _from_huggingface(DATA_ROOT, name, train)  # writes the .npz cache that load_cifar reads first
        ds = load_cifar(DATA_ROOT, name, train)
        print(name, "train" if train else "test", len(ds), ds.classes[:3])

if "pacs" in targets:
    from shared.pacs import PACS_DOMAINS, _normalize_domain_dir, _pacs_from_huggingface, ensure_pacs, list_domain_files

    root = DATA_ROOT / "pacs"
    if force_hf:
        try:
            for d in PACS_DOMAINS:
                _normalize_domain_dir(root, d)
        except FileNotFoundError:
            _pacs_from_huggingface(root)
    root = ensure_pacs(root)
    for d in PACS_DOMAINS:
        paths, labels = list_domain_files(root, d)
        print("PACS", d, len(paths))

if "weights" in targets:
    import open_clip
    from torchvision.models import (ResNet18_Weights, ResNet50_Weights, ViT_B_16_Weights, resnet18, resnet50,
                                    vit_b_16)

    resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", force_quick_gelu=True)
    print("torchvision + OpenCLIP weights cached")

if "adain" in targets:
    from task1.data.make_cue_conflicts import AdaIN

    AdaIN(weights_dir=REPO / "task1" / "checkpoints" / "adain")
    print("AdaIN weights loaded")
