"""Task 4 end-to-end pipeline shared by `train.py`, `extract_outputs.py`,
`evaluate_osr.py` and the notebook.

Data: CIFAR-10 (known) with a stratified 90/10 split of the training set
(seed 6304); the complete CIFAR-10 test set; the fixed near/far CIFAR-100
*test* classes (800 images each), used only after training, checkpoint
selection and score definitions are fixed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.logging import RunLogger
from common.seed import seeded_generator, set_seed

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def transforms_for(method: str):
    from torchvision import transforms

    if method == "gcsc":
        from task4.methods.gcsc import build_train_transform

        train = build_train_transform()  # crop + flip + RandAugment(2, 9) + ToTensor + Normalize
    else:
        train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    evalt = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    return train, evalt


def train_val_loaders(data_root, method: str, batch_size=128, num_workers=4, seed=6304, quick=False):
    from task4.data.cifar10 import build_cifar10_datasets

    train_t, eval_t = transforms_for(method)
    train_ds, val_ds, _ = build_cifar10_datasets(str(data_root), train_t, eval_t)
    if quick:
        from torch.utils.data import Subset

        train_ds = Subset(train_ds, list(range(0, len(train_ds), 50)))
        val_ds = Subset(val_ds, list(range(0, len(val_ds), 10)))
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
                              generator=seeded_generator(seed), **kw)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, **kw)
    return train_loader, val_loader


def train_or_load(method: str, data_root, ckpt_dir: Path, results_dir: Path, device, num_workers=4,
                  retrain=False, quick=False, seed=6304):
    """Returns (extractor, extra_head_or_None, summary). Checkpoints are the
    ones with the highest CIFAR-10 validation accuracy."""
    from task4.models.resnet_cifar import PenultimateExtractor, build_cifar_resnet18

    ckpt = Path(ckpt_dir) / f"{method}.pth"
    set_seed(seed)
    extractor = PenultimateExtractor(build_cifar_resnet18(n_classes=10))
    extra = None
    if method == "proser_last" and not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} is written by the PROSER run: train 'proser' first")
    if method in ("proser", "proser_last"):
        from task4.methods.proser import DummyHead

        extra = DummyHead(extractor.feature_dim, 5)
    elif method == "rpl":
        from task4.methods.rpl import ReciprocalPoints

        extra = ReciprocalPoints(10, extractor.feature_dim)

    if ckpt.exists() and (not retrain or method == "proser_last"):
        payload = torch.load(ckpt, map_location="cpu")
        extractor.net.load_state_dict(payload["net"])
        if extra is not None:
            extra.load_state_dict(payload["extra"])
        print(f"[{method}] loaded {ckpt}")
        return extractor.to(device), (extra.to(device) if extra is not None else None), payload["summary"]

    epochs = {"vanilla": 100, "gcsc": 100, "rpl": 100, "proser": 50}[method]
    if quick:
        epochs = 2
    train_loader, val_loader = train_val_loaders(data_root, "gcsc" if method == "gcsc" else "vanilla",
                                                 num_workers=num_workers, seed=seed, quick=quick)
    logger = RunLogger(results_dir, f"task4_{method}")
    set_seed(seed)
    if method in ("vanilla", "gcsc"):
        from task4.methods._common import OSRTrainConfig, train_classifier

        cfg = OSRTrainConfig(max_epochs=epochs, seed=seed, device=device)
        extractor, best = train_classifier(extractor, train_loader, val_loader, cfg, logger, run_name=method)
    elif method == "proser":
        from task4.methods.proser import ProserConfig, run as proser_run

        vanilla_ckpt = Path(ckpt_dir) / "vanilla.pth"
        if not vanilla_ckpt.exists():
            raise FileNotFoundError("train Vanilla first: PROSER is initialized from the selected Vanilla checkpoint")
        vstate = torch.load(vanilla_ckpt, map_location="cpu")["net"]
        cfg = ProserConfig(max_epochs=epochs, seed=seed, device=device)
        # proser.run builds its own dummy head; seed first so its init is fixed
        set_seed(seed)
        extractor, extra, best, last_state = proser_run(extractor, train_loader, val_loader, logger, vstate, cfg=cfg)
    else:
        from task4.methods.rpl import RPLConfig, run as rpl_run

        cfg = RPLConfig(max_epochs=epochs, seed=seed, device=device)
        extractor, extra, best = rpl_run(extractor, extra, train_loader, val_loader, logger, cfg=cfg)

    hist = [__import__("json").loads(l) for l in (Path(results_dir) / f"task4_{method}.metrics.jsonl").read_text().splitlines() if l.strip()]
    summary = {"method": method, "best_val_acc": best, "epochs": epochs,
               "best_epoch": int(np.argmax([h["val_acc"] for h in hist])) if hist else None}
    payload = {"net": extractor.net.state_dict(), "summary": summary}
    if extra is not None:
        payload["extra"] = extra.state_dict()
    torch.save(payload, ckpt)
    if method == "proser":
        # Final-epoch weights as a separate checkpoint (supplementary row only).
        last_net, last_dummy = last_state
        torch.save({"net": {k.removeprefix("net."): v for k, v in last_net.items()}, "extra": last_dummy,
                    "summary": {"method": "proser_last", "epochs": epochs, "epoch": epochs - 1,
                                "final_val_acc": hist[-1]["val_acc"] if hist else None}},
                   Path(ckpt_dir) / "proser_last.pth")
    return extractor, extra, summary


def eval_loaders(data_root, num_workers=4, quick=False):
    """Unaugmented CIFAR-10 train (for Mahalanobis statistics), val, test, and
    the near/far CIFAR-100 unknown sets. Near/far labels are CIFAR-100 fine ids."""
    from task4.data.cifar10 import build_cifar10_datasets
    from task4.data.cifar100_unknowns import build_unknown_datasets

    _, evalt = transforms_for("vanilla")
    train_ds, val_ds, test_ds = build_cifar10_datasets(str(data_root), evalt, evalt)
    near_ds, far_ds = build_unknown_datasets(str(data_root), evalt)
    sets = {"train": train_ds, "val": val_ds, "test": test_ds, "near": near_ds, "far": far_ds}
    if quick:
        from torch.utils.data import Subset

        sets = {k: Subset(v, list(range(0, len(v), 20))) for k, v in sets.items()}
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return {k: DataLoader(v, batch_size=512, shuffle=False, **kw) for k, v in sets.items()}


@torch.no_grad()
def extract(extractor, extra, loader, device, method: str) -> dict:
    """Saves penultimate features f(x), the 10 known-class logits z(x) and, for
    PROSER / RPL, the dummy logits / reciprocal-point distances."""
    extractor.eval()
    if extra is not None:
        extra.eval()
    F_, Z_, Y_, E_ = [], [], [], []
    for x, y in loader:
        f, z = extractor(x.to(device, non_blocking=True))
        F_.append(f.float().cpu()); Z_.append(z.float().cpu()); Y_.append(y)
        if method in ("proser", "proser_last"):
            E_.append(extra(f).float().cpu())
        elif method == "rpl":
            E_.append(extra.sq_dist(f).float().cpu())
    out = {"features": torch.cat(F_).numpy(), "logits": torch.cat(Z_).numpy(), "labels": torch.cat(Y_).numpy()}
    if E_:
        out["extra"] = torch.cat(E_).numpy()
    if method == "rpl":  # RPL classifies with the distances, not with the unused fc layer
        out["logits"] = out["extra"]
    return out


def extract_all(method, extractor, extra, loaders, device, cache_dir: Path, overwrite=False) -> dict:
    outs = {}
    for split, loader in loaders.items():
        path = Path(cache_dir) / f"{method}_{split}.npz"
        if path.exists() and not overwrite:
            outs[split] = dict(np.load(path))
            continue
        outs[split] = extract(extractor, extra, loader, device, method)
        np.savez(path, **outs[split])
    return outs
