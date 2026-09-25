"""The single PACS training / evaluation pipeline used by every method in
Task 2 (Source-only, DAN, DANN, CDAN) and Task 3 (DAN-DG, SAM).

Everything that the spec requires to be identical across methods lives here:
initialization (seed 6304 before building the network), preprocessing and
augmentation, domain-balanced sampling (8 per source domain [+ 24 target]),
AdamW(lr 1e-4, wd 1e-4), <= 30 epochs, early stopping after 5 epochs without
an improvement of the mean source-validation macro-F1, frozen BatchNorm
running statistics, and checkpoint selection. A method only supplies an
extra loss term (and optionally extra trainable modules or a SAM radius).
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.logging import RunLogger
from common.metrics import macro_f1, top1_accuracy
from common.seed import seeded_generator, set_seed
from shared.pacs import PACSImageDataset
from shared.pacs_protocol import DomainBalancedBatchIterator, _cycle

SOURCE_DOMAINS = ("photo", "art_painting", "cartoon")
TARGET_DOMAIN = "sketch"


@dataclass
class PACSTrainConfig:
    max_epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 6304
    per_source_batch: int = 8
    target_batch: int = 24
    num_workers: int = 4
    eval_batch: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps_per_epoch: Optional[int] = None  # only for quick dry runs


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def train_transform():
    from torchvision import transforms

    return transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop(224),
                               transforms.RandomHorizontalFlip(), transforms.ToTensor()])


def eval_transform():
    from torchvision import transforms

    return transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor()])


def _loader(ds, batch_size, shuffle, cfg: PACSTrainConfig, seed_offset=0, drop_last=False):
    kw = dict(num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    if cfg.num_workers > 0:
        kw["persistent_workers"] = True
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      generator=seeded_generator(cfg.seed + seed_offset) if shuffle else None, **kw)


def source_val_loaders(splits: dict, cfg: PACSTrainConfig) -> dict:
    return {d: _loader(PACSImageDataset(splits[d]["paths"], splits[d]["labels"], splits[d]["val_idx"], eval_transform()),
                       cfg.eval_batch, False, cfg) for d in SOURCE_DOMAINS}


def target_eval_loader(splits: dict, cfg: PACSTrainConfig):
    """ALL Sketch images, centre crop. Only call this from final-evaluation code
    (Task 2 Step 5, Task 3 evaluate_sketch)."""
    d = splits[TARGET_DOMAIN]
    return _loader(PACSImageDataset(d["paths"], d["labels"], None, eval_transform()), cfg.eval_batch, False, cfg)


def training_batches(splits: dict, cfg: PACSTrainConfig, use_target: bool) -> DomainBalancedBatchIterator:
    """8 images from each source domain per step (+ 24 unlabeled Sketch images
    for UDA). One epoch = one pass over the largest source training split;
    smaller domains (and the target) are cycled. Each domain loader has its
    own seeded generator, so the sampling order is identical for every method."""
    loaders, max_len = {}, 0
    for k, d in enumerate(SOURCE_DOMAINS):
        ds = PACSImageDataset(splits[d]["paths"], splits[d]["labels"], splits[d]["train_idx"], train_transform())
        loaders[d] = _loader(ds, cfg.per_source_batch, True, cfg, seed_offset=k, drop_last=True)
        max_len = max(max_len, len(loaders[d]))
    if use_target:
        t = splits[TARGET_DOMAIN]
        # Labels are carried by the dataset object but never read by the training loop.
        ds = PACSImageDataset(t["paths"], t["labels"], None, train_transform())
        loaders["target"] = _loader(ds, cfg.target_batch, True, cfg, seed_offset=99, drop_last=True)
    steps = max_len if cfg.max_steps_per_epoch is None else min(max_len, cfg.max_steps_per_epoch)
    return DomainBalancedBatchIterator(loaders, steps)


# --------------------------------------------------------------------------- #
# Methods plug into the loop through this interface
# --------------------------------------------------------------------------- #

@dataclass
class MethodSpec:
    name: str
    uses_target: bool = False
    # extra_loss(model, feats_by_domain, progress) -> (loss tensor, stats dict)
    extra_loss: Optional[Callable] = None
    extra_modules: list = field(default_factory=list)
    sam_rho: Optional[float] = None
    hparams: dict = field(default_factory=dict)


def _sam_perturb(params, rho):
    grads = [p.grad for p in params if p.grad is not None]
    norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
    scale = rho / (norm + 1e-12)
    eps = []
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                eps.append(None)
                continue
            e = p.grad * scale
            p.add_(e)
            eps.append(e)
    return eps


def _sam_restore(params, eps):
    with torch.no_grad():
        for p, e in zip(params, eps):
            if e is not None:
                p.sub_(e)


def train_method(model: nn.Module, method: MethodSpec, splits: dict, cfg: PACSTrainConfig,
                 logger: RunLogger, run_name: str):
    """Trains `model` (a PACSNet) with `method`; returns (model loaded with the
    best-checkpoint weights, summary dict). Only source validation data is
    used for model selection."""
    device = cfg.device
    model.to(device)
    for m in method.extra_modules:
        m.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    extra_params = [p for m in method.extra_modules for p in m.parameters()]
    optimizer = torch.optim.AdamW(params + extra_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    batches = training_batches(splits, cfg, method.uses_target)
    val_loaders = source_val_loaders(splits, cfg)
    total_steps = cfg.max_epochs * len(batches)

    best = {"score": -1.0, "epoch": -1, "state": copy.deepcopy(model.state_dict()),
            "extra": [copy.deepcopy(m.state_dict()) for m in method.extra_modules]}
    since, step, history, t0 = 0, 0, [], time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        for m in method.extra_modules:
            m.train()
        sums, n_steps, correct, seen = {}, 0, 0, 0
        for batch in batches:
            progress = step / max(total_steps, 1)
            src_x = torch.cat([batch[d][0] for d in SOURCE_DOMAINS]).to(device, non_blocking=True)
            src_y = torch.cat([batch[d][1] for d in SOURCE_DOMAINS]).to(device, non_blocking=True)
            sizes = [batch[d][0].shape[0] for d in SOURCE_DOMAINS]
            x = src_x
            if method.uses_target:
                x = torch.cat([src_x, batch["target"][0].to(device, non_blocking=True)])

            def compute_loss():
                feats = model.features(x)  # BN uses frozen statistics, so joint forward == separate forwards
                src_feat = feats[:src_x.shape[0]]
                logits = model.head(src_feat)
                cls_loss = F.cross_entropy(logits, src_y)
                stats = {}
                total = cls_loss
                if method.extra_loss is not None:
                    by_domain = dict(zip(SOURCE_DOMAINS, torch.split(src_feat, sizes)))
                    if method.uses_target:
                        by_domain["target"] = feats[src_x.shape[0]:]
                    extra, stats = method.extra_loss(model, by_domain, progress)
                    total = total + extra
                    stats = {"extra_loss": float(extra.detach()), **stats}
                return total, cls_loss, logits, stats

            optimizer.zero_grad(set_to_none=True)
            total, cls_loss, logits, stats = compute_loss()
            total.backward()
            if method.sam_rho is not None:
                # Non-adaptive SAM: ascend to w + rho * g/||g||, take the gradient
                # there, restore w, then let AdamW step with the perturbed-point gradient.
                eps = _sam_perturb(params, method.sam_rho)
                optimizer.zero_grad(set_to_none=True)
                total2, _, _, _ = compute_loss()
                total2.backward()
                _sam_restore(params, eps)
                stats["sam_perturbed_loss"] = float(total2.detach())
            optimizer.step()

            step += 1
            n_steps += 1
            correct += int((logits.argmax(1) == src_y).sum())
            seen += src_y.numel()
            for k, v in {"cls_loss": float(cls_loss.detach()), "total_loss": float(total.detach()), **stats}.items():
                sums[k] = sums.get(k, 0.0) + v

        val = evaluate_source_val(model, val_loaders, device)
        mean_f1 = float(np.mean([val[d]["macro_f1"] for d in SOURCE_DOMAINS]))
        row = {"epoch": epoch, "method": method.name, "train_source_acc": correct / max(seen, 1),
               **{k: v / max(n_steps, 1) for k, v in sums.items()},
               "mean_val_macro_f1": mean_f1,
               "mean_val_accuracy": float(np.mean([val[d]["accuracy"] for d in SOURCE_DOMAINS])),
               "worst_val_macro_f1": float(np.min([val[d]["macro_f1"] for d in SOURCE_DOMAINS])),
               **{f"val_acc_{d}": val[d]["accuracy"] for d in SOURCE_DOMAINS},
               **{f"val_f1_{d}": val[d]["macro_f1"] for d in SOURCE_DOMAINS},
               "progress": step / max(total_steps, 1), "minutes": (time.time() - t0) / 60}
        logger.log(epoch, **row)
        history.append(row)
        print(f"[{run_name}] epoch {epoch:2d} cls {row['cls_loss']:.3f} "
              + (f"extra {row.get('extra_loss', 0):.3f} " if method.extra_loss else "")
              + f"src-train-acc {row['train_source_acc']:.3f} mean-val-F1 {mean_f1:.4f}")

        if mean_f1 > best["score"]:
            best = {"score": mean_f1, "epoch": epoch, "state": copy.deepcopy(model.state_dict()),
                    "extra": [copy.deepcopy(m.state_dict()) for m in method.extra_modules]}
            since = 0
        else:
            since += 1
            if since >= cfg.patience:
                break

    model.load_state_dict(best["state"])
    for m, s in zip(method.extra_modules, best["extra"]):
        m.load_state_dict(s)
    summary = {"run": run_name, "method": method.name, "hparams": method.hparams, "config": asdict(cfg),
               "best_epoch": best["epoch"], "best_mean_val_macro_f1": best["score"],
               "epochs_run": len(history), "steps_per_epoch": len(batches),
               "stopped_early": len(history) < cfg.max_epochs, "minutes": (time.time() - t0) / 60}
    logger.save_summary(**summary)
    return model, summary


def run_or_load(method_factory: Callable[[], MethodSpec], run_name: str, splits: dict, cfg: PACSTrainConfig,
                results_dir: Path, ckpt_dir: Path, retrain: bool = False, init_state: dict | None = None):
    """Seeds, builds a fresh PACSNet (identical initialization for every run),
    trains (or loads an existing checkpoint) and returns (model, summary)."""
    from task2.models.backbone import PACSNet

    ckpt = Path(ckpt_dir) / f"{run_name}.pth"
    set_seed(cfg.seed)
    model = PACSNet(n_classes=7)
    method = method_factory()  # created after the network so discriminators get a fixed init too
    if ckpt.exists() and not retrain:
        payload = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(payload["model"])
        print(f"[{run_name}] loaded existing checkpoint {ckpt}")
        return model.to(cfg.device), payload["summary"]
    if init_state is not None:
        model.load_state_dict(init_state)
    logger = RunLogger(results_dir, run_name)
    set_seed(cfg.seed)  # same data-order / augmentation RNG state for every method
    model, summary = train_method(model, method, splits, cfg, logger, run_name)
    torch.save({"model": model.state_dict(), "summary": summary}, ckpt)
    return model, summary


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict(model: nn.Module, loader, device) -> dict:
    model.eval()
    feats, logits, labels = [], [], []
    for x, y in loader:
        f = model.features(x.to(device, non_blocking=True))
        feats.append(f.float().cpu())
        logits.append(model.head(f).float().cpu())
        labels.append(y)
    logits = torch.cat(logits)
    return {"features": torch.cat(feats).numpy(), "logits": logits.numpy(),
            "probs": F.softmax(logits, 1).numpy(), "preds": logits.argmax(1).numpy(),
            "labels": torch.cat(labels).numpy()}


def evaluate_source_val(model, val_loaders: dict, device) -> dict:
    out = {}
    for d, loader in val_loaders.items():
        p = predict(model, loader, device)
        out[d] = {"accuracy": top1_accuracy(p["labels"], p["preds"]), "macro_f1": macro_f1(p["labels"], p["preds"])}
    return out


def load_history(results_dir: Path, run_name: str) -> list[dict]:
    path = Path(results_dir) / f"{run_name}.metrics.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
