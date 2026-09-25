"""Shared Task 4 training loop for Vanilla (Step 1) and GCSC (Step 3), which
use IDENTICAL recipes except for one added augmentation (RandAugment) in
GCSC. SGD lr=0.1, momentum 0.9, wd 5e-4, cosine decay, batch 128, 100 epochs,
seed 6304; checkpoint = highest CIFAR-10 validation accuracy."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logging import RunLogger


@dataclass
class OSRTrainConfig:
    max_epochs: int = 100
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 128
    seed: int = 6304
    device: str = "cpu"


def train_classifier(
    model: nn.Module,  # PenultimateExtractor
    train_loader,
    val_loader,
    cfg: OSRTrainConfig,
    logger: RunLogger,
    run_name: str = "vanilla",
):
    model.to(cfg.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            _, logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        scheduler.step()

        val_acc = _eval_accuracy(model, val_loader, cfg.device)
        logger.log(epoch, run=run_name, train_loss=epoch_loss / max(n_batches, 1), val_acc=val_acc,
                    lr=optimizer.param_groups[0]["lr"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    logger.save_summary(run=run_name, best_val_acc=best_val_acc)
    return model, best_val_acc


@torch.no_grad()
def _eval_accuracy(model, loader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        _, logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)
