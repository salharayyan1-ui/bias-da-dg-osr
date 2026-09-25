"""Task 4 Step 5 (OPTIONAL): Reciprocal Point Learning (Chen et al., ECCV 2020).

Design cross-checked against the official ARPL/RPL reference implementation
(iCGY96/ARPL, RPLoss.py) rather than reconstructed purely from the paper text
(which we couldn't fetch in full here):

- `Dist`: one learnable reciprocal point per class (a `nn.Parameter`,
  `num_classes x feature_dim`), squared-Euclidean distance to a batch of
  features divided by the feature dimension (the reference `l2` metric).
- Classification: the reciprocal-point distances are used DIRECTLY as logits
  (scaled by a temperature), and cross-entropy is applied with the TRUE
  label. Since a reciprocal point represents "the extra-class (negative)
  space" for its class, cross-entropy here pushes a class-y example's
  distance to point P_y to be the LARGEST among all classes' distances --
  i.e. the example should sit far from what "not-y" looks like, which is a
  roundabout (reciprocal) way of confirming it IS y.
- Open-space regularization: for each example, the (per-dimension-mean)
  squared distance to its OWN true class's reciprocal point is pulled toward
  a single learnable radius R via MSE, bounding how far a same-class feature
  is allowed to drift -- this is literally what keeps the feature space
  bounded, per the assignment's description.

At test time we use the same "distance-as-logit" view: MLS-style, the
unknownness score is the negative of the largest per-class distance (an
example far from EVERY class's reciprocal point, in the max sense, looks
like a confident known example under this scheme; a genuinely novel input
tends not to sit far from any one point). Cross-check this design against
Chen et al. (2020) directly before trusting it for anything beyond a
starting point -- this is a good-faith, source-grounded reconstruction of a
paper we could not read in full here, not a verbatim transcription.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logging import RunLogger


@dataclass
class RPLConfig:
    max_epochs: int = 100
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 6304
    device: str = "cpu"
    temperature: float = 1.0
    open_space_weight: float = 0.1  # trade-off for the radius/open-space term (tune; not spec'd numerically by the PDF)


class ReciprocalPoints(nn.Module):
    def __init__(self, n_classes: int, feature_dim: int):
        super().__init__()
        self.centers = nn.Parameter(0.1 * torch.randn(n_classes, feature_dim))
        self.radius = nn.Parameter(torch.zeros(1))  # single learnable bounding radius, per the reference

    def sq_dist(self, features: torch.Tensor) -> torch.Tensor:
        """Mean (over feature dimensions) squared Euclidean distance from every
        feature to every reciprocal point, as in the ARPL/RPL reference `Dist`
        (metric='l2' divides by the feature dimension). [N, D] -> [N, C]."""
        f2 = (features ** 2).sum(dim=1, keepdim=True)
        c2 = (self.centers ** 2).sum(dim=1, keepdim=True).t()
        cross = features @ self.centers.t()
        return (f2 + c2 - 2.0 * cross).clamp(min=0.0) / features.shape[1]

    def open_space_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        own_center = self.centers[labels]                              # [N, D]
        per_example_dist = (features - own_center).pow(2).mean(dim=1)  # [N]
        target_radius = self.radius.expand_as(per_example_dist)
        return F.mse_loss(per_example_dist, target_radius)


def run(extractor, rp_head: ReciprocalPoints, train_loader, val_loader, logger: RunLogger, cfg: RPLConfig | None = None):
    cfg = cfg or RPLConfig()
    extractor.to(cfg.device)
    rp_head.to(cfg.device)
    params = list(extractor.parameters()) + list(rp_head.parameters())
    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    best_val_acc, best_state = -1.0, (copy.deepcopy(extractor.state_dict()), copy.deepcopy(rp_head.state_dict()))

    for epoch in range(cfg.max_epochs):
        extractor.train()
        rp_head.train()
        epoch_cls, epoch_reg, n_batches = 0.0, 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            feat, _ = extractor(images)  # CIFAR ResNet-18's 512-d penultimate feature (fc's logits unused here)
            dist = rp_head.sq_dist(feat)
            cls_loss = F.cross_entropy(dist / cfg.temperature, labels)
            reg_loss = rp_head.open_space_loss(feat, labels)
            loss = cls_loss + cfg.open_space_weight * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_cls += float(cls_loss.detach())
            epoch_reg += float(reg_loss.detach())
            n_batches += 1
        scheduler.step()

        val_acc = _eval_accuracy(extractor, rp_head, val_loader, cfg.device, cfg.temperature)
        logger.log(epoch, run="rpl", cls_loss=epoch_cls / max(n_batches, 1),
                    open_space_loss=epoch_reg / max(n_batches, 1), val_acc=val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = (copy.deepcopy(extractor.state_dict()), copy.deepcopy(rp_head.state_dict()))

    extractor.load_state_dict(best_state[0])
    rp_head.load_state_dict(best_state[1])
    logger.save_summary(run="rpl", best_val_acc=best_val_acc)
    return extractor, rp_head, best_val_acc


@torch.no_grad()
def _eval_accuracy(extractor, rp_head, loader, device: str, temperature: float) -> float:
    extractor.eval()
    rp_head.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        feat, _ = extractor(images)
        dist = rp_head.sq_dist(feat) / temperature
        correct += (dist.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def rpl_scores_from_distances(dist: "np.ndarray") -> dict:
    """Unknownness scores from the per-class reciprocal-point distances
    (larger distance = stronger evidence for that class, so the distances play
    the role of logits):

    * "rpl_softmax_max": 1 - max_k softmax(d)_k. This is the score the
      ARPL/RPL reference implementation uses at test time (`RPLoss.forward`
      returns `softmax(dist)` and the test code takes its maximum as the
      known-class score).
    * "rpl_max_distance": -max_k d_k, the MLS analogue of the same quantity.

    Both are reported; neither was selected by looking at unknown data.
    """
    import numpy as np

    z = dist - dist.max(axis=1, keepdims=True)
    e = np.exp(z)
    return {"rpl_softmax_max": 1.0 - (e / e.sum(axis=1, keepdims=True)).max(axis=1),
            "rpl_max_distance": -dist.max(axis=1)}


def rpl_score(extractor, rp_head: ReciprocalPoints, images: torch.Tensor) -> torch.Tensor:
    """Unknownness score: -max_k dist_k(x), mirroring MLS but with
    reciprocal-point distance standing in for the logit."""
    with torch.no_grad():
        feat, _ = extractor(images)
        dist = rp_head.sq_dist(feat)
        return -dist.max(dim=1).values
