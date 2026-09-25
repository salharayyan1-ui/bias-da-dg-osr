"""Task 4 Step 4: PROSER (Zhou et al., CVPR 2021).

Loss construction cross-checked against two sources (not just recalled from
memory) before writing this:
  1. A survey's restatement of the paper's Eq. 5, giving the classifier-
     placeholder loss as L1 = CE(f_hat(x), y) + beta * CE(f_hat(x)/y, K+1),
     where f_hat(x)/y means the true-class logit is masked out and the target
     becomes the (collapsed) dummy slot.
  2. The official reference implementation (github.com/zhoudw-zdw/CVPR21-Proser,
     proser_unknown_detection.py::traindummy), which realizes this as two
     terms (their loss2 + loss3) plus a third term (their loss1) for the
     manifold-mixup data placeholder.

Two deliberate departures from that reference script, documented here rather
than silently changed:
  - The official script demonstrates C=1 dummy classifier and targets that
    single dummy logit directly. The assignment requires C=5 dummy
    classifiers, so both loss terms below first collapse the 5 dummy logits
    to their MAX and treat that as one appended "slot" (matching how the
    same script evaluates multi-dummy performance at test time via
    `maxdummylogit`) -- this is the natural generalization, not a change to
    the mechanism itself.
  - The assignment's own text assigns the FIRST half of each batch to
    classifier-placeholder training and the SECOND half to data placeholders
    ("Split each mini-batch into two equal parts: use the first half for
    classifier-placeholder training and the second half for manifold-mixup
    data placeholders"), which is swapped relative to the reference script's
    variable names (their `prehalf` is mixup, `laterhalf` is classifier
    placeholder). We follow the assignment's stated halves.
  - (Added after runs v1/v2.) The mixed hidden states pass through
    layer3/layer4 without updating those BatchNorm layers' running
    statistics, which are therefore estimated from real images only (see
    _bn_running_stats_frozen).

Re-derive/verify this against the paper yourself before trusting it -- CVPR
papers' exact equations matter here and this is a good-faith reconstruction,
not a transcription of paper-internal notation we couldn't fetch in full.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logging import RunLogger
from task4.methods.manifold_mixup import cross_class_pair_indices, sample_beta_lambda, mix_features


@dataclass
class ProserConfig:
    max_epochs: int = 50
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 128
    seed: int = 6304
    device: str = "cpu"
    n_dummy: int = 5
    beta_classifier_placeholder: float = 1.0
    gamma_data_placeholder: float = 0.1


class DummyHead(nn.Module):
    """5 randomly-initialized dummy classifiers, operating on the SAME 512-d
    penultimate feature as the real 10-way head."""

    def __init__(self, feature_dim: int, n_dummy: int = 5):
        super().__init__()
        self.fc = nn.Linear(feature_dim, n_dummy)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)


def _classifier_placeholder_losses(extractor, dummy_head, images: torch.Tensor, labels: torch.Tensor, n_known: int):
    """First half of the batch, REAL (unmixed) images. Returns
    (standard_ce, excluded_true_class_ce)."""
    feat, real_logits = extractor(images)          # [B, n_known]
    dummy_logits = dummy_head(feat)                 # [B, n_dummy]
    max_dummy = dummy_logits.max(dim=1, keepdim=True).values  # collapse C dummies -> 1 slot

    # Term 1: ordinary classification, true class should be the overall max
    # among [real_logits, max_dummy] (keeps the dummy from "stealing" the
    # correct prediction on ordinary data).
    combined_for_ce1 = torch.cat([real_logits, max_dummy], dim=1)  # [B, n_known+1]
    standard_ce = F.cross_entropy(combined_for_ce1, labels)

    # Term 2: with the TRUE class masked out, the dummy slot should now be
    # the max among the remaining classes -- this is what actually pushes
    # the dummy to sit at "2nd place" behind the true class.
    masked = real_logits.clone()
    masked.scatter_(1, labels.view(-1, 1), float("-1e9"))
    combined_for_ce2 = torch.cat([masked, max_dummy], dim=1)       # [B, n_known+1]
    dummy_target = torch.full((images.shape[0],), n_known, dtype=torch.long, device=images.device)
    excluded_true_class_ce = F.cross_entropy(combined_for_ce2, dummy_target)

    return standard_ce, excluded_true_class_ce


@contextlib.contextmanager
def _bn_running_stats_frozen(*modules):
    """BatchNorm layers inside `modules` keep normalizing with the batch's own
    statistics (train mode, as in the reference implementation) but do not
    update their running statistics for the duration (momentum 0).

    Used for the manifold-mixup pass: mixed layer2 outputs are synthetic and
    have lower variance than real ones (lambda*h_i + (1-lambda)*h_j), so
    letting them update layer3/layer4 running statistics makes the statistics
    used at test time a blend of real and mixed batches (run v1: CIFAR-10
    validation accuracy fell from 0.948 to 0.886 within four epochs, and the
    checkpoint rule kept epoch 0). Run v2 instead put these layers in eval
    mode, which normalizes the mixed batch with the running statistics and
    changes the training dynamics: validation accuracy swung between 0.72 and
    0.94, and the losses rose in the last epochs while the learning rate was
    ~1e-6. Momentum 0 keeps v1's training computation exactly and only stops
    the contamination. v1/v2 logs: task4/results/diagnostics/."""
    bns = [m for mod in modules for m in mod.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.momentum = 0.0  # running = (1 - 0) * running + 0 * batch
    try:
        yield
    finally:
        for m, mom in zip(bns, momenta):
            m.momentum = mom


def _data_placeholder_loss(extractor, dummy_head, images: torch.Tensor, labels: torch.Tensor, n_known: int, seed: int):
    """Second half of the batch: manifold mixup after layer2, mixed
    representations trained toward the dummy slot."""
    labels_np = labels.detach().cpu().numpy()
    perm = cross_class_pair_indices(labels_np, seed=seed)
    perm_t = torch.as_tensor(perm, device=images.device, dtype=torch.long)

    valid = torch.as_tensor(labels_np != labels_np[perm], device=images.device)  # drop any residual same-class pairs
    if valid.sum() == 0:
        return images.new_zeros(())

    h = extractor.forward_up_to_layer2(images)
    lam = torch.as_tensor(sample_beta_lambda(images.shape[0], alpha=2.0, beta=2.0, seed=seed), device=images.device)
    h_mixed = mix_features(h, h[perm_t], lam)

    h_mixed, lam, valid_idx = h_mixed[valid], lam[valid], valid.nonzero(as_tuple=True)[0]
    with _bn_running_stats_frozen(extractor.net.layer3, extractor.net.layer4):
        feat_mixed, real_logits_mixed = extractor.forward_from_layer3(h_mixed)
    dummy_logits_mixed = dummy_head(feat_mixed)
    max_dummy_mixed = dummy_logits_mixed.max(dim=1, keepdim=True).values

    combined = torch.cat([real_logits_mixed, max_dummy_mixed], dim=1)  # [B', n_known+1]
    dummy_target = torch.full((h_mixed.shape[0],), n_known, dtype=torch.long, device=images.device)
    return F.cross_entropy(combined, dummy_target)


def run(extractor, train_loader, val_loader, logger: RunLogger, vanilla_checkpoint_state: dict, cfg: ProserConfig | None = None):
    """`extractor` should be a fresh PenultimateExtractor already loaded with
    the selected Vanilla checkpoint's weights (`vanilla_checkpoint_state`),
    per "Initialize PROSER from the selected Vanilla checkpoint."""
    cfg = cfg or ProserConfig()
    extractor.net.load_state_dict(vanilla_checkpoint_state)
    extractor.to(cfg.device)
    dummy_head = DummyHead(extractor.feature_dim, cfg.n_dummy).to(cfg.device)

    params = list(extractor.parameters()) + list(dummy_head.parameters())
    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    n_known = extractor.net.fc.out_features
    best_val_acc, best_state = -1.0, (copy.deepcopy(extractor.state_dict()), copy.deepcopy(dummy_head.state_dict()))

    for epoch in range(cfg.max_epochs):
        extractor.train()
        dummy_head.train()
        epoch_l1, epoch_l2, epoch_l3, n_batches = 0.0, 0.0, 0.0, 0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            half = images.shape[0] // 2
            first_images, first_labels = images[:half], labels[:half]
            second_images, second_labels = images[half:], labels[half:]

            std_ce, excl_ce = _classifier_placeholder_losses(extractor, dummy_head, first_images, first_labels, n_known)
            data_ph_loss = _data_placeholder_loss(extractor, dummy_head, second_images, second_labels, n_known,
                                                    seed=cfg.seed + epoch * 100000 + step)

            loss = std_ce + cfg.beta_classifier_placeholder * excl_ce + cfg.gamma_data_placeholder * data_ph_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_l1 += float(std_ce.detach())
            epoch_l2 += float(excl_ce.detach())
            epoch_l3 += float(data_ph_loss.detach())
            n_batches += 1
        scheduler.step()

        val_acc = _eval_known_accuracy(extractor, val_loader, cfg.device)
        logger.log(epoch, run="proser",
                    standard_ce=epoch_l1 / max(n_batches, 1),
                    classifier_placeholder_ce=epoch_l2 / max(n_batches, 1),
                    data_placeholder_ce=epoch_l3 / max(n_batches, 1),
                    val_acc=val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = (copy.deepcopy(extractor.state_dict()), copy.deepcopy(dummy_head.state_dict()))

    # Final-epoch weights, kept for a supplementary row: the checkpoint rule
    # (best CIFAR-10 validation accuracy) can prefer the least-modified model,
    # because placeholder training trades some closed-set accuracy for rejection.
    last_state = (copy.deepcopy(extractor.state_dict()), copy.deepcopy(dummy_head.state_dict()))
    extractor.load_state_dict(best_state[0])
    dummy_head.load_state_dict(best_state[1])
    logger.save_summary(run="proser", best_val_acc=best_val_acc, final_val_acc=val_acc)
    return extractor, dummy_head, best_val_acc, last_state


@torch.no_grad()
def _eval_known_accuracy(extractor, loader, device: str) -> float:
    """CSA computed using ONLY the known-class logits, per spec."""
    extractor.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        _, logits = extractor(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def proser_mls_score(extractor, dummy_head, images: torch.Tensor) -> torch.Tensor:
    """PROSER evaluated with MLS on the known-class logits only (for direct
    comparison with Vanilla/GCSC), per "report PROSER with MLS on the ten
    known-class logits.\""""
    with torch.no_grad():
        _, logits = extractor(images)
        return -logits.max(dim=1).values


def placeholder_scores_from_logits(logits: np.ndarray, dummy_logits: np.ndarray) -> dict:
    """Placeholder-based unknownness scores, following the reference code
    (combine the known logits with the STRONGEST dummy logit):

    * "proser_placeholder": u = max_k dummy_k - max_j z_j. The paper calibrates
      rejection by adding a bias b to the dummy logit, chosen on validation
      data so that 95% of known examples are still predicted as known; an input
      is rejected when max dummy + b > max_j z_j, i.e. when u > -b. Thresholding
      u at its 95th percentile on CIFAR-10 validation is exactly that rule.
    * "proser_dummy_prob": softmax([z, max dummy])[-1], the probability the
      reference implementation uses as its detection score for AUROC.
    """
    md = dummy_logits.max(axis=1, keepdims=True)
    comb = np.concatenate([logits, md], axis=1)
    comb = comb - comb.max(axis=1, keepdims=True)
    e = np.exp(comb)
    return {"proser_placeholder": (md[:, 0] - logits.max(axis=1)),
            "proser_dummy_prob": e[:, -1] / e.sum(axis=1)}
