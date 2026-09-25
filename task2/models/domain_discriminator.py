"""Gradient reversal + domain discriminator shared by DANN (Step 3) and CDAN
(Step 4). Both use the same discriminator (256-unit hidden layer, ReLU,
dropout 0.5, two-way output), GRL schedule and unit domain-loss weight; only
the discriminator input differs (f for DANN, vec(f (x) p) for CDAN)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReversalFunction(torch.autograd.Function):
    """Identity forward; multiplies the incoming gradient by -alpha backward
    (Ganin et al., 2016)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _GradientReversalFunction.apply(x, alpha)


def discriminator_input(features: torch.Tensor) -> torch.Tensor:
    """sqrt(D) * f / ||f||: the feature's direction at a fixed norm of sqrt(D).

    Fixing the norm stops the reversed gradient from inflating the feature
    scale (with frozen BatchNorm nothing else bounds it; the un-normalized run
    diverged, see task2/results/diagnostics/). The sqrt(D) factor keeps the
    per-coordinate scale near 1. Run v1 used the unit norm: the discriminator
    then separated the domains in sign but its logits stayed near zero (much
    higher held-out cross-entropy; task2/results/diagnostics/
    discriminator_input_scale.json), and the normalization divided the reversed
    gradient reaching the features by ||f|| (about 30-55)."""
    return F.normalize(features, dim=1) * math.sqrt(features.shape[1])


def grl_alpha_schedule(p: float, max_alpha: float = 1.0) -> float:
    """alpha(p) = max_alpha * (2 / (1 + exp(-10 p)) - 1), p in [0, 1] = fraction
    of the full training budget (max_epochs x steps_per_epoch). max_alpha = 1
    is the standard schedule; other values only rescale it (DANN design study)."""
    return max_alpha * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def cdan_conditioning(features: torch.Tensor, class_probs: torch.Tensor) -> torch.Tensor:
    """g(x) = vec(f (x) p): per-example outer product of the 512-d feature and
    the 7-way softmax, flattened to 3584 dimensions. Neither f nor p is detached
    and no entropy weighting is used (as required)."""
    return torch.bmm(features.unsqueeze(2), class_probs.unsqueeze(1)).flatten(1)
