"""Task 3 Step 4: the common local sharpness proxy.

Delta_sharp = L_val(theta + eps) - L_val(theta),  eps = rho * g / ||g||_2,
g = grad of the cross-entropy on ONE fixed validation batch (32 images from
each source domain, chosen with seed 6304 and shared by all models), with the
model in evaluation mode. rho = 0.05 is the required setting; other radii are
reported only as a robustness check of the ranking.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from shared.pacs import PACSImageDataset


def build_fixed_sharpness_batch(splits: dict, source_domains, transform, n_per_source: int = 32, seed: int = 6304):
    """Returns (images [96,3,224,224], labels [96], chosen indices per domain)."""
    rng = np.random.RandomState(seed)
    xs, ys, chosen = [], [], {}
    for d in source_domains:
        val_idx = np.asarray(splits[d]["val_idx"])
        pick = rng.choice(val_idx, size=min(n_per_source, len(val_idx)), replace=False)
        chosen[d] = pick.tolist()
        ds = PACSImageDataset(splits[d]["paths"], splits[d]["labels"], pick, transform)
        for i in range(len(ds)):
            x, y = ds[i]
            xs.append(x); ys.append(y)
    return torch.stack(xs), torch.tensor(ys), chosen


def compute_sharpness(model, images, labels, rho: float = 0.05, device="cpu") -> dict:
    """Returns {"loss": L(theta), "perturbed_loss": L(theta+eps), "delta": ...}.
    Parameters are restored exactly afterwards."""
    model = model.to(device).eval()
    x, y = images.to(device), labels.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    loss = F.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, params)
    norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
    backup = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, g in zip(params, grads):
            p.add_(g, alpha=float(rho / (norm + 1e-12)))
        perturbed = F.cross_entropy(model(x), y)
        for p, b in zip(params, backup):
            p.copy_(b)
    return {"loss": float(loss.detach()), "perturbed_loss": float(perturbed), "delta": float(perturbed - loss.detach()),
            "grad_norm": float(norm)}
