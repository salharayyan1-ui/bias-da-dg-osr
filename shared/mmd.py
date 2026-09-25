"""Multi-kernel RBF MMD^2, shared verbatim by Task 2's DAN (Step 2, source vs.
target) and Task 3's DAN-DG (Step 2, every pair of *observed* sources) -- the
PDF explicitly requires "the same MMD implementation and kernel construction
as Task 2 so that the role of target access can later be examined without
changing the discrepancy measure."

The pure-numpy version of exactly this formula was checked separately for
sanity (near-zero MMD^2 for two samples from the same distribution, clearly
positive and larger for a shifted distribution) before being transcribed here;
see the PR/commit history or re-run that check yourself before trusting this
in a real training loop -- this file itself requires torch to run.
"""
from __future__ import annotations

import torch


def pairwise_sq_dists(z: torch.Tensor) -> torch.Tensor:
    """z: [N, D]. Returns the [N, N] matrix of squared L2 distances, via
    ||zi-zj||^2 = ||zi||^2 + ||zj||^2 - 2 zi.zj (numerically clamped >= 0)."""
    sq = (z ** 2).sum(dim=1)
    d = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (z @ z.t())
    return d.clamp(min=0.0)


def median_heuristic_bandwidth(z: torch.Tensor) -> torch.Tensor:
    """Median of the *off-diagonal* pairwise squared distances in the current
    combined batch (source+target for DAN; the pair of source domains for
    DAN-DG), detached so the bandwidth itself doesn't receive gradient (only
    the kernel *values* should backprop, per standard MMD practice)."""
    d = pairwise_sq_dists(z)
    n = d.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    return d[mask].median().detach()


def multi_kernel_rbf(z: torch.Tensor, bandwidths: list[torch.Tensor]) -> torch.Tensor:
    """Sum of RBF kernels exp(-D / bandwidth_i) over the given bandwidths."""
    d = pairwise_sq_dists(z)
    k = torch.zeros_like(d)
    for b in bandwidths:
        k = k + torch.exp(-d / b.clamp(min=1e-8))
    return k


def mmd2_biased(x: torch.Tensor, y: torch.Tensor, kernel_mults=(0.5, 1.0, 2.0)) -> torch.Tensor:
    """L_H^2 = ||E_s[phi(F(xs))] - E_t[phi(F(xt))]||^2_H via the kernel trick:
    E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')], using the biased (V-statistic)
    empirical estimator over the current batch. Bandwidths = kernel_mults *
    median pairwise squared distance of the *combined* [x; y] batch, per spec.
    """
    n, m = x.shape[0], y.shape[0]
    z = torch.cat([x, y], dim=0)
    med = median_heuristic_bandwidth(z)
    bandwidths = [med * mult for mult in kernel_mults]
    k = multi_kernel_rbf(z, bandwidths)
    kxx = k[:n, :n]
    kyy = k[n:, n:]
    kxy = k[:n, n:]
    return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()


def mmd2_unbiased(x: torch.Tensor, y: torch.Tensor, kernel_mults=(0.5, 1.0, 2.0)) -> torch.Tensor:
    """Unbiased (U-statistic) estimate of MMD^2 (Gretton et al., 2012, Eq. 3):
    within-sample means exclude the diagonal k(x_i, x_i). This is the estimator
    used for training in DAN and DAN-DG.

    Why not the biased V-statistic above: with the small per-domain batches of
    this protocol (8 images per source domain in DAN-DG) the diagonal terms add
    a constant ~ 3/n per sample (3 kernels, each = 1 on the diagonal), so the
    biased value is dominated by estimator bias, never approaches 0 even for
    identical distributions, and - combined with the median bandwidth that
    tracks the feature scale - its gradient drives the features towards a
    collapsed, all-zero ReLU solution (observed: feature norm 26 -> 3, 87%
    exact zeros, classification loss stuck at ln 7 for DAN-DG with lambda=1).
    The unbiased estimate has expectation MMD^2 (0 for matched distributions)."""
    n, m = x.shape[0], y.shape[0]
    z = torch.cat([x, y], dim=0)
    med = median_heuristic_bandwidth(z)
    k = multi_kernel_rbf(z, [med * mult for mult in kernel_mults])
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    sxx = (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
    syy = (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
    return sxx + syy - 2.0 * kxy.mean()


def pairwise_mmd2_over_domains(features_by_domain: dict[str, torch.Tensor], kernel_mults=(0.5, 1.0, 2.0)) -> torch.Tensor:
    """Task 3 DAN-DG: average MMD^2 over every unordered pair of (observed)
    domains, each pair using its *own* median-heuristic bandwidth computed
    from that pair's combined batch, per "bandwidths ... for each domain pair
    in the current batch." """
    names = sorted(features_by_domain.keys())
    total = features_by_domain[names[0]].new_zeros(())
    n_pairs = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            total = total + mmd2_unbiased(features_by_domain[names[i]], features_by_domain[names[j]], kernel_mults)
            n_pairs += 1
    return total / max(n_pairs, 1)
