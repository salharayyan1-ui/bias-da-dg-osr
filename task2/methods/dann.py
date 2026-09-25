"""Task 2 Step 3: DANN. A binary discriminator on the 512-d feature learns
pooled-source vs target; the gradient-reversal layer (alpha(p) schedule)
sends the reversed gradient to the backbone. Only source images enter the
class loss; source and target images both enter the domain loss (unit weight)."""
import torch
import torch.nn.functional as F

from shared.pacs_engine import SOURCE_DOMAINS, MethodSpec
from task2.models.domain_discriminator import (DomainDiscriminator, discriminator_input, grad_reverse,
                                               grl_alpha_schedule)


def domain_labels(n_src, n_tgt, device):
    return torch.cat([torch.zeros(n_src, dtype=torch.long, device=device),
                      torch.ones(n_tgt, dtype=torch.long, device=device)])


def make_method(max_alpha: float = 1.0, feature_dim: int = 512) -> MethodSpec:
    disc = DomainDiscriminator(feature_dim)

    def extra_loss(model, feats, progress):
        src = torch.cat([feats[d] for d in SOURCE_DOMAINS])
        tgt = feats["target"]
        alpha = grl_alpha_schedule(progress, max_alpha)
        # The discriminator sees the feature's direction at a fixed norm of
        # sqrt(512). With frozen BatchNorm statistics nothing bounds the feature
        # scale, and under AdamW the reversed gradient then inflates the
        # features to make the discriminator confidently wrong (observed:
        # feature norm x1.5 per step, losses ~1e5 within the first epoch).
        # Fixing the norm removes that degenerate direction; the sqrt(512)
        # factor keeps the input on the raw feature's per-coordinate scale (see
        # discriminator_input). Architecture, GRL schedule and unit loss weight
        # are unchanged.
        f = grad_reverse(discriminator_input(torch.cat([src, tgt])), alpha)
        logits = disc(f)
        y = domain_labels(len(src), len(tgt), f.device)
        loss = F.cross_entropy(logits, y)
        acc = float((logits.argmax(1) == y).float().mean())
        return loss, {"domain_loss": float(loss.detach()), "disc_acc": acc, "grl_alpha": alpha}

    return MethodSpec(name="dann", uses_target=True, extra_loss=extra_loss, extra_modules=[disc],
                      hparams={"max_alpha": max_alpha})
