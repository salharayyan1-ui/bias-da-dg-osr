"""Task 2 Step 4: CDAN. Same discriminator width, dropout, GRL schedule and
unit loss weight as DANN, but the discriminator sees g(x) = vec(f (x) p) with
p = softmax(head(f)) for every source and target example. f and p are not
detached and no entropy conditioning is used, so the reversed gradient
reaches the backbone through both f and p (and the head through p)."""
import torch
import torch.nn.functional as F

from shared.pacs_engine import SOURCE_DOMAINS, MethodSpec
from task2.methods.dann import domain_labels
from task2.models.domain_discriminator import (DomainDiscriminator, cdan_conditioning, discriminator_input,
                                               grad_reverse, grl_alpha_schedule)


def make_method(max_alpha: float = 1.0, feature_dim: int = 512, n_classes: int = 7) -> MethodSpec:
    disc = DomainDiscriminator(feature_dim * n_classes)

    def extra_loss(model, feats, progress):
        src = torch.cat([feats[d] for d in SOURCE_DOMAINS])
        tgt = feats["target"]
        f = torch.cat([src, tgt])
        p = F.softmax(model.head(f), dim=1)
        alpha = grl_alpha_schedule(progress, max_alpha)
        # Reversing the gradient at g = vec(f (x) p) is identical to reversing
        # it at both factors; applying the GRL to g keeps that explicit.
        # f enters the conditioning at a fixed norm of sqrt(512), exactly as in
        # DANN (unbounded feature scale under frozen BN + AdamW); p is the
        # classifier's softmax on the raw feature. Nothing is detached.
        g = grad_reverse(cdan_conditioning(discriminator_input(f), p), alpha)
        logits = disc(g)
        y = domain_labels(len(src), len(tgt), f.device)
        loss = F.cross_entropy(logits, y)
        acc = float((logits.argmax(1) == y).float().mean())
        return loss, {"domain_loss": float(loss.detach()), "disc_acc": acc, "grl_alpha": alpha}

    return MethodSpec(name="cdan", uses_target=True, extra_loss=extra_loss, extra_modules=[disc],
                      hparams={"max_alpha": max_alpha})
