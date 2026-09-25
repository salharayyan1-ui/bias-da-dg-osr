"""Task 3 Step 2: DAN-DG. L = L_ERM + (lambda_DG / 3) * sum over the three
unordered source-domain pairs of MMD^2, with the same multi-kernel RBF MMD as
Task 2 (bandwidths from each pair's own combined batch). Sketch is never used."""
from shared.mmd import pairwise_mmd2_over_domains
from shared.pacs_engine import SOURCE_DOMAINS, MethodSpec


def make_method(lambda_dg: float = 1.0) -> MethodSpec:
    def extra_loss(model, feats, progress):
        assert "target" not in feats, "DAN-DG must never see the target domain"
        mmd = pairwise_mmd2_over_domains({d: feats[d] for d in SOURCE_DOMAINS})
        return lambda_dg * mmd, {"mean_pairwise_mmd2": float(mmd.detach())}

    return MethodSpec(name="dan_dg", uses_target=False, extra_loss=extra_loss, hparams={"lambda_dg": lambda_dg})
