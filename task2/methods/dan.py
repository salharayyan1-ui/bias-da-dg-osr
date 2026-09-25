"""Task 2 Step 2: DAN. L = L_cls + lambda_MMD * MMD^2(pooled source features,
target features) on the 512-d feature before the head, with a sum of RBF
kernels at 0.5/1/2 x the median pairwise squared distance of the combined
batch (shared/mmd.py, the same implementation DAN-DG uses in Task 3). The
unbiased U-statistic estimate of MMD^2 is used (see shared/mmd.py)."""
import torch

from shared.mmd import mmd2_unbiased
from shared.pacs_engine import SOURCE_DOMAINS, MethodSpec


def make_method(lambda_mmd: float = 1.0) -> MethodSpec:
    def extra_loss(model, feats, progress):
        src = torch.cat([feats[d] for d in SOURCE_DOMAINS])
        mmd = mmd2_unbiased(src, feats["target"])
        return lambda_mmd * mmd, {"mmd2": float(mmd.detach())}

    return MethodSpec(name="dan", uses_target=True, extra_loss=extra_loss, hparams={"lambda_mmd": lambda_mmd})
