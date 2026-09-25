"""Task 3 Step 3: non-adaptive SAM (Foret et al., 2021) around AdamW.

Per source batch: (1) gradient g at w; (2) move to w + rho * g / ||g||_2 (one
global norm over all trainable parameters); (3) gradient at the perturbed
point; (4) restore w and let AdamW (lr 1e-4, wd 1e-4) step with that gradient.
Two forward/backward passes per step; BatchNorm running statistics stay frozen
in both passes (PACSNet.train()). Implemented in shared/pacs_engine.py; the
two-step structure follows the public reference github.com/davda54/sam."""
from shared.pacs_engine import MethodSpec


def make_method(rho: float = 0.05) -> MethodSpec:
    return MethodSpec(name="sam", uses_target=False, sam_rho=rho, hparams={"rho": rho})
