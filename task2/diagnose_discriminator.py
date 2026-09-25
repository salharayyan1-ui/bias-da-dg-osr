"""Label-free diagnostic for the DANN/CDAN discriminator input (Task 2).

Run v1 fed the discriminator the unit-norm feature f/||f||. Its domain loss
stayed near ln 2 (~0.64 in the first epoch, GRL alpha < 0.17), its accuracy then
fell to chance, while a linear probe on the frozen features separated source
from Sketch at 99.6%. This script isolates the effect of the input scale on
the discriminator alone. It takes the frozen Source-only backbone and trains the prescribed
discriminator (512-256-2, ReLU, dropout 0.5, AdamW 1e-4 / 1e-4) with the
training protocol's batches (8 per source domain + 24 Sketch), with no
gradient reversal, on three versions of the same features:

    raw         f
    unit_norm   f / ||f||                 (run v1)
    sqrt_d_norm sqrt(512) * f / ||f||     (run v2)

It uses domain identities only; no Sketch class label is read (the dataset
object carries labels, and this script discards them).

    python task2/diagnose_discriminator.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import checkpoints_dir, results_dir  # noqa: E402
from common.seed import set_seed  # noqa: E402
from shared.pacs import PACSImageDataset  # noqa: E402
from shared.pacs_engine import SOURCE_DOMAINS, TARGET_DOMAIN, eval_transform  # noqa: E402
from task2.models.backbone import PACSNet  # noqa: E402
from task2.models.domain_discriminator import DomainDiscriminator  # noqa: E402
from task2.train import get_splits  # noqa: E402

SEED = 6304
EPOCHS = 3
STEPS_PER_EPOCH = 234  # = largest source training split (cartoon, 1875) // 8, as in training
EVAL_EVERY = 39


@torch.no_grad()
def features(model, ds, device):
    out = []
    for x, _ in DataLoader(ds, batch_size=128, shuffle=False, num_workers=4):  # labels discarded
        out.append(model.features(x.to(device)).float().cpu())
    return torch.cat(out)


def transform_input(f, variant):
    if variant == "raw":
        return f
    if variant == "unit_norm":
        return F.normalize(f, dim=1)
    return F.normalize(f, dim=1) * math.sqrt(f.shape[1])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = get_splits()
    model = PACSNet(n_classes=7)
    model.load_state_dict(torch.load(checkpoints_dir(2) / "source_only.pth", map_location="cpu")["model"])
    model.to(device).eval()

    feats = {d: features(model, PACSImageDataset(splits[d]["paths"], splits[d]["labels"], splits[d]["train_idx"],
                                                 eval_transform()), device) for d in SOURCE_DOMAINS}
    t = splits[TARGET_DOMAIN]
    feats["target"] = features(model, PACSImageDataset(t["paths"], t["labels"], None, eval_transform()), device)

    # 80/20 split per domain for training / held-out discriminator accuracy
    rng = np.random.RandomState(SEED)
    tr, te = {}, {}
    for d, f in feats.items():
        perm = rng.permutation(len(f))
        k = int(0.8 * len(f))
        tr[d], te[d] = f[perm[:k]], f[perm[k:]]
    te_src = torch.cat([te[d] for d in SOURCE_DOMAINS])
    n = min(len(te_src), len(te["target"]))
    te_x = torch.cat([te_src[:n], te["target"][:n]])
    te_y = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])

    # Linear-probe reference on the same held-out split (standardized, C=1)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr_src = torch.cat([tr[d] for d in SOURCE_DOMAINS])
    m = min(len(tr_src), len(tr["target"]))
    Xp = torch.cat([tr_src[:m], tr["target"][:m]]).numpy()
    yp = np.r_[np.zeros(m), np.ones(m)]
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))
    probe.fit(Xp, yp)
    probe_acc = float(probe.score(te_x.numpy(), te_y.numpy()))

    result = {"probe_accuracy_standardized_raw_features": probe_acc,
              "mean_feature_norm": {d: float(f.norm(dim=1).mean()) for d, f in feats.items()},
              "steps_per_epoch": STEPS_PER_EPOCH, "curves": {}}
    for variant in ["raw", "unit_norm", "sqrt_d_norm"]:
        set_seed(SEED)
        disc = DomainDiscriminator(512).to(device)
        opt = torch.optim.AdamW(disc.parameters(), lr=1e-4, weight_decay=1e-4)
        g = torch.Generator().manual_seed(SEED)
        curve = []
        for step in range(1, EPOCHS * STEPS_PER_EPOCH + 1):
            disc.train()
            xb = [tr[d][torch.randint(len(tr[d]), (8,), generator=g)] for d in SOURCE_DOMAINS]
            xb.append(tr["target"][torch.randint(len(tr["target"]), (24,), generator=g)])
            x = transform_input(torch.cat(xb), variant).to(device)
            y = torch.cat([torch.zeros(24, dtype=torch.long), torch.ones(24, dtype=torch.long)]).to(device)
            loss = F.cross_entropy(disc(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % EVAL_EVERY == 0:
                disc.eval()
                with torch.no_grad():
                    logits = disc(transform_input(te_x, variant).to(device))
                    acc = float((logits.argmax(1).cpu() == te_y).float().mean())
                    held_loss = float(F.cross_entropy(logits, te_y.to(device)))
                curve.append({"step": step, "epoch": step / STEPS_PER_EPOCH, "heldout_acc": acc, "heldout_loss": held_loss})
        result["curves"][variant] = curve
        print(variant, " ".join(f"{c['epoch']:.2f}:{c['heldout_acc']:.3f}" for c in curve))
    print("linear probe (standardized raw features):", round(probe_acc, 4))
    out = results_dir(2) / "diagnostics" / "discriminator_input_scale.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
