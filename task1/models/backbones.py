"""Frozen backbones, linear classifier heads, and CLIP zero-shot for Task 1.

Representations (per the spec):
  * ResNet-50 (IMAGENET1K_V2): global-average-pooled 2048-d feature.
  * ViT-B/16 (IMAGENET1K_V1): final (post-LayerNorm) class token, 768-d.
  * OpenCLIP ViT-B-32 (openai): L2-normalized 512-d image embedding.

Every backbone receives the same 224x224 image in [0, 1] and applies its own
normalization inside `forward`.
"""
from __future__ import annotations

import copy
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BackboneName = Literal["resnet50", "vit_b_16", "clip_vit_b_32"]
BACKBONES: tuple[BackboneName, ...] = ("resnet50", "vit_b_16", "clip_vit_b_32")


class _Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


class FrozenBackbone(nn.Module):
    def __init__(self, name: BackboneName):
        super().__init__()
        self.name = name
        if name == "resnet50":
            from torchvision.models import ResNet50_Weights, resnet50

            net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.body = nn.Sequential(*list(net.children())[:-1])  # through AdaptiveAvgPool2d
            self.feature_dim = net.fc.in_features
            self.normalize = _Normalize(IMAGENET_MEAN, IMAGENET_STD)
        elif name == "vit_b_16":
            from torchvision.models import ViT_B_16_Weights, vit_b_16

            self.body = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.feature_dim = self.body.hidden_dim
            self.normalize = _Normalize(IMAGENET_MEAN, IMAGENET_STD)
        elif name == "clip_vit_b_32":
            import open_clip

            # force_quick_gelu: the OpenAI CLIP checkpoints were trained with
            # QuickGELU. open_clip >= 3.0 no longer applies it implicitly for the
            # plain "ViT-B-32" config and only warns about the mismatch, which
            # would run the pretrained weights through the wrong activation.
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai", force_quick_gelu=True)
            self.body = model
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self.feature_dim = model.visual.output_dim
            self.normalize = _Normalize(CLIP_MEAN, CLIP_STD)
        else:
            raise ValueError(name)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # always frozen / eval mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x01)
        if self.name == "resnet50":
            return self.body(x).flatten(1)
        if self.name == "vit_b_16":
            # Mirrors torchvision VisionTransformer.forward up to (not including)
            # the classification head: encoder output includes the final LayerNorm.
            net = self.body
            h = net._process_input(x)
            h = torch.cat([net.class_token.expand(h.shape[0], -1, -1), h], dim=1)
            return net.encoder(h)[:, 0]
        feat = self.body.encode_image(x)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def clip_text_embeddings(self, class_names: list[str], device) -> torch.Tensor:
        """Normalized text embeddings for the fixed prompt 'a photo of a {class}.'"""
        assert self.name == "clip_vit_b_32"
        tokens = self.tokenizer([f"a photo of a {c}." for c in class_names]).to(device)
        return F.normalize(self.body.encode_text(tokens), dim=-1)

    def clip_logit_scale(self) -> float:
        return float(self.body.logit_scale.exp())


@torch.no_grad()
def extract_features(backbone: FrozenBackbone, images01: torch.Tensor, device, batch_size: int = 64) -> np.ndarray:
    """images01: [N,3,224,224] float in [0,1] (CPU). Returns [N,D] float32."""
    out = []
    for i in range(0, len(images01), batch_size):
        out.append(backbone(images01[i:i + batch_size].to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def extract_features_from_loader(backbone: FrozenBackbone, loader, device):
    feats, labels = [], []
    for x, y in loader:
        feats.append(backbone(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


class LinearHead(nn.Module):
    """Linear classifier on the frozen representation, with a fixed per-dimension
    standardization (training-split mean/std) in front of it.

    The standardization is an affine map, so head(f) = W' f + b' is still a
    linear classifier on the unchanged representation; it only puts the three
    backbones' features on a common scale for the prescribed optimizer. Without
    it (run v1) the CLIP head, whose input is a unit-norm 512-d embedding
    (coordinates ~0.04), learned logits ~20x too small in the epochs before
    early stopping: 98.4% accuracy at 0.59 mean confidence and a validation
    loss of 0.35, versus 0.05-0.06 for ResNet-50 and ViT-B/16."""

    def __init__(self, feature_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, n_classes)
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("std", torch.ones(feature_dim))

    def set_standardization(self, train_feats: np.ndarray, eps: float = 1e-6):
        self.mean.copy_(torch.from_numpy(train_feats.mean(0)).float())
        self.std.copy_(torch.from_numpy(train_feats.std(0) + eps).float())

    def forward(self, feats):
        return self.fc((feats - self.mean) / self.std)


def train_linear_head(train_feats: np.ndarray, train_labels: np.ndarray, val_feats: np.ndarray,
                      val_labels: np.ndarray, n_classes: int, device, seed: int = 6304,
                      max_epochs: int = 50, lr: float = 1e-3, weight_decay: float = 1e-4,
                      patience: int = 5, batch_size: int = 64):
    """AdamW linear head on cached frozen features (equivalent to running the
    frozen backbone every epoch, since no augmentation is used). Features are
    standardized with training-split statistics inside the head (see
    LinearHead). Early stopping after `patience` epochs without an improvement
    in validation accuracy; the best-validation weights are restored. Returns
    (head, history, best_val_acc)."""
    torch.manual_seed(seed)
    head = LinearHead(train_feats.shape[1], n_classes)
    head.set_standardization(train_feats)
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr = torch.from_numpy(train_feats).float().to(device)
    ytr = torch.from_numpy(np.asarray(train_labels)).long().to(device)
    Xva = torch.from_numpy(val_feats).float().to(device)
    yva = torch.from_numpy(np.asarray(val_labels)).long().to(device)
    gen = torch.Generator().manual_seed(seed)

    best_acc, best_state, since, history = -1.0, copy.deepcopy(head.state_dict()), 0, []
    for epoch in range(max_epochs):
        head.train()
        perm = torch.randperm(len(Xtr), generator=gen).to(device)
        tot, n = 0.0, 0
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(head(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            n += len(idx)
        head.eval()
        with torch.no_grad():
            val_logits = head(Xva)
            val_acc = float((val_logits.argmax(1) == yva).float().mean())
            val_loss = float(F.cross_entropy(val_logits, yva))
        history.append({"epoch": epoch, "train_loss": tot / n, "val_loss": val_loss, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc, best_state, since = val_acc, copy.deepcopy(head.state_dict()), 0
        else:
            since += 1
            if since >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head, history, best_acc
