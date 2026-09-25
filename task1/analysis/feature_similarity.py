"""Runs every Task 1 model on a batch of common 224x224 images in one pass.

`ModelSuite.run` returns both the backbone representations (for Step 6) and
the predictions of the four evaluated models, so every step uses exactly the
same forward pass:
  resnet50 / vit_b_16 / clip_head : frozen backbone + trained linear head
  clip_zeroshot                    : CLIP image embedding vs. text prompts
"""
from __future__ import annotations

import numpy as np
import torch

from task1.models.backbones import BACKBONES, extract_features

MODEL_KEYS = ("resnet50", "vit_b_16", "clip_head", "clip_zeroshot")
MODEL_TO_BACKBONE = {"resnet50": "resnet50", "vit_b_16": "vit_b_16",
                     "clip_head": "clip_vit_b_32", "clip_zeroshot": "clip_vit_b_32"}


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class ModelSuite:
    def __init__(self, backbones: dict, heads: dict, class_names: list[str], device):
        self.backbones, self.heads, self.device = backbones, heads, device
        clip = backbones["clip_vit_b_32"]
        self.text_emb = clip.clip_text_embeddings(class_names, device).float().cpu().numpy()
        self.logit_scale = clip.clip_logit_scale()

    @torch.no_grad()
    def run(self, images01: torch.Tensor) -> dict:
        feats = {name: extract_features(self.backbones[name], images01, self.device) for name in BACKBONES}
        outputs = {}
        for key in ("resnet50", "vit_b_16", "clip_head"):
            head = self.heads[MODEL_TO_BACKBONE[key]]
            f = torch.from_numpy(feats[MODEL_TO_BACKBONE[key]]).to(self.device)
            logits = head(f).float().cpu().numpy()
            outputs[key] = {"logits": logits, "probs": softmax_np(logits)}
        # Zero-shot: softmax over logit_scale * cosine(image, text prompt).
        zs_logits = self.logit_scale * feats["clip_vit_b_32"] @ self.text_emb.T
        outputs["clip_zeroshot"] = {"logits": zs_logits, "probs": softmax_np(zs_logits)}
        for out in outputs.values():
            out["preds"] = out["probs"].argmax(axis=1)
        return {"feats": feats, "outputs": outputs}
