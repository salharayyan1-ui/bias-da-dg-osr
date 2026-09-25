"""ResNet-18 network shared by Tasks 2 and 3.

`PACSNet` = ImageNet normalization -> ResNet-18 trunk (IMAGENET1K_V1, through
global average pooling, 512-d) -> 7-way linear head. The whole network is
fine-tuned. Checkpoints are the `state_dict()` of this module, so Task 3 loads
Task 2's Source-only checkpoint into exactly the same class.

BatchNorm policy (both tasks): running means/variances stay frozen at their
ImageNet values; gamma/beta remain trainable. `PACSNet.train()` therefore
puts every BatchNorm layer back into eval mode after switching the rest of the
network to training mode.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from task2.models.classifier_head import ClassifierHead

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_resnet18_backbone(pretrained: bool = True) -> nn.Module:
    """ResNet-18 trunk returning the 512-d pooled feature (fc = Identity)."""
    from torchvision.models import ResNet18_Weights, resnet18

    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    net.fc = nn.Identity()
    net.feature_dim = 512
    return net


def freeze_batchnorm_running_stats(model: nn.Module) -> None:
    """Put only the BatchNorm modules in eval mode (running stats frozen)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


class ImageNetNormalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def imagenet_normalize() -> nn.Module:
    return ImageNetNormalize()


class PACSNet(nn.Module):
    def __init__(self, n_classes: int = 7, pretrained: bool = True):
        super().__init__()
        self.normalize = ImageNetNormalize()
        self.backbone = build_resnet18_backbone(pretrained)
        self.feature_dim = 512
        self.head = ClassifierHead(self.feature_dim, n_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            freeze_batchnorm_running_stats(self)
        return self

    def features(self, x01: torch.Tensor) -> torch.Tensor:
        """x01: images in [0, 1] -> 512-d feature before the classifier head."""
        return self.backbone(self.normalize(x01))

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x01))
