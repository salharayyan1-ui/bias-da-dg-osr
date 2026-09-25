"""CIFAR-appropriate ResNet-18 for Task 4, shared by Vanilla/GCSC/PROSER/RPL.

Per spec: "Replace the ImageNet 7x7, stride-2 first convolution with a 3x3,
stride-1 convolution and remove the initial max-pooling layer. Operate on the
original 32x32 images." We build this from torchvision's resnet18 (random
init, NOT pretrained -- Task 4 trains "from random initialization") and patch
the stem, since torchvision's BasicBlock/layer structure is otherwise exactly
the well-known "CIFAR ResNet-18" used across the OSR literature (Vaze et al.
2022, Zhou et al. 2021, etc).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_cifar_resnet18(n_classes: int = 10) -> nn.Module:
    from torchvision.models import resnet18
    net = resnet18(weights=None, num_classes=n_classes)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.feature_dim = net.fc.in_features  # 512
    return net


class PenultimateExtractor(nn.Module):
    """Wraps a CIFAR ResNet-18 to expose BOTH the penultimate 512-d feature
    f(x) and the 10-class logits z(x) in one forward pass (Task 4 Step 1
    requires saving both for every example)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.feature_dim = net.feature_dim

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        # Manually replay torchvision's ResNet.forward up to avgpool so we can
        # keep the pre-fc feature around.
        n = self.net
        h = n.conv1(x)
        h = n.bn1(h)
        h = n.relu(h)
        h = n.maxpool(h)  # nn.Identity() here, per the CIFAR stem patch
        h = n.layer1(h)
        h = n.layer2(h)
        h = n.layer3(h)
        h = n.layer4(h)
        h = n.avgpool(h)
        feat = torch.flatten(h, 1)
        logits = n.fc(feat)
        return feat, logits

    def forward_up_to_layer2(self, x) -> torch.Tensor:
        """For PROSER's manifold mixup: everything through layer2 (phi_pre)."""
        n = self.net
        h = n.conv1(x)
        h = n.bn1(h)
        h = n.relu(h)
        h = n.maxpool(h)
        h = n.layer1(h)
        h = n.layer2(h)
        return h

    def forward_from_layer3(self, h) -> tuple[torch.Tensor, torch.Tensor]:
        """The remainder of the network, given a (possibly mixed) layer2
        output -- used by PROSER to finish the forward pass on mixed hidden
        states."""
        n = self.net
        h = n.layer3(h)
        h = n.layer4(h)
        h = n.avgpool(h)
        feat = torch.flatten(h, 1)
        logits = n.fc(feat)
        return feat, logits
