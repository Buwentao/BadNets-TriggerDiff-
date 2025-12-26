from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


@torch.no_grad()
def extract_resnet18_feat(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    x = backbone.avgpool(x)
    x = torch.flatten(x, 1)
    return x


@torch.no_grad()
def extract_resnet18_layer4(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    return x


class ResNet18WithMask(nn.Module):
    def __init__(self, backbone: nn.Module, mask: torch.Tensor) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mask", mask.float().view(1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = extract_resnet18_feat(self.backbone, x)
        feat = feat * self.mask
        return self.backbone.fc(feat)

    @torch.no_grad()
    def forward_with_feat(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = extract_resnet18_feat(self.backbone, x)
        masked = feat * self.mask
        logits = self.backbone.fc(masked)
        return feat, logits


class ResNet18WithLayer4Mask(nn.Module):
    """
    Apply a channel mask on layer4 output (N,512,H,W), then avgpool->fc.
    """

    def __init__(self, backbone: nn.Module, channel_mask: torch.Tensor) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("layer4_mask", channel_mask.float().view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = extract_resnet18_layer4(self.backbone, x)
        h = h * self.layer4_mask
        h = self.backbone.avgpool(h)
        feat = torch.flatten(h, 1)
        return self.backbone.fc(feat)

    @torch.no_grad()
    def forward_with_layer4(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = extract_resnet18_layer4(self.backbone, x)
        masked = h * self.layer4_mask
        pooled = self.backbone.avgpool(masked)
        feat = torch.flatten(pooled, 1)
        logits = self.backbone.fc(feat)
        return h, logits

