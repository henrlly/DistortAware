"""Two-head DID classifier: a ResNet on the first-order error map (d1) and one on
the second-order error map (d2).  Final score fuses the two heads.

Following arXiv:2602.23732, the two orders carry complementary signal; we fuse by
averaging the two sigmoid probabilities (an alternative AND rule at per-head
threshold 1-sqrt(0.5) is provided for reference in evaluate.py).
"""
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights, resnet50, ResNet50_Weights)

_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
}


def _backbone(pretrained=True, name="resnet18"):
    ctor, weights = _BACKBONES[name]
    m = ctor(weights=weights if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


class DIDClassifier(nn.Module):
    def __init__(self, pretrained=True, backbone="resnet18"):
        super().__init__()
        self.backbone = backbone
        self.head_d1 = _backbone(pretrained, backbone)
        self.head_d2 = _backbone(pretrained, backbone)

    @staticmethod
    def norm_d1(d1):
        # abs error in [0,1] -> ImageNet-style standardisation
        mean = torch.tensor([0.09, 0.09, 0.09], device=d1.device).view(1, 3, 1, 1)
        std = torch.tensor([0.12, 0.12, 0.12], device=d1.device).view(1, 3, 1, 1)
        return (d1 - mean) / std

    @staticmethod
    def norm_d2(d2):
        # signed, roughly zero-mean; scale up
        return d2 / 0.08

    def forward(self, d1, d2):
        l1 = self.head_d1(self.norm_d1(d1)).squeeze(1)
        l2 = self.head_d2(self.norm_d2(d2)).squeeze(1)
        return l1, l2

    @torch.no_grad()
    def score(self, d1, d2):
        l1, l2 = self.forward(d1, d2)
        return 0.5 * (torch.sigmoid(l1) + torch.sigmoid(l2))
