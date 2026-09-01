"""Small residual-aware multi-task U-Net for SID segmentation and classification."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


CLASS_NAMES = ["authentic", "fully_synthetic", "ai_tampered"]


def image_tensor(image: Image.Image, size: int = 192) -> torch.Tensor:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    residual = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
    # Signed residual is centered at 0.5 so positive/negative detail is retained.
    residual = np.clip(residual * 4.0 + 0.5, 0.0, 1.0)
    data = np.concatenate([rgb.astype(np.float32) / 255.0, residual[..., None]], axis=2)
    return torch.from_numpy(data).permute(2, 0, 1)


def target_mask(mask: Image.Image | None, label: int, size: int = 192) -> torch.Tensor:
    if label == 0:
        result = np.zeros((size, size), dtype=np.float32)
    elif label == 1:
        result = np.ones((size, size), dtype=np.float32)
    else:
        if mask is None:
            raise ValueError("SID tampered sample has no localization mask")
        array = np.asarray(mask.convert("L"), dtype=np.uint8)
        result = cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        result = (result >= 0.25).astype(np.float32)
    return torch.from_numpy(result).unsqueeze(0)


class Block(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.BatchNorm2d(outgoing), nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.BatchNorm2d(outgoing), nn.SiLU(),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualUNet(nn.Module):
    """~0.5M parameter U-Net with a three-class bottleneck head."""

    def __init__(self):
        super().__init__()
        self.e1, self.e2, self.e3 = Block(4, 16), Block(16, 32), Block(32, 64)
        self.bridge = Block(64, 96)
        self.d3, self.d2, self.d1 = Block(96 + 64, 64), Block(64 + 32, 32), Block(32 + 16, 16)
        self.mask_head = nn.Conv2d(16, 1, 1)
        self.class_head = nn.Linear(96, 3)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        bridge = self.bridge(F.max_pool2d(e3, 2))
        classes = self.class_head(F.adaptive_avg_pool2d(bridge, 1).flatten(1))
        d3 = self.d3(torch.cat([F.interpolate(bridge, size=e3.shape[-2:], mode="bilinear"), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear"), e1], 1))
        return self.mask_head(d1), classes


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum((1, 2, 3))
    denominator = probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1.0 - (2 * intersection + 1) / (denominator + 1)).mean()
