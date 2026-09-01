"""PatchHead detector: a frozen DINOv3 ViT-L/16 backbone + LoRA adapters +
a small spatial head that scores every patch and pools the patch evidence.

Reproduces the recipe of "PatchHead: Learning Spatial Patch Evidence for
Generalizable AI-Generated Image Detection" (arXiv:2608.09223) at hackathon
scale, and runs on the *same* WildFake / SID_Set subsets and the same 14-
transform robustness suite as the DID detector in ../did, so the two can be
compared image-for-image (see compare.py).

Backbone weights come from the non-gated timm mirror
`timm/vit_large_patch16_dinov3.lvd1689m` (Meta's `facebook/dinov3-*` repos are
gated).  ~300M frozen params; only the LoRA adapters (~3.1M) + the heads
(~0.3M) are trained.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .transforms import DISTORTION_TARGET_DIM, MAGNITUDE_NAMES, TYPE_NAMES
except ImportError:  # direct execution from patchhead/
    from transforms import DISTORTION_TARGET_DIM, MAGNITUDE_NAMES, TYPE_NAMES

DINOV3 = "vit_large_patch16_dinov3.lvd1689m"
ANALYTIC_FEATURE_DIM = 12
MAGNITUDE_TYPE_INDEX = (0, 1, 2, 3, 4, 4, 4, 5)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """Frozen nn.Linear + trainable low-rank update  W' = W + (alpha/r) B A."""

    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.a = nn.Linear(base.in_features, r, bias=False)
        self.b = nn.Linear(r, base.out_features, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.b(self.drop(self.a(x)))


def inject_lora(model, r=8, alpha=16, targets=("qkv", "proj", "fc1", "fc2")):
    """Replace every `nn.Linear` whose attribute name is in `targets` with a
    LoRALinear wrapper.  Returns the number of adapters injected."""
    n = 0
    for module in model.modules():
        for cname, child in list(module.named_children()):
            if cname in targets and isinstance(child, nn.Linear):
                setattr(module, cname, LoRALinear(child, r, alpha))
                n += 1
    return n


# --------------------------------------------------------------------------- #
# spatial patch head
# --------------------------------------------------------------------------- #
class PatchHead(nn.Module):
    """Per-patch evidence head.  Takes the ViT patch-token grid (B,C,H,W),
    mixes it locally with a depth-wise 3x3 conv, projects to a hidden width,
    and emits three class logits per patch.  The image-level logits are the mean patch
    logit (global average pooling over the evidence map)."""

    def __init__(self, dim, hidden=256, dropout=0.1):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, hidden, 1)
        self.norm = nn.GroupNorm(1, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout)
        self.patch_logit = nn.Conv2d(hidden, 3, 1)

    def forward(self, grid):  # (B, C, H, W)
        z = self.dw(grid)
        z = self.act(self.norm(self.pw(z)))
        z = self.drop(z)
        patch_logits = self.patch_logit(z)                    # (B, 3, H, W)
        img_logit = patch_logits.flatten(2).mean(dim=2)        # (B, 3)
        return img_logit, patch_logits


def analytic_distortion_features(x):
    """Blind, differentiable distortion cues measured on standardized pixels.

    They are deliberately generic: high-pass/noise energy, gradient and
    Laplacian statistics, plus approximate 8-pixel boundary discontinuities.
    The learned head combines these cues with semantic DINO features so texture
    is less likely to be mistaken for noise or compression.
    """
    gray = x[:, :1] * .299 + x[:, 1:2] * .587 + x[:, 2:3] * .114
    horizontal = gray[..., :, 1:] - gray[..., :, :-1]
    vertical = gray[..., 1:, :] - gray[..., :-1, :]
    smooth = F.avg_pool2d(gray, 3, stride=1, padding=1)
    residual = gray - smooth
    lap_kernel = x.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    laplacian = F.conv2d(gray, lap_kernel, padding=1)

    def mean_std(value):
        flat = value.flatten(1)
        return flat.mean(1), flat.std(1, unbiased=False)

    gray_mean, gray_std = mean_std(gray)
    h_mean, h_std = mean_std(horizontal.abs())
    v_mean, v_std = mean_std(vertical.abs())
    residual_mean, residual_std = mean_std(residual.abs())
    lap_mean, lap_std = mean_std(laplacian.abs())

    # JPEG blocks may be rescaled by canonical preprocessing, so these are
    # supporting cues only; the DINO features remain the main estimator input.
    h_all = horizontal.abs().flatten(1).mean(1).clamp_min(1e-6)
    v_all = vertical.abs().flatten(1).mean(1).clamp_min(1e-6)
    h_boundary = horizontal[..., 7::8].abs().flatten(1).mean(1) if horizontal.shape[-1] >= 8 else h_all
    v_boundary = vertical[..., 7::8, :].abs().flatten(1).mean(1) if vertical.shape[-2] >= 8 else v_all
    h_block_ratio = (h_boundary / h_all).clamp(0, 5)
    v_block_ratio = (v_boundary / v_all).clamp(0, 5)
    return torch.stack([
        gray_mean, gray_std, h_mean, h_std, v_mean, v_std,
        residual_mean, residual_std, lap_mean, lap_std,
        h_block_ratio, v_block_ratio,
    ], dim=1)


class DistortionHead(nn.Module):
    """Hybrid learned/analytic blind distortion type and severity estimator."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(dim + ANALYTIC_FEATURE_DIM),
            nn.Linear(dim + ANALYTIC_FEATURE_DIM, hidden),
            nn.GELU(),
            nn.Dropout(.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.type_head = nn.Linear(hidden, len(TYPE_NAMES))
        self.magnitude_head = nn.Linear(hidden, len(MAGNITUDE_NAMES))

    def forward(self, pooled_features, analytic_features):
        hidden = self.trunk(torch.cat([pooled_features, analytic_features], dim=1))
        return self.type_head(hidden), self.magnitude_head(hidden)


class DistortionThresholdAdapter(nn.Module):
    """Predict a per-image threshold shift in logit space.

    A positive shift raises the AI threshold; a negative shift lowers it.  The
    final layer starts at zero, so enabling the adapter initially reproduces the
    unconditioned detector exactly.
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DISTORTION_TARGET_DIM, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, distortion_vector):
        return 2.5 * torch.tanh(self.net(distortion_vector).squeeze(1))


class PatchHeadDetector(nn.Module):
    def __init__(self, lora_r=8, lora_alpha=16, hidden=256, dropout=0.1,
                 pretrained=True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(DINOV3, pretrained=pretrained,
                                          num_classes=0)
        cfg = timm.data.resolve_model_data_config(self.backbone)
        self.input_size = int(cfg["input_size"][-1])
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.n_lora = inject_lora(self.backbone, lora_r, lora_alpha) if lora_r else 0

        dim = self.backbone.embed_dim
        self.n_prefix = self.backbone.num_prefix_tokens
        self.head = PatchHead(dim, hidden, dropout)
        self.cls_head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout),
                                      nn.Linear(dim, 3))
        self.distortion_head = DistortionHead(dim)
        self.threshold_adapter = DistortionThresholdAdapter()

    # ------------------------------------------------------------------ #
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_counts(self):
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora = sum(p.numel() for n, p in self.named_parameters()
                   if p.requires_grad and (".a." in n or ".b." in n))
        return dict(total=tot, trainable=tr, lora=lora, head=tr - lora,
                    n_lora_adapters=self.n_lora)

    # ------------------------------------------------------------------ #
    def _encode_grid(self, x):
        """Return the shared DINO patch grid and CLS token for all heads."""
        x = (x - self.mean) / self.std
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bicubic",
                              align_corners=False)
        tokens = self.backbone.forward_features(x)             # (B, N, C)
        patches = tokens[:, self.n_prefix:]                    # (B, P, C)
        cls = tokens[:, 0]                                     # (B, C)
        B, P, C = patches.shape
        g = int(round(math.sqrt(P)))
        grid = patches.transpose(1, 2).reshape(B, C, g, g)
        return grid, cls

    def forward_with_features(self, x):
        """Return official logits plus the same-pass dense DINO feature grid.

        The feature grid is not part of the AIGC score.  It is exposed in
        memory so optional physics correspondence heads can reuse the primary
        backbone without serialising hundreds of channels into detector JSON.
        """
        img_logit, cls_logit, patch_logits, _cls, grid = self._shared_forward(x)
        return img_logit, cls_logit, patch_logits, grid

    def _shared_forward(self, x):
        """Run the shared backbone and return features needed by all heads."""
        grid, cls = self._encode_grid(x)
        img_logit, patch_logits = self.head(grid)
        cls_logit = self.cls_head(cls).squeeze(1)
        return img_logit, cls_logit, patch_logits, cls, grid

    def forward(self, x):
        """x: (B,3,H,W) float in [0,1].  Returns img_logit, cls_logit,
        patch_logits (B,h,w)."""
        img_logit, cls_logit, patch_logits, _grid = self.forward_with_features(x)
        return img_logit, cls_logit, patch_logits

    def forward_distortion_aware(self, x, oracle_distortion=None, oracle_mix=0.0):
        """Return base heads plus blind distortion and corrected-score outputs.

        `oracle_distortion` is accepted only for training/ablation. Normal
        inference passes None and therefore conditions solely on predictions.
        """
        img_logit, cls_logit, patch_logits, cls, _grid = self._shared_forward(x)
        analytic = analytic_distortion_features(x)
        type_logits, magnitude_logits = self.distortion_head(cls, analytic)
        type_probabilities = torch.sigmoid(type_logits)
        magnitude_probabilities = torch.sigmoid(magnitude_logits)
        magnitude_gate = type_probabilities[:, list(MAGNITUDE_TYPE_INDEX)]
        predicted = torch.cat(
            [type_probabilities, magnitude_probabilities * magnitude_gate], dim=1)
        condition = predicted
        if oracle_distortion is not None and oracle_mix > 0:
            mix = float(np.clip(oracle_mix, 0.0, 1.0))
            condition = mix * oracle_distortion + (1.0 - mix) * predicted

        probabilities = .5 * (
            torch.softmax(img_logit.float(), dim=1)
            + torch.softmax(cls_logit.float(), dim=1)
        )
        base_score = probabilities[:, 1:].sum(dim=1).clamp(1e-5, 1 - 1e-5)
        base_logit = torch.logit(base_score)
        threshold_shift = self.threshold_adapter(condition.detach())
        corrected_logit = base_logit - threshold_shift
        return img_logit, cls_logit, patch_logits, {
            "type_logits": type_logits,
            "magnitude_logits": magnitude_logits,
            "type_probabilities": type_probabilities,
            "magnitude_probabilities": magnitude_probabilities,
            "predicted_distortion": predicted,
            "analytic_features": analytic,
            "conditioning_distortion": condition,
            "threshold_shift": threshold_shift,
            "base_score": base_score,
            "corrected_logit": corrected_logit,
            "corrected_score": torch.sigmoid(corrected_logit),
        }

    @torch.no_grad()
    def score(self, x, distortion_aware=False):
        if distortion_aware:
            return self.forward_distortion_aware(x)[3]["corrected_score"]
        img_logit, cls_logit, _ = self.forward(x)
        return 0.5 * (torch.softmax(img_logit, dim=1)[:, 1:].sum(dim=1) +
                      torch.softmax(cls_logit, dim=1)[:, 1:].sum(dim=1))


def load_detector(ckpt_path, device=None, pretrained_backbone=True):
    """Rebuild a PatchHeadDetector and load a trainable-only checkpoint onto it."""
    device = device or get_device()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if int(ckpt.get("num_classes", 3)) != 3:
        raise ValueError("This evaluator expects a three-class PatchHead checkpoint; train the new model first.")
    model = PatchHeadDetector(lora_r=int(ckpt.get("lora_r", 8)),
                              pretrained=pretrained_backbone)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    distortion_aware = bool(ckpt.get("distortion_aware", False))
    optional_prefixes = () if distortion_aware else ("distortion_head.", "threshold_adapter.")
    leaked = [m for m in missing if m not in frozen and m not in ("mean", "std")
              and not m.startswith(optional_prefixes)]
    assert not leaked, f"checkpoint missing trainable tensors: {leaked[:5]}"
    assert not unexpected, f"unexpected tensors in checkpoint: {unexpected[:5]}"
    model.to(device).eval()
    return model, ckpt
