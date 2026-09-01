"""Lightweight configuration for optional automatic physics proposals.

This module intentionally has no OpenCV or model-runtime imports.  Keeping the
configuration separate lets detector-only and checkpoint-contract tooling
import the unified entry point without installing the optional physics stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"
DEFAULT_DINO_MODEL = "vit_small_patch16_dinov3.lvd1689m"
DEFAULT_CLIPSEG_REVISION = "999e0328d9e10b484360c477313983f9afdd7050"
DEFAULT_DINO_REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"


@dataclass(slots=True)
class AutomaticProposalConfig:
    enabled: bool = False
    mask_backend: str = "heuristic"
    feature_backend: str = "appearance"
    object_backend: str = "edges"
    mask_model: str = DEFAULT_CLIPSEG_MODEL
    dino_model: str = DEFAULT_DINO_MODEL
    mask_revision: str | None = DEFAULT_CLIPSEG_REVISION
    dino_revision: str | None = DEFAULT_DINO_REVISION
    device: str | None = None
    cache_dir: str | Path | None = None
    local_files_only: bool = False
    allow_model_fallback: bool = True
    shadow_threshold: float = 0.40
    mirror_threshold: float = 0.54
    min_shadow_area_fraction: float = 0.0015
    max_shadow_area_fraction: float = 0.34
    min_mirror_area_fraction: float = 0.025
    max_mirror_area_fraction: float = 0.82
    min_shadow_pair_confidence: float = 0.28
    min_reflection_pair_confidence: float = 0.38
    min_feature_similarity: float = 0.56
    min_feature_margin: float = 0.015
    max_shadow_pairs: int = 24
    max_reflection_pairs: int = 36
    appearance_grid_size: int = 32
    appearance_fallback_on_insufficient_external: bool = False
    min_pairs: int = 3
    min_pairs_for_definitive_inconsistency: int = 4
    shadow_inlier_threshold_degrees: float = 12.0
    reflection_inlier_threshold_degrees: float = 8.0

    def __post_init__(self) -> None:
        if not self.mask_revision or (
            self.mask_model != DEFAULT_CLIPSEG_MODEL
            and self.mask_revision == DEFAULT_CLIPSEG_REVISION
        ):
            self.mask_revision = None
        if not self.dino_revision or (
            self.dino_model != DEFAULT_DINO_MODEL
            and self.dino_revision == DEFAULT_DINO_REVISION
        ):
            self.dino_revision = None
        if self.mask_backend not in {"heuristic", "clipseg"}:
            raise ValueError("mask_backend must be `heuristic` or `clipseg`")
        if self.feature_backend not in {"appearance", "dinov3", "external", "none"}:
            raise ValueError(
                "feature_backend must be `appearance`, `dinov3`, `external`, or `none`"
            )
        if self.object_backend not in {"edges", "torchvision", "none"}:
            raise ValueError("object_backend must be `edges`, `torchvision`, or `none`")
        for name in (
            "shadow_threshold",
            "mirror_threshold",
            "min_shadow_area_fraction",
            "max_shadow_area_fraction",
            "min_mirror_area_fraction",
            "max_mirror_area_fraction",
            "min_shadow_pair_confidence",
            "min_reflection_pair_confidence",
            "min_feature_similarity",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie within [0, 1]")
        if self.min_feature_margin < 0.0:
            raise ValueError("min_feature_margin cannot be negative")
        if self.max_shadow_area_fraction <= self.min_shadow_area_fraction:
            raise ValueError("shadow area bounds are invalid")
        if self.max_mirror_area_fraction <= self.min_mirror_area_fraction:
            raise ValueError("mirror area bounds are invalid")
        if self.min_pairs < 3:
            raise ValueError("Automatic projective tests require at least three pairs")
        if self.max_shadow_pairs < self.min_pairs or self.max_reflection_pairs < self.min_pairs:
            raise ValueError("Automatic pair limits cannot be below min_pairs")
        if self.min_pairs_for_definitive_inconsistency < self.min_pairs:
            raise ValueError(
                "min_pairs_for_definitive_inconsistency cannot be below min_pairs"
            )
        if self.appearance_grid_size < 8:
            raise ValueError("appearance_grid_size must be at least eight")
        for name in (
            "shadow_inlier_threshold_degrees",
            "reflection_inlier_threshold_degrees",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 45.0:
                raise ValueError(f"{name} must lie within (0, 45]")


__all__ = [
    "AutomaticProposalConfig",
    "DEFAULT_CLIPSEG_MODEL",
    "DEFAULT_CLIPSEG_REVISION",
    "DEFAULT_DINO_MODEL",
    "DEFAULT_DINO_REVISION",
]
