"""Automatic, confidence-gated shadow and planar-reflection proposals.

The geometric cue implementations in :mod:`shadow` and :mod:`reflection`
deliberately start from point correspondences.  This module supplies those
points without making the proposal model the judge of physical consistency:

* semantic/photometric masks propose cast-shadow and mirror regions;
* object boxes or image edges suggest the object-ground end of a shadow;
* dense appearance or DINO features propose direct/reflected point matches;
* the existing robust projective fit remains the independent verifier.

All heavyweight models are optional and imported lazily.  The OpenCV fallback
is useful for deterministic demos and constrained scenes, but its lower
confidence and explicit provenance prevent it from being presented as a
learned detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Protocol

import cv2
import numpy as np
from PIL import Image

from .annotations import ReflectionPair, ShadowPair
from .automatic_config import AutomaticProposalConfig


class AutomaticProposalError(RuntimeError):
    """Raised when a requested automatic proposal backend cannot run safely."""


@dataclass(slots=True)
class SemanticMasks:
    shadow: np.ndarray
    mirror: np.ndarray
    backend: str
    model: str | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DenseFeatures:
    values: np.ndarray
    backend: str
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ObjectDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    label: str | None = None


@dataclass(slots=True)
class RegionProposal:
    kind: str
    component_id: int
    confidence: float
    area_fraction: float
    bbox: tuple[int, int, int, int]
    contour: list[tuple[float, float]]
    mask: np.ndarray = field(repr=False)

    def evidence(self) -> dict[str, Any]:
        return {
            "kind": f"{self.kind}_region",
            "component_id": self.component_id,
            "confidence": round(float(self.confidence), 6),
            "area_fraction": round(float(self.area_fraction), 7),
            "bbox_xyxy": [int(value) for value in self.bbox],
            "contour": [
                [round(float(x), 2), round(float(y), 2)] for x, y in self.contour
            ],
            "role": "proposal_region",
        }


@dataclass(slots=True)
class AutomaticCueProposals:
    cue: str
    pairs: list[ShadowPair] | list[ReflectionPair]
    applicable: bool
    confidence: float
    reason: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AutomaticProposalBundle:
    shadow: AutomaticCueProposals
    reflection: AutomaticCueProposals
    backend: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


class MaskProvider(Protocol):
    def predict(self, image: Image.Image) -> SemanticMasks: ...


class FeatureProvider(Protocol):
    def extract(self, image: Image.Image) -> DenseFeatures: ...


class ObjectProvider(Protocol):
    def detect(self, image: Image.Image) -> tuple[list[ObjectDetection], dict[str, Any]]: ...


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _as_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _normalize_probability(values: Any, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or min(array.shape, default=0) <= 0:
        raise AutomaticProposalError("Semantic masks must be non-empty two-dimensional arrays")
    if not np.isfinite(array).all():
        raise AutomaticProposalError("Semantic masks must contain only finite values")
    if array.shape != shape:
        array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(array, 0.0, 1.0)


def _heuristic_shadow_probability(image: Image.Image) -> np.ndarray:
    """Photometric cast-shadow prior designed to fail conservatively.

    Shadows tend to be darker than their local neighbourhood while preserving
    approximate channel ratios.  Strong internal edges are down-weighted so
    dark objects are less likely to become shadow regions.
    """

    rgb = _as_rgb_array(image).astype(np.float32) / 255.0
    height, width = rgb.shape[:2]
    lab = cv2.cvtColor(np.round(rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(
        np.float32
    )
    luminance = lab[..., 0] / 255.0
    sigma = max(3.0, min(height, width) / 24.0)
    local_luminance = cv2.GaussianBlur(luminance, (0, 0), sigmaX=sigma, sigmaY=sigma)
    darkness = np.clip((local_luminance - luminance - 0.025) / 0.22, 0.0, 1.0)

    local_rgb = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    ratio = (rgb + 0.035) / (local_rgb + 0.035)
    ratio_spread = ratio.max(axis=2) - ratio.min(axis=2)
    chromatic_consistency = np.exp(-7.0 * ratio_spread)

    gx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 1.0)
    smoothness = np.exp(-8.0 * np.clip(edge, 0.0, 1.0))

    probability = darkness * (0.28 + 0.72 * chromatic_consistency) * (
        0.35 + 0.65 * smoothness
    )
    return np.clip(probability.astype(np.float32), 0.0, 1.0)


def _polygon_score(
    gray: np.ndarray, edges: np.ndarray, polygon: np.ndarray, area_fraction: float
) -> float:
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    kernel = np.ones((5, 5), np.uint8)
    inner = cv2.erode(mask, kernel)
    outer = cv2.dilate(mask, kernel)
    ring = cv2.subtract(outer, inner) > 0
    inside = inner > 0
    if int(inside.sum()) < 25 or int(ring.sum()) < 10:
        return 0.0
    border_strength = float((edges[ring] > 0).mean())
    texture = min(float(gray[inside].std()) / 48.0, 1.0)
    area_score = min(area_fraction / 0.12, 1.0)
    return _clamp01(0.18 + 0.48 * border_strength + 0.19 * texture + 0.15 * area_score)


def _heuristic_mirror_probability(image: Image.Image) -> np.ndarray:
    """Return a conservative rectangular planar-reflector prior.

    The fallback deliberately covers only framed, approximately quadrilateral
    reflectors.  Semantic CLIPSeg masks are preferred for uncontrolled scenes.
    """

    rgb = _as_rgb_array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    image_area = float(height * width)
    probability = np.zeros((height, width), dtype=np.float32)

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 0.12 * math.hypot(width, height):
            continue
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        polygon2 = polygon.reshape(4, 2)
        area = abs(float(cv2.contourArea(polygon2)))
        area_fraction = area / image_area
        if not 0.02 <= area_fraction <= 0.84:
            continue
        x, y, box_width, box_height = cv2.boundingRect(polygon2)
        rectangularity = area / max(float(box_width * box_height), 1.0)
        if rectangularity < 0.68 or min(box_width, box_height) < 24:
            continue
        if x <= 1 and y <= 1 and x + box_width >= width - 1 and y + box_height >= height - 1:
            continue
        score = _polygon_score(gray, edges, polygon2, area_fraction)
        score *= min(rectangularity / 0.9, 1.0)
        if score < 0.46:
            continue
        candidate = np.zeros_like(probability)
        cv2.fillConvexPoly(candidate, polygon2, float(score))
        probability = np.maximum(probability, candidate)
    return probability


class HeuristicMaskProvider:
    def predict(self, image: Image.Image) -> SemanticMasks:
        return SemanticMasks(
            shadow=_heuristic_shadow_probability(image),
            mirror=_heuristic_mirror_probability(image),
            backend="opencv_photometric_geometry",
            metadata={
                "learned": False,
                "scope": "constrained_fallback",
                "coordinate_space": "full_frame_pixels",
            },
        )


def _fuse_semantic_and_physical_masks(
    semantic_shadow: np.ndarray,
    semantic_mirror: np.ndarray,
    photometric_shadow: np.ndarray,
    geometric_mirror: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse zero-shot masks without discarding strong deterministic priors."""

    shadow_blend = np.clip(
        0.78 * semantic_shadow
        + 0.22 * np.sqrt(np.clip(semantic_shadow * photometric_shadow, 0.0, 1.0)),
        0.0,
        1.0,
    )
    mirror_blend = np.clip(
        0.86 * semantic_mirror + 0.14 * geometric_mirror, 0.0, 1.0
    )
    shadow = np.maximum(shadow_blend, photometric_shadow)
    mirror = np.maximum(mirror_blend, 0.80 * geometric_mirror)
    return shadow.astype(np.float32), mirror.astype(np.float32)


def _select_device(torch: Any, requested: str | None) -> str:
    if requested:
        return requested
    if bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ClipSegMaskProvider:
    """Zero-shot semantic masks; model loading occurs only on first image."""

    SHADOW_PROMPTS = (
        "cast shadow on the ground",
        "object shadow",
        "a dark cast shadow",
    )
    MIRROR_PROMPTS = (
        "a mirror surface",
        "a framed mirror",
        "a planar reflection",
    )

    def __init__(self, config: AutomaticProposalConfig) -> None:
        self.config = config
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        except ModuleNotFoundError as exc:
            raise AutomaticProposalError(
                "CLIPSeg requires the optional `auto` dependencies (torch and transformers)"
            ) from exc
        cache_dir = str(Path(self.config.cache_dir).expanduser()) if self.config.cache_dir else None
        kwargs = {
            "cache_dir": cache_dir,
            "local_files_only": self.config.local_files_only,
            "revision": self.config.mask_revision,
        }
        try:
            self._processor = CLIPSegProcessor.from_pretrained(
                self.config.mask_model, **kwargs
            )
            self._model = CLIPSegForImageSegmentation.from_pretrained(
                self.config.mask_model, **kwargs
            )
        except Exception as exc:
            raise AutomaticProposalError(
                f"Could not load CLIPSeg model {self.config.mask_model!r}: {exc}"
            ) from exc
        self._torch = torch
        self._device = _select_device(torch, self.config.device)
        self._model.to(self._device).eval()

    def _prompt_map(self, image: Image.Image, prompts: tuple[str, ...]) -> np.ndarray:
        assert self._processor is not None and self._model is not None
        torch = self._torch
        inputs = self._processor(
            text=list(prompts),
            images=[image] * len(prompts),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self._model(**inputs).logits
        probabilities = torch.sigmoid(logits).float().cpu().numpy()
        if probabilities.ndim == 2:
            probabilities = probabilities[None, ...]
        combined = np.quantile(probabilities, 0.72, axis=0).astype(np.float32)
        return cv2.resize(combined, image.size, interpolation=cv2.INTER_LINEAR)

    def predict(self, image: Image.Image) -> SemanticMasks:
        self._load()
        semantic_shadow = self._prompt_map(image, self.SHADOW_PROMPTS)
        semantic_mirror = self._prompt_map(image, self.MIRROR_PROMPTS)
        photometric_shadow = _heuristic_shadow_probability(image)
        geometric_mirror = _heuristic_mirror_probability(image)
        # Semantics proposes the region while the cheap physical priors retain
        # dark, chromatically stable shadows and framed mirrors that the prompt
        # model can miss.  The max fusion weights were checked on the existing
        # bounded 24-image SBU/PMD smoke subsets: shadow macro IoU improved from
        # 0.452 to 0.487, while the conservative 0.8 mirror fallback preserved
        # precision and nudged macro IoU from 0.296 to 0.298.
        shadow, mirror = _fuse_semantic_and_physical_masks(
            semantic_shadow,
            semantic_mirror,
            photometric_shadow,
            geometric_mirror,
        )
        return SemanticMasks(
            shadow=shadow,
            mirror=mirror,
            backend="clipseg_prompt_ensemble",
            model=self.config.mask_model,
            metadata={
                "learned": True,
                "zero_shot": True,
                "device": self._device,
                "revision": self.config.mask_revision,
                "coordinate_space": "full_frame_pixels",
                "shadow_prompts": list(self.SHADOW_PROMPTS),
                "mirror_prompts": list(self.MIRROR_PROMPTS),
                "shadow_fusion": "maximum_of_semantic_photometric_fusion_and_photometric_prior",
                "mirror_fusion": "maximum_of_semantic_geometric_fusion_and_0.8x_geometric_prior",
            },
        )


class DenseAppearanceFeatureProvider:
    """Small, dependency-free dense descriptor for fallback correspondence."""

    def __init__(self, grid_size: int = 32) -> None:
        self.grid_size = grid_size

    def extract(self, image: Image.Image) -> DenseFeatures:
        rgb = _as_rgb_array(image).astype(np.float32) / 255.0
        width, height = image.size
        aspect = width / max(height, 1)
        if aspect >= 1.0:
            columns = self.grid_size
            rows = max(8, int(round(self.grid_size / aspect)))
        else:
            rows = self.grid_size
            columns = max(8, int(round(self.grid_size * aspect)))
        small = cv2.resize(rgb, (columns, rows), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(np.round(small * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(
            np.float32
        ) / 255.0
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
        local_detail = np.abs(gray - blur)
        local_mean = cv2.GaussianBlur(gray, (0, 0), 1.0)
        local_square = cv2.GaussianBlur(gray * gray, (0, 0), 1.0)
        local_std = np.sqrt(np.maximum(local_square - local_mean * local_mean, 0.0))
        values = np.concatenate(
            [
                small,
                lab,
                np.abs(gx)[..., None],
                np.abs(gy)[..., None],
                local_detail[..., None],
                local_std[..., None],
            ],
            axis=2,
        ).astype(np.float32)
        flattened = values.reshape(-1, values.shape[-1])
        mean = flattened.mean(axis=0, keepdims=True)
        std = flattened.std(axis=0, keepdims=True)
        values = ((flattened - mean) / np.maximum(std, 0.08)).reshape(values.shape)
        return DenseFeatures(
            values=values,
            backend="dense_appearance",
            metadata={
                "learned": False,
                "grid_shape": [rows, columns],
                "coordinate_space": "normalized_full_frame",
            },
        )


class TimmDinoFeatureProvider:
    """Small DINOv3 feature grid, independent of the unavailable PatchHead head."""

    def __init__(self, config: AutomaticProposalConfig) -> None:
        self.config = config
        self._torch: Any = None
        self._model: Any = None
        self._device: str | None = None
        self._input_size = 256
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import timm
            import torch
        except ModuleNotFoundError as exc:
            raise AutomaticProposalError(
                "DINOv3 correspondence requires the optional `auto` dependencies (torch and timm)"
            ) from exc
        if self.config.cache_dir:
            torch.hub.set_dir(str(Path(self.config.cache_dir).expanduser() / "torch"))
        try:
            cache_dir = (
                Path(self.config.cache_dir).expanduser() / "huggingface"
                if self.config.cache_dir
                else None
            )
            from huggingface_hub import hf_hub_download

            pretrained_config = timm.get_pretrained_cfg(self.config.dino_model)
            repository = pretrained_config.hf_hub_id
            if not repository:
                raise AutomaticProposalError(
                    f"DINO model {self.config.dino_model!r} has no Hugging Face artifact"
                )
            checkpoint_path = hf_hub_download(
                repository,
                filename="model.safetensors",
                revision=self.config.dino_revision,
                cache_dir=str(cache_dir) if cache_dir else None,
                local_files_only=self.config.local_files_only,
            )
            self._model = timm.create_model(
                self.config.dino_model,
                pretrained=False,
                checkpoint_path=checkpoint_path,
                num_classes=0,
            )
        except Exception as exc:
            raise AutomaticProposalError(
                f"Could not load DINO feature model {self.config.dino_model!r}: {exc}"
            ) from exc
        data_config = timm.data.resolve_model_data_config(self._model)
        self._input_size = int(data_config["input_size"][-1])
        self._mean = np.asarray(data_config["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.asarray(data_config["std"], dtype=np.float32).reshape(1, 3, 1, 1)
        self._torch = torch
        self._device = _select_device(torch, self.config.device)
        self._model.to(self._device).eval()

    def extract(self, image: Image.Image) -> DenseFeatures:
        self._load()
        torch = self._torch
        array = np.asarray(
            image.resize((self._input_size, self._input_size), Image.Resampling.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        batch = np.transpose(array, (2, 0, 1))[None, ...]
        assert self._mean is not None and self._std is not None
        batch = (batch - self._mean) / self._std
        tensor = torch.from_numpy(batch).to(self._device)
        with torch.inference_mode():
            tokens = self._model.forward_features(tensor)
        patch_tokens_only = False
        if isinstance(tokens, dict):
            patch_tokens = tokens.get("x_norm_patchtokens")
            if patch_tokens is not None:
                tokens = patch_tokens
                patch_tokens_only = True
            else:
                tokens = tokens.get("x_prenorm")
        if tokens is None or getattr(tokens, "ndim", None) != 3:
            raise AutomaticProposalError("DINO model did not return a patch-token tensor")
        prefix = (
            0
            if patch_tokens_only
            else int(getattr(self._model, "num_prefix_tokens", 1))
        )
        patches = tokens[:, prefix:, :]
        count = int(patches.shape[1])
        side = int(round(math.sqrt(count)))
        if side * side != count:
            raise AutomaticProposalError(
                f"DINO patch count {count} cannot be mapped to a square full-frame grid"
            )
        values = patches[0].float().cpu().numpy().reshape(side, side, -1)
        return DenseFeatures(
            values=values,
            backend="dinov3_dense_tokens",
            model=self.config.dino_model,
            metadata={
                "learned": True,
                "revision": self.config.dino_revision,
                "grid_shape": [side, side],
                "coordinate_space": "normalized_full_frame",
                "device": self._device,
                "shared_primary_forward": False,
            },
        )


class EdgeObjectProvider:
    def detect(self, image: Image.Image) -> tuple[list[ObjectDetection], dict[str, Any]]:
        return [], {
            "backend": "local_edge_contact",
            "learned": False,
            "note": "Shadow roots are selected from endpoint-adjacent image edges.",
        }


class NullObjectProvider:
    def detect(self, image: Image.Image) -> tuple[list[ObjectDetection], dict[str, Any]]:
        return [], {
            "backend": "none",
            "learned": False,
            "note": "No object detector was requested; local endpoint support remains available.",
        }


class TorchvisionObjectProvider:
    """Portable COCO object boxes used only to improve shadow-root association."""

    def __init__(self, config: AutomaticProposalConfig) -> None:
        self.config = config
        self._model: Any = None
        self._torch: Any = None
        self._device: str | None = None
        self._categories: list[str] = []

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from torchvision.models.detection import (
                FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                fasterrcnn_mobilenet_v3_large_320_fpn,
            )
        except ModuleNotFoundError as exc:
            raise AutomaticProposalError(
                "Torchvision object boxes require the optional `auto` dependencies"
            ) from exc
        if self.config.cache_dir:
            torch.hub.set_dir(str(Path(self.config.cache_dir).expanduser() / "torch"))
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        if self.config.local_files_only:
            checkpoint = (
                Path(torch.hub.get_dir())
                / "checkpoints"
                / Path(str(weights.url)).name
            )
            if not checkpoint.is_file():
                raise AutomaticProposalError(
                    f"Offline object-detector checkpoint is missing: {checkpoint}"
                )
        try:
            self._model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        except Exception as exc:
            raise AutomaticProposalError(f"Could not load torchvision object detector: {exc}") from exc
        self._categories = list(weights.meta.get("categories", []))
        self._torch = torch
        # Detection kernels are more consistently supported on CPU than MPS.
        self._device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device).eval()

    def detect(self, image: Image.Image) -> tuple[list[ObjectDetection], dict[str, Any]]:
        self._load()
        torch = self._torch
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).to(self._device)
        with torch.inference_mode():
            prediction = self._model([tensor])[0]
        detections: list[ObjectDetection] = []
        for box, score, label in zip(
            prediction["boxes"].float().cpu().numpy(),
            prediction["scores"].float().cpu().numpy(),
            prediction["labels"].cpu().numpy(),
        ):
            if float(score) < 0.55:
                continue
            label_index = int(label)
            category = (
                self._categories[label_index]
                if 0 <= label_index < len(self._categories)
                else str(label_index)
            )
            detections.append(
                ObjectDetection(
                    xyxy=tuple(float(value) for value in box),
                    confidence=float(score),
                    label=category,
                )
            )
            if len(detections) >= 40:
                break
        return detections, {
            "backend": "torchvision_fasterrcnn_mobilenet_v3_320_fpn",
            "learned": True,
            "device": self._device,
            "detections": len(detections),
        }


def _clean_binary(probability: np.ndarray, threshold: float) -> np.ndarray:
    binary = (probability >= threshold).astype(np.uint8)
    size = max(3, int(round(min(binary.shape) / 160.0)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary


def _decimate_contour(contour: np.ndarray, max_points: int = 56) -> list[tuple[float, float]]:
    points = contour.reshape(-1, 2)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
    return [(float(x), float(y)) for x, y in points]


def _regions_from_probability(
    probability: np.ndarray,
    *,
    threshold: float,
    min_area_fraction: float,
    max_area_fraction: float,
    kind: str,
) -> list[RegionProposal]:
    height, width = probability.shape
    binary = _clean_binary(probability, threshold)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    image_area = float(height * width)
    regions: list[RegionProposal] = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        area_fraction = area / image_area
        if not min_area_fraction <= area_fraction <= max_area_fraction:
            continue
        component_mask = labels == component
        confidence = float(probability[component_mask].mean())
        if confidence < max(0.20, threshold * 0.72):
            continue
        mask_u8 = component_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        box_width = int(stats[component, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component, cv2.CC_STAT_HEIGHT])
        regions.append(
            RegionProposal(
                kind=kind,
                component_id=len(regions),
                confidence=_clamp01(confidence),
                area_fraction=area_fraction,
                bbox=(x, y, x + box_width, y + box_height),
                contour=_decimate_contour(contour),
                mask=component_mask,
            )
        )
    return sorted(regions, key=lambda region: region.confidence * region.area_fraction, reverse=True)


def _principal_endpoints(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 12:
        return None
    points = np.column_stack((xs, ys)).astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov(points - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    major = eigenvectors[:, order[-1]]
    projections = (points - center) @ major
    low, high = np.quantile(projections, [0.035, 0.965])
    low_points = points[projections <= low]
    high_points = points[projections >= high]
    first = low_points.mean(axis=0)
    second = high_points.mean(axis=0)
    length = float(np.linalg.norm(second - first))
    elongation = float(eigenvalues[order[-1]] / max(eigenvalues.sum(), 1e-9))
    return first, second, length, elongation


def _edge_density(edge: np.ndarray, point: np.ndarray, radius: int) -> float:
    x, y = int(round(point[0])), int(round(point[1]))
    y1, y2 = max(0, y - radius), min(edge.shape[0], y + radius + 1)
    x1, x2 = max(0, x - radius), min(edge.shape[1], x + radius + 1)
    patch = edge[y1:y2, x1:x2]
    return float(patch.mean()) if patch.size else 0.0


def _vertical_object_support(
    gray: np.ndarray,
    edge: np.ndarray,
    shadow_mask: np.ndarray,
    point: np.ndarray,
    radius: int,
) -> float:
    """Estimate whether foreground structure rises above a shadow endpoint.

    Cast-shadow contacts are commonly adjacent to an object whose appearance or
    outline continues upward in the image.  This is only a weak 2-D prior, so it
    contributes to proposal confidence but never determines consistency.
    Shadow-boundary pixels are removed before measuring support to avoid simply
    rewarding both ends of the same dark component.
    """

    x, y = int(round(point[0])), int(round(point[1]))
    x1, x2 = max(0, x - 2 * radius), min(gray.shape[1], x + 2 * radius + 1)
    y1, y2 = max(0, y - 6 * radius), min(gray.shape[0], y + radius + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    local_mask = shadow_mask[y1:y2, x1:x2]
    dilation_size = max(3, 2 * (radius // 3) + 1)
    dilated = cv2.dilate(
        local_mask.astype(np.uint8),
        np.ones((dilation_size, dilation_size), np.uint8),
        iterations=1,
    ).astype(bool)
    yy = np.arange(y1, y2)[:, None]
    above = yy <= y - max(1, radius // 2)
    valid = (~dilated) & above
    if int(valid.sum()) < 20:
        return 0.0
    edge_values = edge[y1:y2, x1:x2][valid]
    gray_values = gray[y1:y2, x1:x2][valid]
    edge_fraction = float((edge_values >= 0.18).mean())
    edge_strength = float(np.quantile(edge_values, 0.88))
    appearance_contrast = min(float(gray_values.std()) / 0.19, 1.0)
    return _clamp01(
        0.42 * min(edge_fraction / 0.08, 1.0)
        + 0.26 * edge_strength
        + 0.32 * appearance_contrast
    )


def _box_contact(
    endpoint_a: np.ndarray,
    endpoint_b: np.ndarray,
    detections: list[ObjectDetection],
    diagonal: float,
) -> tuple[np.ndarray, np.ndarray, float, ObjectDetection | None]:
    best: tuple[float, np.ndarray, np.ndarray, ObjectDetection] | None = None
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        for root, tip in ((endpoint_a, endpoint_b), (endpoint_b, endpoint_a)):
            contact = np.asarray((np.clip(root[0], x1, x2), y2), dtype=np.float64)
            distance = float(np.linalg.norm(contact - root))
            vertical_penalty = max(0.0, y1 - root[1]) / max(diagonal, 1.0)
            normalized = distance / max(diagonal, 1.0) + 0.5 * vertical_penalty
            score = normalized / max(detection.confidence, 0.1)
            if best is None or score < best[0]:
                best = (score, contact, tip, detection)
    if best is None or best[0] > 0.19:
        return endpoint_a, endpoint_b, 0.0, None
    association = _clamp01((0.22 - best[0]) / 0.22) * best[3].confidence
    return best[1], best[2], association, best[3]


def _shadow_proposals(
    image: Image.Image,
    probability: np.ndarray,
    detections: list[ObjectDetection],
    config: AutomaticProposalConfig,
    *,
    mask_metadata: dict[str, Any],
    object_metadata: dict[str, Any],
    regions: list[RegionProposal] | None = None,
) -> AutomaticCueProposals:
    if regions is None:
        regions = _regions_from_probability(
            probability,
            threshold=config.shadow_threshold,
            min_area_fraction=config.min_shadow_area_fraction,
            max_area_fraction=config.max_shadow_area_fraction,
            kind="shadow",
        )
    gray = cv2.cvtColor(_as_rgb_array(image), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 1.0)
    edge_scale = max(float(np.quantile(edge, 0.97)), 1e-5)
    edge = np.clip(edge / edge_scale, 0.0, 1.0)
    diagonal = math.hypot(*image.size)
    radius = max(3, int(round(diagonal * 0.015)))
    pairs: list[ShadowPair] = []
    pair_evidence: list[dict[str, Any]] = []

    for region in regions[: config.max_shadow_pairs]:
        endpoints = _principal_endpoints(region.mask)
        if endpoints is None:
            continue
        first, second, length, elongation = endpoints
        if length < max(12.0, diagonal * 0.025) or elongation < 0.57:
            continue
        contact, tip, box_association, detection = _box_contact(
            first, second, detections, diagonal
        )
        if detection is None:
            first_support = _vertical_object_support(
                gray, edge, region.mask, first, radius
            )
            second_support = _vertical_object_support(
                gray, edge, region.mask, second, radius
            )
            first_edge = _edge_density(edge, first, radius)
            second_edge = _edge_density(edge, second, radius)
            first_combined = 0.78 * first_support + 0.22 * first_edge
            second_combined = 0.78 * second_support + 0.22 * second_edge
            # Weak ground-plane image prior: for ordinary upright-camera
            # scenes, a contact tends to project above the distal shadow tip.
            # It is intentionally too small to rescue unsupported endpoints.
            if first[1] <= second[1]:
                first_combined = _clamp01(first_combined + 0.12)
            else:
                second_combined = _clamp01(second_combined + 0.12)
            if second_combined > first_combined:
                contact, tip = second, first
                contact_support, tip_support = second_combined, first_combined
            else:
                contact, tip = first, second
                contact_support, tip_support = first_combined, second_combined
            support_margin = max(contact_support - tip_support, 0.0)
            association = _clamp01(
                0.16
                + 0.54 * contact_support
                + 0.30 * min(support_margin / 0.45, 1.0)
            )
            association_kind = "foreground_support_above_endpoint"
            object_label = None
        else:
            association = _clamp01(0.55 + 0.45 * box_association)
            association_kind = "object_box_to_shadow_component"
            object_label = detection.label
            contact_support = association
            tip_support = None
        length_score = _clamp01(length / max(diagonal * 0.16, 1.0))
        elongation_score = _clamp01((elongation - 0.5) / 0.42)
        confidence = _clamp01(
            region.confidence
            * (0.42 + 0.28 * length_score + 0.30 * elongation_score)
            * association
        )
        if confidence < config.min_shadow_pair_confidence:
            continue
        pair = ShadowPair(
            object_contact=(float(contact[0]), float(contact[1])),
            shadow_tip=(float(tip[0]), float(tip[1])),
            confidence=confidence,
        )
        pairs.append(pair)
        pair_evidence.append(
            {
                "kind": "shadow_pair_proposal",
                "component_id": region.component_id,
                "object_contact": [round(float(contact[0]), 3), round(float(contact[1]), 3)],
                "shadow_tip": [round(float(tip[0]), 3), round(float(tip[1]), 3)],
                "confidence": round(confidence, 6),
                "association": association_kind,
                "association_confidence": round(association, 6),
                "contact_support": round(float(contact_support), 6),
                "alternate_endpoint_support": (
                    round(float(tip_support), 6) if tip_support is not None else None
                ),
                "object_label": object_label,
                "role": "proposal",
            }
        )

    confidence = float(np.mean([pair.confidence for pair in pairs])) if pairs else 0.0
    applicable = len(pairs) >= config.min_pairs
    reason = (
        f"Generated {len(pairs)} high-confidence object-contact/shadow-tip pair(s)."
        if applicable
        else (
            f"Generated {len(pairs)} high-confidence shadow pair(s); "
            f"at least {config.min_pairs} are required."
        )
    )
    return AutomaticCueProposals(
        cue="cast_shadow",
        pairs=pairs,
        applicable=applicable,
        confidence=_clamp01(confidence),
        reason=reason,
        evidence=[region.evidence() for region in regions] + pair_evidence,
        measurements={
            "proposal_origin": "automatic",
            "candidate_shadow_regions": len(regions),
            "accepted_shadow_pairs": len(pairs),
            "required_pairs": config.min_pairs,
            "mask_backend": mask_metadata,
            "object_backend": object_metadata,
        },
        limitations=[
            "Automatic shadow masks can confuse dark material, self-shadow, and cast shadow.",
            "Object-ground contacts derived from masks, edges, or generic object boxes are proposals rather than observed 3-D contacts.",
            "Soft shadows, multiple lights, uneven terrain, and merged shadow components should abstain or be reviewed.",
        ],
    )


def _feature_saliency(values: np.ndarray) -> np.ndarray:
    flattened = values.reshape(-1, values.shape[-1]).astype(np.float64)
    center = np.median(flattened, axis=0, keepdims=True)
    distances = np.linalg.norm(flattened - center, axis=1)
    scale = max(
        float(np.quantile(distances, 0.90)),
        0.10 * float(distances.max(initial=0.0)),
    )
    if scale <= 1e-9:
        return np.zeros(values.shape[:2], dtype=np.float64)
    return np.clip(distances.reshape(values.shape[:2]) / scale, 0.0, 1.0)


def _normalize_features(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or min(array.shape, default=0) <= 0:
        raise AutomaticProposalError("Dense features must have shape (rows, columns, channels)")
    if not np.isfinite(array).all():
        raise AutomaticProposalError("Dense features must contain only finite values")
    norms = np.linalg.norm(array, axis=2, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def _mutual_feature_matches(
    features: DenseFeatures,
    mirror_mask: np.ndarray,
    mirror_confidence: float,
    image_size: tuple[int, int],
    config: AutomaticProposalConfig,
) -> list[tuple[ReflectionPair, dict[str, Any]]]:
    values = _normalize_features(features.values)
    rows, columns = values.shape[:2]
    grid_mask = cv2.resize(
        mirror_mask.astype(np.uint8), (columns, rows), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    kernel = np.ones((3, 3), np.uint8)
    inside = cv2.erode(grid_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    outside = ~cv2.dilate(grid_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    saliency = _feature_saliency(features.values)
    inside &= saliency >= 0.16
    outside &= saliency >= 0.16
    inside_indices = np.argwhere(inside)
    outside_indices = np.argwhere(outside)
    if len(inside_indices) < 3 or len(outside_indices) < 3:
        return []

    inside_features = values[inside]
    outside_features = values[outside]
    similarity = inside_features @ outside_features.T
    best_outside = np.argmax(similarity, axis=1)
    best_inside = np.argmax(similarity, axis=0)
    sorted_similarity = np.sort(similarity, axis=1)
    best_values = sorted_similarity[:, -1]
    second_values = sorted_similarity[:, -2] if similarity.shape[1] > 1 else np.full(len(inside_indices), -1.0)
    margins = best_values - second_values
    image_width, image_height = image_size
    candidates: list[tuple[float, ReflectionPair, dict[str, Any]]] = []
    for inside_offset, outside_offset in enumerate(best_outside):
        if int(best_inside[outside_offset]) != inside_offset:
            continue
        similarity_value = float(best_values[inside_offset])
        margin = float(margins[inside_offset])
        if similarity_value < config.min_feature_similarity or margin < config.min_feature_margin:
            continue
        inside_row, inside_column = inside_indices[inside_offset]
        outside_row, outside_column = outside_indices[outside_offset]
        dx = (inside_column - outside_column) / columns
        dy = (inside_row - outside_row) / rows
        separation = math.hypot(dx, dy)
        if separation < 0.08:
            continue
        inside_saliency = float(saliency[inside_row, inside_column])
        outside_saliency = float(saliency[outside_row, outside_column])
        match_quality = _clamp01(
            0.42 * ((similarity_value + 1.0) / 2.0)
            + 0.26 * min(margin / 0.12, 1.0)
            + 0.20 * min(inside_saliency, outside_saliency)
            + 0.12 * min(separation / 0.35, 1.0)
        )
        confidence = _clamp01(mirror_confidence * match_quality)
        if confidence < config.min_reflection_pair_confidence:
            continue
        object_point = (
            (float(outside_column) + 0.5) * image_width / columns,
            (float(outside_row) + 0.5) * image_height / rows,
        )
        reflection_point = (
            (float(inside_column) + 0.5) * image_width / columns,
            (float(inside_row) + 0.5) * image_height / rows,
        )
        pair = ReflectionPair(
            object_point=object_point,
            reflection_point=reflection_point,
            confidence=confidence,
        )
        evidence = {
            "kind": "reflection_pair_proposal",
            "object_point": [round(value, 3) for value in object_point],
            "reflection_point": [round(value, 3) for value in reflection_point],
            "confidence": round(confidence, 6),
            "feature_similarity": round(similarity_value, 6),
            "nearest_neighbour_margin": round(margin, 6),
            "mutual_nearest_neighbour": True,
            "role": "proposal",
        }
        candidates.append((confidence, pair, evidence))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[ReflectionPair, dict[str, Any]]] = []
    selected_grid_points: list[tuple[float, float, float, float]] = []
    minimum_grid_distance = 1.35
    for _confidence, pair, evidence in candidates:
        current = (
            pair.object_point[0] * columns / image_width,
            pair.object_point[1] * rows / image_height,
            pair.reflection_point[0] * columns / image_width,
            pair.reflection_point[1] * rows / image_height,
        )
        if any(
            math.hypot(current[0] - prior[0], current[1] - prior[1]) < minimum_grid_distance
            or math.hypot(current[2] - prior[2], current[3] - prior[3]) < minimum_grid_distance
            for prior in selected_grid_points
        ):
            continue
        selected.append((pair, evidence))
        selected_grid_points.append(current)
        if len(selected) >= config.max_reflection_pairs:
            break
    return selected


def _reflection_proposals(
    image: Image.Image,
    probability: np.ndarray,
    features: DenseFeatures | None,
    config: AutomaticProposalConfig,
    *,
    mask_metadata: dict[str, Any],
    regions: list[RegionProposal] | None = None,
    unavailable_feature_metadata: dict[str, Any] | None = None,
) -> AutomaticCueProposals:
    if regions is None:
        regions = _regions_from_probability(
            probability,
            threshold=config.mirror_threshold,
            min_area_fraction=config.min_mirror_area_fraction,
            max_area_fraction=config.max_mirror_area_fraction,
            kind="mirror",
        )
    selected: list[tuple[ReflectionPair, dict[str, Any]]] = []
    if features is not None:
        for region in regions:
            region_matches = _mutual_feature_matches(
                features,
                region.mask,
                region.confidence,
                image.size,
                config,
            )
            for pair, evidence in region_matches:
                evidence["component_id"] = region.component_id
                selected.append((pair, evidence))
        selected.sort(key=lambda item: item[0].confidence, reverse=True)
        selected = selected[: config.max_reflection_pairs]

    pairs = [pair for pair, _ in selected]
    confidence = float(np.mean([pair.confidence for pair in pairs])) if pairs else 0.0
    applicable = len(pairs) >= config.min_pairs
    if not regions:
        reason = "No sufficiently confident planar-reflector region was proposed."
    elif features is None:
        reason = "A reflector was proposed, but no dense correspondence feature backend was available."
    elif not applicable:
        reason = (
            f"Generated {len(pairs)} high-confidence direct/reflected match(es); "
            f"at least {config.min_pairs} are required."
        )
    else:
        reason = f"Generated {len(pairs)} high-confidence direct/reflected match(es)."
    feature_metadata = (
        {"backend": features.backend, "model": features.model, **features.metadata}
        if features is not None
        else (unavailable_feature_metadata or {"backend": "none"})
    )
    return AutomaticCueProposals(
        cue="reflection",
        pairs=pairs,
        applicable=applicable,
        confidence=_clamp01(confidence),
        reason=reason,
        evidence=[region.evidence() for region in regions]
        + [evidence for _, evidence in selected],
        measurements={
            "proposal_origin": "automatic",
            "candidate_mirror_regions": len(regions),
            "accepted_reflection_pairs": len(pairs),
            "required_pairs": config.min_pairs,
            "mask_backend": mask_metadata,
            "feature_backend": feature_metadata,
        },
        limitations=[
            "Mirror segmentation can confuse windows, framed pictures, openings, and glossy surfaces.",
            "Feature matching is possible only when corresponding content is visible both directly and inside the reflector.",
            "Curved, rippled, translucent, or multiple reflecting surfaces invalidate the planar test.",
            "Feature matches are proposed independently of the final geometry fit to avoid circular verification.",
        ],
    )


class AutomaticProposalEngine:
    """Reusable proposal runtime loaded once for a directory-scale analysis."""

    def __init__(
        self,
        config: AutomaticProposalConfig | None = None,
        *,
        mask_provider: MaskProvider | None = None,
        feature_provider: FeatureProvider | None = None,
        object_provider: ObjectProvider | None = None,
    ) -> None:
        self.config = config or AutomaticProposalConfig()
        self._fallback_masks = HeuristicMaskProvider()
        self.mask_provider = mask_provider or self._build_mask_provider()
        self.feature_provider = feature_provider or self._build_feature_provider()
        self.object_provider = object_provider or self._build_object_provider()

    def _build_mask_provider(self) -> MaskProvider:
        if self.config.mask_backend == "clipseg":
            return ClipSegMaskProvider(self.config)
        return self._fallback_masks

    def _build_feature_provider(self) -> FeatureProvider | None:
        if self.config.feature_backend == "appearance":
            return DenseAppearanceFeatureProvider(self.config.appearance_grid_size)
        if self.config.feature_backend == "dinov3":
            return TimmDinoFeatureProvider(self.config)
        return None

    def _build_object_provider(self) -> ObjectProvider:
        if self.config.object_backend == "torchvision":
            return TorchvisionObjectProvider(self.config)
        if self.config.object_backend == "none":
            return NullObjectProvider()
        return EdgeObjectProvider()

    def _masks(self, image: Image.Image) -> SemanticMasks:
        try:
            return self.mask_provider.predict(image)
        except Exception as exc:
            if not self.config.allow_model_fallback or isinstance(
                self.mask_provider, HeuristicMaskProvider
            ):
                if isinstance(exc, AutomaticProposalError):
                    raise
                raise AutomaticProposalError(f"Mask proposal failed: {exc}") from exc
            fallback = self._fallback_masks.predict(image)
            fallback.warnings.append(
                f"Requested mask backend failed and the heuristic fallback was used: {exc}"
            )
            fallback.metadata["requested_backend"] = self.config.mask_backend
            fallback.metadata["fallback_reason"] = str(exc)
            return fallback

    def _objects(self, image: Image.Image) -> tuple[list[ObjectDetection], dict[str, Any], list[str]]:
        try:
            detections, metadata = self.object_provider.detect(image)
            return detections, metadata, []
        except Exception as exc:
            if not self.config.allow_model_fallback:
                raise AutomaticProposalError(f"Object proposal failed: {exc}") from exc
            return [], {
                "backend": "local_edge_contact",
                "requested_backend": self.config.object_backend,
                "fallback_reason": str(exc),
                "learned": False,
            }, [f"Object detector failed; endpoint-edge fallback used: {exc}"]

    def _features(
        self, image: Image.Image, external_features: object | None
    ) -> tuple[DenseFeatures | None, list[str]]:
        if external_features is not None:
            backend = "shared_patchhead_dinov3_tokens"
            model = None
            metadata: dict[str, Any] = {}
            values_source = external_features
            if isinstance(external_features, Mapping):
                if "values" not in external_features:
                    raise AutomaticProposalError(
                        "An external dense-feature payload must contain `values`"
                    )
                values_source = external_features["values"]
                supplied_backend = external_features.get("backend")
                supplied_model = external_features.get("model")
                supplied_metadata = external_features.get("metadata", {})
                if supplied_backend is not None:
                    if not isinstance(supplied_backend, str) or not supplied_backend:
                        raise AutomaticProposalError(
                            "External dense-feature backend must be a non-empty string"
                        )
                    backend = supplied_backend
                if supplied_model is not None:
                    if not isinstance(supplied_model, str) or not supplied_model:
                        raise AutomaticProposalError(
                            "External dense-feature model must be a non-empty string"
                        )
                    model = supplied_model
                if not isinstance(supplied_metadata, Mapping):
                    raise AutomaticProposalError(
                        "External dense-feature metadata must be a mapping"
                    )
                metadata = dict(supplied_metadata)
            values = np.asarray(values_source, dtype=np.float32)
            if (
                values.ndim != 3
                or min(values.shape, default=0) <= 0
                or not np.isfinite(values).all()
            ):
                raise AutomaticProposalError(
                    "External dense features must have finite shape "
                    "(rows, columns, channels)"
                )
            metadata.update(
                {
                    "learned": True,
                    "grid_shape": list(values.shape[:2]),
                    "coordinate_space": "normalized_full_frame",
                    "shared_primary_forward": True,
                }
            )
            return DenseFeatures(
                values=values,
                backend=backend,
                model=model,
                metadata=metadata,
            ), []
        if self.config.feature_backend == "external":
            return None, ["No external dense feature grid was supplied for this image."]
        if self.feature_provider is None:
            return None, []
        try:
            return self.feature_provider.extract(image), []
        except Exception as exc:
            if not self.config.allow_model_fallback or isinstance(
                self.feature_provider, DenseAppearanceFeatureProvider
            ):
                if isinstance(exc, AutomaticProposalError):
                    raise
                raise AutomaticProposalError(f"Feature extraction failed: {exc}") from exc
            fallback = DenseAppearanceFeatureProvider(
                self.config.appearance_grid_size
            ).extract(image)
            fallback.metadata["requested_backend"] = self.config.feature_backend
            fallback.metadata["fallback_reason"] = str(exc)
            return fallback, [f"DINO feature extraction failed; appearance fallback used: {exc}"]

    def propose(
        self,
        image: Image.Image,
        *,
        external_features: object | None = None,
        include_shadow: bool = True,
        include_reflection: bool = True,
    ) -> AutomaticProposalBundle:
        if not self.config.enabled:
            raise AutomaticProposalError("Automatic proposals are not enabled")
        if not include_shadow and not include_reflection:
            raise AutomaticProposalError("At least one automatic cue must be requested")
        masks = self._masks(image)
        shape = (image.height, image.width)
        shadow_probability = _normalize_probability(masks.shadow, shape)
        mirror_probability = _normalize_probability(masks.mirror, shape)
        shadow_regions = (
            _regions_from_probability(
                shadow_probability,
                threshold=self.config.shadow_threshold,
                min_area_fraction=self.config.min_shadow_area_fraction,
                max_area_fraction=self.config.max_shadow_area_fraction,
                kind="shadow",
            )
            if include_shadow
            else []
        )
        mirror_regions = (
            _regions_from_probability(
                mirror_probability,
                threshold=self.config.mirror_threshold,
                min_area_fraction=self.config.min_mirror_area_fraction,
                max_area_fraction=self.config.max_mirror_area_fraction,
                kind="mirror",
            )
            if include_reflection
            else []
        )
        if not include_shadow:
            detections, object_warnings = [], []
            object_metadata = {
                "backend": "skipped_reviewed_shadow_evidence",
                "requested_backend": self.config.object_backend,
                "learned": False,
            }
        elif shadow_regions:
            detections, object_metadata, object_warnings = self._objects(image)
        else:
            detections, object_warnings = [], []
            object_metadata = {
                "backend": "skipped_no_shadow_region",
                "requested_backend": self.config.object_backend,
                "learned": False,
            }
        if not include_reflection:
            features, feature_warnings = None, []
            unavailable_feature_metadata = {
                "backend": "skipped_reviewed_reflection_evidence",
                "requested_backend": self.config.feature_backend,
                "learned": False,
            }
        elif mirror_regions:
            features, feature_warnings = self._features(image, external_features)
            unavailable_feature_metadata = None
        else:
            features, feature_warnings = None, []
            unavailable_feature_metadata = {
                "backend": "skipped_no_mirror_region",
                "requested_backend": self.config.feature_backend,
                "learned": False,
            }
        mask_metadata = {
            "backend": masks.backend,
            "model": masks.model,
            **masks.metadata,
        }
        shadow = _shadow_proposals(
            image,
            shadow_probability,
            detections,
            self.config,
            mask_metadata=mask_metadata,
            object_metadata=object_metadata,
            regions=shadow_regions,
        )
        reflection = _reflection_proposals(
            image,
            mirror_probability,
            features,
            self.config,
            mask_metadata=mask_metadata,
            regions=mirror_regions,
            unavailable_feature_metadata=unavailable_feature_metadata,
        )
        if (
            include_reflection
            and external_features is not None
            and mirror_regions
            and not reflection.applicable
            and self.config.appearance_fallback_on_insufficient_external
            and self.feature_provider is not None
        ):
            primary_feature_metadata = dict(
                reflection.measurements.get("feature_backend", {})
            )
            fallback_features, fallback_warnings = self._features(image, None)
            fallback_reflection = _reflection_proposals(
                image,
                mirror_probability,
                fallback_features,
                self.config,
                mask_metadata=mask_metadata,
                regions=mirror_regions,
            )
            if len(fallback_reflection.pairs) > len(reflection.pairs):
                fallback_reflection.measurements["feature_selection"] = {
                    "policy": "appearance_after_insufficient_external_correspondences",
                    "primary_backend": primary_feature_metadata,
                    "primary_accepted_pairs": len(reflection.pairs),
                    "fallback_accepted_pairs": len(fallback_reflection.pairs),
                    "minimum_required_pairs": self.config.min_pairs,
                }
                fallback_reflection.warnings.append(
                    "Shared DINO produced too few reflection correspondences; "
                    "the local appearance fallback supplied the displayed proposal set."
                )
                fallback_reflection.limitations.append(
                    "Appearance fallback can match repeated colour or texture and must "
                    "remain subject to the independent geometry and four-pair inconsistency gate."
                )
                reflection = fallback_reflection
            feature_warnings.extend(fallback_warnings)
        if not include_shadow:
            shadow.reason = "Automatic shadow proposals were skipped because reviewed evidence is present."
        if not include_reflection:
            reflection.reason = (
                "Automatic reflection proposals were skipped because reviewed evidence is present."
            )
        warnings = list(masks.warnings) + object_warnings + feature_warnings
        shadow.warnings.extend(warnings)
        reflection.warnings.extend(warnings)
        return AutomaticProposalBundle(
            shadow=shadow,
            reflection=reflection,
            backend={
                "mask": mask_metadata,
                "object": object_metadata,
                "feature": reflection.measurements["feature_backend"],
            },
            warnings=warnings,
        )


__all__ = [
    "AutomaticCueProposals",
    "AutomaticProposalBundle",
    "AutomaticProposalConfig",
    "AutomaticProposalEngine",
    "AutomaticProposalError",
    "ClipSegMaskProvider",
    "DenseAppearanceFeatureProvider",
    "DenseFeatures",
    "FeatureProvider",
    "HeuristicMaskProvider",
    "MaskProvider",
    "NullObjectProvider",
    "ObjectDetection",
    "ObjectProvider",
    "RegionProposal",
    "SemanticMasks",
    "TimmDinoFeatureProvider",
    "TorchvisionObjectProvider",
]
