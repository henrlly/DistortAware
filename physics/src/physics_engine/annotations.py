"""Loading and validating optional reviewed physics annotations."""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np


Point = tuple[float, float]
Region = tuple[float, float, float, float]
REVIEW_APPLICABILITY = {"applicable", "not_applicable", "uncertain", "unreviewed"}


@dataclass(slots=True)
class ShadowPair:
    object_contact: Point
    shadow_tip: Point
    confidence: float = 1.0


@dataclass(slots=True)
class ReflectionPair:
    object_point: Point
    reflection_point: Point
    confidence: float = 1.0


@dataclass(slots=True)
class PerspectiveRegion:
    xyxy: Region
    confidence: float = 1.0


@dataclass(slots=True)
class ImageAnnotations:
    perspective_regions: list[PerspectiveRegion] = field(default_factory=list)
    shadow_pairs: list[ShadowPair] = field(default_factory=list)
    reflection_pairs: list[ReflectionPair] = field(default_factory=list)
    shadow_applicability: str | None = None
    reflection_applicability: str | None = None
    source_key: str | None = None


def _point(value: Any, *, width: int, height: int, normalized: bool) -> Point:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Every point must be a two-element JSON array")
    x, y = float(value[0]), float(value[1])
    if not np.isfinite([x, y]).all():
        raise ValueError("Point coordinates must be finite")
    if normalized:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("Normalized points must lie within [0, 1]")
        return x * width, y * height
    if not (0.0 <= x <= width and 0.0 <= y <= height):
        raise ValueError("Pixel points must lie within the image bounds")
    return x, y


def _confidence(value: Any) -> float:
    confidence = float(1.0 if value is None else value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Annotation confidence must lie within [0, 1]")
    return confidence


def _region(value: Any, *, width: int, height: int, normalized: bool) -> Region:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Every perspective region must be an xyxy four-element array")
    x1, y1 = _point(value[:2], width=width, height=height, normalized=normalized)
    x2, y2 = _point(value[2:], width=width, height=height, normalized=normalized)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Perspective regions must have positive width and height")
    return x1, y1, x2, y2


def _review_applicability(section: dict[str, Any], cue: str) -> str | None:
    value = section.get("applicability")
    if value is None:
        return None
    if value not in REVIEW_APPLICABILITY:
        raise ValueError(
            f"{cue}.applicability must be one of {sorted(REVIEW_APPLICABILITY)}"
        )
    return str(value)


class AnnotationStore:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        images = payload.get("images", {})
        if not isinstance(images, dict):
            raise ValueError("Annotation `images` must be a JSON object")
        self._images: dict[str, dict[str, Any]] = images
        self._default_coordinate_space = payload.get("coordinate_space", "normalized")
        if self._default_coordinate_space not in {"normalized", "pixels"}:
            raise ValueError("coordinate_space must be `normalized` or `pixels`")

    @classmethod
    def from_path(cls, path: str | Path | None) -> "AnnotationStore":
        if path is None:
            return cls()
        annotation_path = Path(path)
        if annotation_path.is_file():
            return cls(cls._load_payload(annotation_path))
        if not annotation_path.is_dir():
            raise FileNotFoundError(f"Annotation path does not exist: {annotation_path}")

        annotation_files = sorted(annotation_path.glob("*.json"))
        if not annotation_files:
            raise ValueError(f"No JSON annotation files were found in {annotation_path}")

        merged_images: dict[str, dict[str, Any]] = {}
        owners: dict[str, Path] = {}
        for annotation_file in annotation_files:
            payload = cls._load_payload(annotation_file)
            default_space = payload.get("coordinate_space", "normalized")
            if default_space not in {"normalized", "pixels"}:
                raise ValueError(
                    f"Invalid coordinate_space in annotation file {annotation_file}"
                )
            images = payload.get("images", {})
            if not isinstance(images, dict):
                raise ValueError(
                    f"Annotation `images` must be an object in {annotation_file}"
                )
            for image_key, raw_entry in images.items():
                if image_key in merged_images:
                    raise ValueError(
                        f"Duplicate annotation key {image_key!r} in {owners[image_key]} "
                        f"and {annotation_file}"
                    )
                if not isinstance(raw_entry, dict):
                    raise ValueError(
                        f"Annotation for {image_key!r} in {annotation_file} must be an object"
                    )
                entry = deepcopy(raw_entry)
                entry.setdefault("coordinate_space", default_space)
                merged_images[image_key] = entry
                owners[image_key] = annotation_file
        return cls({"coordinate_space": "normalized", "images": merged_images})

    @staticmethod
    def _load_payload(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"The annotation file root must be a JSON object: {path}")
        return payload

    def _matching_entry(
        self, image_path: Path, input_root: Path
    ) -> tuple[str | None, dict[str, Any] | None]:
        candidates: list[str] = []
        if input_root.is_dir():
            try:
                candidates.append(image_path.relative_to(input_root).as_posix())
            except ValueError:
                pass
        candidates.extend([image_path.name, image_path.resolve().as_posix()])
        for candidate in candidates:
            if candidate in self._images:
                return candidate, self._images[candidate]
        return None, None

    def for_image(
        self, image_path: str | Path, input_root: str | Path, width: int, height: int
    ) -> ImageAnnotations:
        path = Path(image_path)
        root = Path(input_root)
        source_key, entry = self._matching_entry(path, root)
        if entry is None:
            return ImageAnnotations()
        if not isinstance(entry, dict):
            raise ValueError(f"Annotation for {source_key} must be a JSON object")

        coordinate_space = entry.get("coordinate_space", self._default_coordinate_space)
        if coordinate_space not in {"normalized", "pixels"}:
            raise ValueError(f"Invalid coordinate_space for {source_key}")
        normalized = coordinate_space == "normalized"

        perspective_regions: list[PerspectiveRegion] = []
        perspective_section = entry.get("perspective", {})
        if perspective_section is not None:
            if not isinstance(perspective_section, dict):
                raise ValueError(f"perspective for {source_key} must be an object")
            raw_regions = perspective_section.get("regions", [])
            if not isinstance(raw_regions, list):
                raise ValueError(f"perspective regions for {source_key} must be an array")
            for region in raw_regions:
                if not isinstance(region, dict):
                    raise ValueError("Each perspective region must be an object")
                perspective_regions.append(
                    PerspectiveRegion(
                        xyxy=_region(
                            region.get("xyxy"),
                            width=width,
                            height=height,
                            normalized=normalized,
                        ),
                        confidence=_confidence(region.get("confidence")),
                    )
                )

        shadow_pairs: list[ShadowPair] = []
        shadow_applicability: str | None = None
        shadow_section = entry.get("cast_shadow", {})
        if shadow_section is not None:
            if not isinstance(shadow_section, dict):
                raise ValueError(f"cast_shadow for {source_key} must be an object")
            shadow_applicability = _review_applicability(
                shadow_section, "cast_shadow"
            )
            raw_pairs = shadow_section.get("pairs", [])
            if not isinstance(raw_pairs, list):
                raise ValueError(f"cast_shadow pairs for {source_key} must be an array")
            for pair in raw_pairs:
                if not isinstance(pair, dict):
                    raise ValueError("Each cast-shadow pair must be an object")
                shadow_pairs.append(
                    ShadowPair(
                        object_contact=_point(
                            pair.get("object_contact"),
                            width=width,
                            height=height,
                            normalized=normalized,
                        ),
                        shadow_tip=_point(
                            pair.get("shadow_tip"),
                            width=width,
                            height=height,
                            normalized=normalized,
                        ),
                        confidence=_confidence(pair.get("confidence")),
                    )
                )

        reflection_pairs: list[ReflectionPair] = []
        reflection_applicability: str | None = None
        reflection_section = entry.get("reflection", {})
        if reflection_section is not None:
            if not isinstance(reflection_section, dict):
                raise ValueError(f"reflection for {source_key} must be an object")
            reflection_applicability = _review_applicability(
                reflection_section, "reflection"
            )
            raw_pairs = reflection_section.get("pairs", [])
            if not isinstance(raw_pairs, list):
                raise ValueError(f"reflection pairs for {source_key} must be an array")
            for pair in raw_pairs:
                if not isinstance(pair, dict):
                    raise ValueError("Each reflection pair must be an object")
                reflection_pairs.append(
                    ReflectionPair(
                        object_point=_point(
                            pair.get("object_point"),
                            width=width,
                            height=height,
                            normalized=normalized,
                        ),
                        reflection_point=_point(
                            pair.get("reflection_point"),
                            width=width,
                            height=height,
                            normalized=normalized,
                        ),
                        confidence=_confidence(pair.get("confidence")),
                    )
                )

        return ImageAnnotations(
            perspective_regions=perspective_regions,
            shadow_pairs=shadow_pairs,
            reflection_pairs=reflection_pairs,
            shadow_applicability=shadow_applicability,
            reflection_applicability=reflection_applicability,
            source_key=source_key,
        )
