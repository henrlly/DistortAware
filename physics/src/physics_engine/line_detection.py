"""Pixel-to-line extraction using OpenCV's line segment detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .geometry import Segment, segment_length


Region = tuple[float, float, float, float]


@dataclass(slots=True)
class LineDetectionConfig:
    min_length_ratio: float = 0.045
    max_lines: int = 300
    denoise_sigma: float = 0.8
    enhance_contrast: bool = True


@dataclass(slots=True)
class LineDetectionResult:
    segments: list[Segment]
    diagnostics: dict[str, Any]


def _load_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in misconfigured installs
        raise RuntimeError(
            "OpenCV is required for automatic line extraction. Install the project "
            "with `python -m pip install -e .`."
        ) from exc
    return cv2


def _orientation_statistics(segments: list[Segment]) -> dict[str, float]:
    if not segments:
        return {"orientation_peak_fraction": 0.0, "orientation_entropy": 0.0}

    angles: list[float] = []
    weights: list[float] = []
    for x1, y1, x2, y2 in segments:
        angles.append(float(np.arctan2(y2 - y1, x2 - x1) % np.pi))
        weights.append(segment_length((x1, y1, x2, y2)))

    histogram, _ = np.histogram(
        np.asarray(angles), bins=18, range=(0.0, np.pi), weights=np.asarray(weights)
    )
    total = float(np.sum(histogram))
    if total <= 1e-12:
        return {"orientation_peak_fraction": 0.0, "orientation_entropy": 0.0}
    probabilities = histogram / total
    nonzero = probabilities[probabilities > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero))) / float(np.log(len(histogram)))
    top_bins = np.sort(histogram)[-4:]
    return {
        "orientation_peak_fraction": float(np.sum(top_bins) / total),
        "orientation_entropy": entropy,
    }


def detect_line_segments(
    image: Image.Image,
    config: LineDetectionConfig | None = None,
    regions: list[Region] | None = None,
) -> LineDetectionResult:
    config = config or LineDetectionConfig()
    cv2 = _load_cv2()

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if config.denoise_sigma > 0:
        # Stabilise long edges under JPEG ringing and sensor-like noise before
        # local contrast enhancement, which would otherwise amplify both.
        gray = cv2.GaussianBlur(
            gray,
            (0, 0),
            sigmaX=config.denoise_sigma,
            sigmaY=config.denoise_sigma,
        )
    if config.enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(gray)
    raw_lines = detected[0] if detected is not None else None
    height, width = gray.shape
    diagonal = float(np.hypot(width, height))
    minimum_length = max(8.0, config.min_length_ratio * diagonal)

    segments: list[Segment] = []
    if raw_lines is not None:
        for raw_line in raw_lines.reshape(-1, 4):
            segment = tuple(float(value) for value in raw_line)
            if segment_length(segment) >= minimum_length:
                segments.append(segment)

    count_before_region_filter = len(segments)
    if regions is not None:
        def selected(segment: Segment) -> bool:
            x1, y1, x2, y2 = segment
            midpoint = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            for left, top, right, bottom in regions:
                def inside(point: tuple[float, float]) -> bool:
                    return (
                        left <= point[0] <= right and top <= point[1] <= bottom
                    )

                if inside(midpoint) and (inside((x1, y1)) or inside((x2, y2))):
                    return True
            return False

        segments = [segment for segment in segments if selected(segment)]

    segments.sort(key=segment_length, reverse=True)
    raw_filtered_count = len(segments)
    segments = segments[: config.max_lines]
    total_length = float(sum(segment_length(segment) for segment in segments))

    diagnostics: dict[str, Any] = {
        "detector": "opencv_lsd",
        "denoise_sigma": config.denoise_sigma,
        "raw_detected_count": 0 if raw_lines is None else int(len(raw_lines)),
        "reviewed_region_filter_applied": regions is not None,
        "reviewed_region_count": len(regions or []),
        "long_line_count_before_region_filter": count_before_region_filter,
        "minimum_length_pixels": minimum_length,
        "filtered_count_before_cap": raw_filtered_count,
        "retained_count": len(segments),
        "total_length_pixels": total_length,
        "total_length_diagonals": total_length / max(diagonal, 1e-12),
    }
    diagnostics.update(_orientation_statistics(segments))
    return LineDetectionResult(segments=segments, diagnostics=diagnostics)
