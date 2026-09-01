"""Spatial association helpers for detector maps and physics evidence.

The functions in this module deliberately measure *association*, not causal
attribution.  A high detector score near a physics outlier does not establish
that the detector used that physical cue, and it does not turn a physics
violation score into an AIGC probability.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


class SpatialEvidenceError(ValueError):
    """Raised when a spatial evidence map or geometry payload is malformed."""


def validate_score_grid(values: Any) -> np.ndarray:
    """Return a finite 2-D score grid with values constrained to ``[0, 1]``."""

    try:
        grid = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SpatialEvidenceError("Patch evidence values must be a numeric 2-D array") from exc
    if grid.ndim != 2 or min(grid.shape, default=0) <= 0:
        raise SpatialEvidenceError("Patch evidence values must be a non-empty 2-D array")
    if not np.isfinite(grid).all():
        raise SpatialEvidenceError("Patch evidence values must all be finite")
    if float(grid.min()) < 0.0 or float(grid.max()) > 1.0:
        raise SpatialEvidenceError("Patch evidence values must lie within [0, 1]")
    return grid


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def segment_from_evidence(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Extract a pixel-space segment from one supported physics evidence item."""

    kind = item.get("kind")
    if kind == "line_segment":
        raw = item.get("xyxy")
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            return None
        try:
            segment = tuple(float(value) for value in raw)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in segment):
            return None
        return segment  # type: ignore[return-value]
    if kind == "shadow_vector":
        first = _point(item.get("object_contact"))
        second = _point(item.get("shadow_tip"))
    elif kind == "reflection_connector":
        first = _point(item.get("object_point"))
        second = _point(item.get("reflection_point"))
    else:
        return None
    if first is None or second is None:
        return None
    return first[0], first[1], second[0], second[1]


def _distance_to_segment(
    x: np.ndarray,
    y: np.ndarray,
    segment: tuple[float, float, float, float],
) -> np.ndarray:
    x1, y1, x2, y2 = segment
    dx, dy = x2 - x1, y2 - y1
    denominator = dx * dx + dy * dy
    if denominator <= 1e-16:
        return np.hypot(x - x1, y - y1)
    t = np.clip(((x - x1) * dx + (y - y1) * dy) / denominator, 0.0, 1.0)
    return np.hypot(x - (x1 + t * dx), y - (y1 + t * dy))


def rasterize_segments(
    segments: Iterable[tuple[float, float, float, float]],
    *,
    image_width: int,
    image_height: int,
    grid_shape: tuple[int, int],
    radius_in_patch_diagonals: float = 0.75,
) -> np.ndarray:
    """Rasterize pixel-space segments onto a low-resolution patch grid.

    Patch centres within ``radius_in_patch_diagonals`` patch diagonals of a
    segment are selected.  The small buffer makes thin geometric lines visible
    on DINO's 16x16 grid without claiming dense object localization.
    """

    rows, columns = grid_shape
    if image_width <= 0 or image_height <= 0:
        raise SpatialEvidenceError("Physics image dimensions must be positive")
    if rows <= 0 or columns <= 0:
        raise SpatialEvidenceError("Patch grid dimensions must be positive")
    if radius_in_patch_diagonals < 0:
        raise SpatialEvidenceError("Rasterization radius cannot be negative")

    xs = (np.arange(columns, dtype=np.float64) + 0.5) / columns
    ys = (np.arange(rows, dtype=np.float64) + 0.5) / rows
    x_grid, y_grid = np.meshgrid(xs, ys)
    radius = radius_in_patch_diagonals * math.hypot(1.0 / columns, 1.0 / rows)
    mask = np.zeros((rows, columns), dtype=bool)
    for x1, y1, x2, y2 in segments:
        normalized = (
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height,
        )
        mask |= _distance_to_segment(x_grid, y_grid, normalized) <= radius
    return mask


def physics_outlier_mask(
    cue: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a grid mask from physics evidence explicitly labelled ``outlier``."""

    evidence = cue.get("evidence", [])
    if not isinstance(evidence, list):
        raise SpatialEvidenceError("Physics cue evidence must be an array")
    supported = 0
    outlier_segments: list[tuple[float, float, float, float]] = []
    for raw_item in evidence:
        if not isinstance(raw_item, dict):
            continue
        segment = segment_from_evidence(raw_item)
        if segment is None:
            continue
        supported += 1
        if raw_item.get("role") == "outlier":
            outlier_segments.append(segment)
    mask = rasterize_segments(
        outlier_segments,
        image_width=image_width,
        image_height=image_height,
        grid_shape=grid_shape,
    )
    return mask, {
        "supported_evidence_segments": supported,
        "outlier_segments": len(outlier_segments),
        "selected_patches": int(mask.sum()),
    }


def grid_association(
    score_grid: Any,
    selected_mask: np.ndarray,
    *,
    top_fraction: float = 0.15,
) -> dict[str, Any]:
    """Compare detector evidence near selected geometry with the background."""

    grid = validate_score_grid(score_grid)
    mask = np.asarray(selected_mask, dtype=bool)
    if mask.shape != grid.shape:
        raise SpatialEvidenceError(
            f"Selected mask shape {mask.shape} does not match score grid {grid.shape}"
        )
    if not 0.0 < top_fraction <= 1.0:
        raise SpatialEvidenceError("top_fraction must lie within (0, 1]")

    selected_count = int(mask.sum())
    total_count = int(mask.size)
    background_count = total_count - selected_count
    if selected_count == 0:
        return {
            "applicable": False,
            "reason": "No supported physics outlier geometry selected a detector patch.",
            "selected_patch_count": 0,
            "total_patch_count": total_count,
        }
    if background_count == 0:
        return {
            "applicable": False,
            "reason": "Physics outlier geometry covers the entire detector grid.",
            "selected_patch_count": selected_count,
            "total_patch_count": total_count,
        }

    selected = grid[mask]
    background = grid[~mask]
    coverage = selected_count / total_count
    top_count = max(1, int(math.ceil(total_count * top_fraction)))
    flat = grid.reshape(-1)
    top_indices = np.argpartition(flat, -top_count)[-top_count:]
    top_mask = np.zeros(total_count, dtype=bool)
    top_mask[top_indices] = True
    overlap = int(np.logical_and(top_mask.reshape(grid.shape), mask).sum())
    top_precision = overlap / top_count
    enrichment = top_precision / coverage if coverage > 0 else None
    selected_mean = float(selected.mean())
    background_mean = float(background.mean())
    lift = selected_mean - background_mean

    if lift >= 0.05 and enrichment is not None and enrichment >= 1.5:
        label = "positive"
    elif lift <= -0.05 and enrichment is not None and enrichment <= 0.67:
        label = "negative"
    else:
        label = "weak_or_mixed"

    direction = "higher" if lift >= 0 else "lower"
    return {
        "applicable": True,
        "association_label": label,
        "selected_patch_count": selected_count,
        "background_patch_count": background_count,
        "total_patch_count": total_count,
        "selected_coverage_fraction": coverage,
        "mean_selected_patch_score": selected_mean,
        "mean_background_patch_score": background_mean,
        "selected_minus_background": lift,
        "top_fraction": top_fraction,
        "top_patch_count": top_count,
        "top_patch_overlap_count": overlap,
        "top_patch_precision": top_precision,
        "selected_patch_top_recall": overlap / selected_count,
        "top_patch_enrichment_over_area": enrichment,
        "summary": (
            f"DINO patch evidence is {abs(lift):.3f} {direction} near physics "
            "outlier geometry than elsewhere. This is spatial association, not "
            "proof that the detector used the physical cue."
        ),
    }
