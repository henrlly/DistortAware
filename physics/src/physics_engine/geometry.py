"""Small, auditable projective-geometry utilities.

Coordinates are ordinary Cartesian image coordinates unless a caller explicitly
normalizes them. Homogeneous points support both finite vanishing points and
directions at infinity.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


Segment = tuple[float, float, float, float]


def segment_length(segment: Segment) -> float:
    x1, y1, x2, y2 = segment
    return float(np.hypot(x2 - x1, y2 - y1))


def segment_midpoint(segment: Segment) -> np.ndarray:
    x1, y1, x2, y2 = segment
    return np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


def segment_direction(segment: Segment) -> np.ndarray:
    x1, y1, x2, y2 = segment
    vector = np.asarray([x2 - x1, y2 - y1], dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return np.zeros(2, dtype=np.float64)
    return vector / norm


def segment_to_line(segment: Segment) -> np.ndarray:
    x1, y1, x2, y2 = segment
    p1 = np.asarray([x1, y1, 1.0], dtype=np.float64)
    p2 = np.asarray([x2, y2, 1.0], dtype=np.float64)
    line = np.cross(p1, p2)
    norm = np.linalg.norm(line[:2])
    if norm <= 1e-12:
        raise ValueError("Cannot construct a line from a zero-length segment")
    return line / norm


def normalize_hpoint(point: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64)
    norm = np.linalg.norm(point)
    if norm <= 1e-12:
        raise ValueError("Degenerate homogeneous point")
    point = point / norm
    if abs(point[2]) > 1e-10 and point[2] < 0:
        point = -point
    return point


def homogeneous_point_payload(point: np.ndarray) -> dict[str, object]:
    point = normalize_hpoint(point)
    if abs(point[2]) > 1e-8:
        xy = point[:2] / point[2]
        return {
            "kind": "finite",
            "homogeneous": [float(value) for value in point],
            "xy": [float(xy[0]), float(xy[1])],
        }
    direction = point[:2]
    norm = np.linalg.norm(direction)
    if norm > 1e-12:
        direction = direction / norm
    return {
        "kind": "infinite",
        "homogeneous": [float(value) for value in point],
        "direction": [float(direction[0]), float(direction[1])],
    }


def undirected_angle_degrees(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 90.0
    cosine = abs(float(np.dot(vector_a, vector_b) / (norm_a * norm_b)))
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def directed_angle_degrees(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 180.0
    cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def segment_point_error_degrees(segment: Segment, point: np.ndarray) -> float:
    """Angular error between a segment and its ray toward a homogeneous point."""

    direction = segment_direction(segment)
    point = normalize_hpoint(point)
    if abs(point[2]) > 1e-8:
        finite_point = point[:2] / point[2]
        expected = finite_point - segment_midpoint(segment)
    else:
        expected = point[:2]
    return undirected_angle_degrees(direction, expected)


def normalize_segments(
    segments: Sequence[Segment], width: int, height: int
) -> tuple[list[Segment], float, float, float]:
    diagonal = float(np.hypot(width, height))
    center_x = width / 2.0
    center_y = height / 2.0
    scale = diagonal if diagonal > 0 else 1.0
    normalized = [
        (
            (x1 - center_x) / scale,
            (y1 - center_y) / scale,
            (x2 - center_x) / scale,
            (y2 - center_y) / scale,
        )
        for x1, y1, x2, y2 in segments
    ]
    return normalized, center_x, center_y, scale


def denormalize_hpoint(
    point: np.ndarray, center_x: float, center_y: float, scale: float
) -> dict[str, object]:
    payload = homogeneous_point_payload(point)
    if payload["kind"] == "finite":
        x, y = payload["xy"]  # type: ignore[misc]
        payload["xy_normalized"] = [x, y]
        payload["xy"] = [center_x + scale * x, center_y + scale * y]
    return payload


@dataclass(slots=True)
class LineBundleFit:
    point: np.ndarray
    inliers: np.ndarray
    errors_degrees: np.ndarray
    support_weight: float
    total_weight: float

    @property
    def support_fraction(self) -> float:
        if self.total_weight <= 1e-12:
            return 0.0
        return float(self.support_weight / self.total_weight)

    @property
    def median_inlier_error(self) -> float:
        values = self.errors_degrees[self.inliers]
        if values.size == 0:
            return 90.0
        return float(np.median(values))


def _refine_hpoint(lines: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted_lines = lines * np.sqrt(np.maximum(weights, 1e-12))[:, None]
    _, _, vh = np.linalg.svd(weighted_lines, full_matrices=False)
    return normalize_hpoint(vh[-1])


def _candidate_pairs(count: int, max_trials: int, seed: int) -> Iterable[tuple[int, int]]:
    total_pairs = count * (count - 1) // 2
    if total_pairs <= max_trials:
        yield from combinations(range(count), 2)
        return

    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int]] = set()
    while len(seen) < max_trials:
        pair = tuple(sorted(rng.choice(count, size=2, replace=False).tolist()))
        if pair in seen:
            continue
        seen.add(pair)
        yield pair


def fit_line_bundle(
    segments: Sequence[Segment],
    *,
    weights: Sequence[float] | None = None,
    threshold_degrees: float = 3.0,
    min_inliers: int = 3,
    max_trials: int = 600,
    seed: int = 7,
) -> LineBundleFit | None:
    """Robustly fit a finite or infinite common point to line segments."""

    if len(segments) < min_inliers:
        return None

    segment_array = list(segments)
    line_array = np.stack([segment_to_line(segment) for segment in segment_array])
    weight_array = (
        np.asarray(weights, dtype=np.float64)
        if weights is not None
        else np.asarray([segment_length(segment) for segment in segment_array], dtype=np.float64)
    )
    if weight_array.shape != (len(segment_array),):
        raise ValueError("weights must contain one value per segment")
    weight_array = np.maximum(weight_array, 1e-9)

    best_point: np.ndarray | None = None
    best_inliers: np.ndarray | None = None
    best_support = -1.0
    best_median = float("inf")

    for first, second in _candidate_pairs(len(segment_array), max_trials, seed):
        candidate = np.cross(line_array[first], line_array[second])
        if np.linalg.norm(candidate) <= 1e-10:
            continue
        candidate = normalize_hpoint(candidate)
        errors = np.asarray(
            [segment_point_error_degrees(segment, candidate) for segment in segment_array]
        )
        inliers = errors <= threshold_degrees
        if int(np.count_nonzero(inliers)) < min_inliers:
            continue
        support = float(np.sum(weight_array[inliers]))
        median = float(np.median(errors[inliers]))
        if support > best_support + 1e-12 or (
            abs(support - best_support) <= 1e-12 and median < best_median
        ):
            best_point = candidate
            best_inliers = inliers
            best_support = support
            best_median = median

    if best_point is None or best_inliers is None:
        return None

    point = best_point
    inliers = best_inliers
    for _ in range(3):
        point = _refine_hpoint(line_array[inliers], weight_array[inliers])
        errors = np.asarray(
            [segment_point_error_degrees(segment, point) for segment in segment_array]
        )
        updated = errors <= threshold_degrees
        if int(np.count_nonzero(updated)) < min_inliers:
            break
        if np.array_equal(updated, inliers):
            inliers = updated
            break
        inliers = updated

    errors = np.asarray(
        [segment_point_error_degrees(segment, point) for segment in segment_array]
    )
    return LineBundleFit(
        point=point,
        inliers=inliers,
        errors_degrees=errors,
        support_weight=float(np.sum(weight_array[inliers])),
        total_weight=float(np.sum(weight_array)),
    )


@dataclass(slots=True)
class MultipleBundleFit:
    bundles: list[LineBundleFit]
    assignments: np.ndarray
    errors_degrees: np.ndarray


def fit_multiple_line_bundles(
    segments: Sequence[Segment],
    *,
    weights: Sequence[float] | None = None,
    max_bundles: int = 3,
    threshold_degrees: float = 3.0,
    min_inliers: int = 4,
    min_global_support_fraction: float = 0.08,
    seed: int = 7,
) -> MultipleBundleFit:
    segment_array = list(segments)
    weight_array = (
        np.asarray(weights, dtype=np.float64)
        if weights is not None
        else np.asarray([segment_length(segment) for segment in segment_array], dtype=np.float64)
    )
    total_weight = float(np.sum(weight_array))
    assignments = np.full(len(segment_array), -1, dtype=np.int32)
    errors = np.full(len(segment_array), 90.0, dtype=np.float64)
    remaining = np.arange(len(segment_array), dtype=np.int32)
    bundles: list[LineBundleFit] = []

    for bundle_index in range(max_bundles):
        if remaining.size < min_inliers:
            break
        subset_segments = [segment_array[index] for index in remaining]
        subset_weights = weight_array[remaining]
        fit = fit_line_bundle(
            subset_segments,
            weights=subset_weights,
            threshold_degrees=threshold_degrees,
            min_inliers=min_inliers,
            seed=seed + bundle_index,
        )
        if fit is None:
            break
        global_support = fit.support_weight / max(total_weight, 1e-12)
        if global_support < min_global_support_fraction:
            break

        selected_global = remaining[fit.inliers]
        assignments[selected_global] = bundle_index
        errors[selected_global] = fit.errors_degrees[fit.inliers]
        bundles.append(fit)
        remaining = remaining[~fit.inliers]

    return MultipleBundleFit(bundles=bundles, assignments=assignments, errors_degrees=errors)
