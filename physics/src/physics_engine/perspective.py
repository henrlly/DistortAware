"""Automatic perspective consistency analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .geometry import (
    denormalize_hpoint,
    fit_multiple_line_bundles,
    normalize_segments,
    segment_length,
)
from .line_detection import LineDetectionConfig, Region, detect_line_segments
from .schema import CueResult, not_applicable


@dataclass(slots=True)
class PerspectiveConfig:
    line_detection: LineDetectionConfig = field(default_factory=LineDetectionConfig)
    min_lines: int = 12
    min_total_length_diagonals: float = 2.25
    max_vanishing_points: int = 3
    inlier_threshold_degrees: float = 3.0
    min_bundle_lines: int = 4
    min_bundle_support_fraction: float = 0.08
    consistent_threshold: float = 0.32
    inconsistent_threshold: float = 0.62
    max_frame_aspect_ratio: float = 3.0
    min_reviewed_region_confidence: float = 0.5
    min_orientation_peak_fraction: float = 0.50
    max_orientation_entropy: float = 0.88
    enable_stability_gate: bool = True
    stability_crop_fractions: tuple[float, ...] = (0.90, 0.80)
    stability_min_applicable_views: int = 2
    stability_max_score_range: float = 0.22


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _status(score: float, config: PerspectiveConfig) -> str:
    if score >= config.inconsistent_threshold:
        return "inconsistent"
    if score <= config.consistent_threshold:
        return "consistent"
    return "indeterminate"


def _analyze_single_view(
    image: Image.Image,
    config: PerspectiveConfig | None = None,
    regions: list[Region] | None = None,
) -> CueResult:
    config = config or PerspectiveConfig()
    width, height = image.size
    frame_aspect_ratio = max(width / max(height, 1), height / max(width, 1))
    if frame_aspect_ratio > config.max_frame_aspect_ratio:
        result = not_applicable(
            "perspective",
            (
                "The frame aspect ratio is panorama-like, so a pinhole-camera "
                "perspective test is not defensible without projection metadata."
            ),
            limitations=[
                "Wide crops and stitched or equirectangular panoramas cannot be distinguished reliably from pixels alone.",
                "This conservative gate can withhold valid perspective evidence from unusually wide or tall photographs.",
            ],
        )
        result.assumptions = [
            "The scene is approximately pinhole-projected rather than panoramic or strongly distorted."
        ]
        result.measurements = {
            "frame_width": width,
            "frame_height": height,
            "frame_aspect_ratio": frame_aspect_ratio,
            "maximum_frame_aspect_ratio": config.max_frame_aspect_ratio,
            "projection_safety_gate": {
                "passed": False,
                "reason": "panorama_like_frame_aspect_ratio",
            },
        }
        return result
    detection = detect_line_segments(image, config.line_detection, regions)
    segments = detection.segments
    diagnostics = detection.diagnostics
    if regions is not None:
        diagnostics["reviewed_regions_xyxy"] = [
            [float(value) for value in region] for region in regions
        ]
    total_length_diagonals = float(diagnostics["total_length_diagonals"])

    assumptions = [
        "The scene is approximately pinhole-projected rather than panoramic or strongly distorted.",
        "Visible structural edges are straight in the depicted world.",
        "At most three dominant vanishing-point families explain most long structural lines.",
    ]
    limitations = [
        "Curved or non-Manhattan architecture can be physically valid but receive a high residual.",
        "Blur, severe downscaling, texture, and decorative edges can change line extraction.",
        "This score measures geometric consistency; it is not an AIGC probability.",
    ]
    if regions is not None:
        assumptions.append(
            "The reviewed rectangles contain structural perspective evidence and exclude primarily semantic or decorative edges."
        )
        limitations.append(
            "Region selection is reviewer-dependent; agreement should be measured before drawing subgroup conclusions."
        )

    if len(segments) < config.min_lines:
        result = not_applicable(
            "perspective",
            f"Only {len(segments)} usable long lines were detected; at least {config.min_lines} are required.",
            limitations=limitations,
        )
        result.assumptions = assumptions
        result.measurements = diagnostics
        return result

    if total_length_diagonals < config.min_total_length_diagonals:
        result = not_applicable(
            "perspective",
            "Detected straight-line coverage is too low for a stable perspective test.",
            limitations=limitations,
        )
        result.assumptions = assumptions
        result.measurements = diagnostics
        return result

    orientation_peak = float(diagnostics.get("orientation_peak_fraction", 0.0))
    orientation_entropy = float(diagnostics.get("orientation_entropy", 0.0))
    if (
        orientation_peak < config.min_orientation_peak_fraction
        and orientation_entropy > config.max_orientation_entropy
    ):
        result = not_applicable(
            "perspective",
            (
                "Long-line evidence is too orientation-diffuse to establish a "
                "defensible structural perspective region."
            ),
            limitations=limitations,
        )
        result.assumptions = assumptions
        result.measurements = {
            **diagnostics,
            "structural_applicability_gate": {
                "passed": False,
                "orientation_peak_fraction": orientation_peak,
                "minimum_orientation_peak_fraction": config.min_orientation_peak_fraction,
                "orientation_entropy": orientation_entropy,
                "maximum_orientation_entropy": config.max_orientation_entropy,
                "reason": "orientation_diffuse_line_field",
            },
        }
        return result

    normalized, center_x, center_y, scale = normalize_segments(segments, width, height)
    weights = np.asarray([segment_length(segment) for segment in segments], dtype=np.float64)
    fit = fit_multiple_line_bundles(
        normalized,
        weights=weights,
        max_bundles=config.max_vanishing_points,
        threshold_degrees=config.inlier_threshold_degrees,
        min_inliers=config.min_bundle_lines,
        min_global_support_fraction=config.min_bundle_support_fraction,
    )

    assigned = fit.assignments >= 0
    explained_weight = float(np.sum(weights[assigned]))
    total_weight = float(np.sum(weights))
    explained_fraction = explained_weight / max(total_weight, 1e-12)
    unexplained_fraction = 1.0 - explained_fraction
    assigned_errors = fit.errors_degrees[assigned]
    median_error = float(np.median(assigned_errors)) if assigned_errors.size else 90.0

    # A small unexplained fraction is normal: line detectors also capture text,
    # ornamentation, and object edges. The score rises only beyond that allowance.
    unexplained_component = _clamp01((unexplained_fraction - 0.18) / 0.62)
    residual_component = _clamp01((median_error - 0.5) / 5.5)
    no_bundle_penalty = 0.18 if not fit.bundles else 0.0
    violation_score = _clamp01(
        0.82 * unexplained_component + 0.18 * residual_component + no_bundle_penalty
    )

    count_factor = _clamp01((len(segments) - config.min_lines) / 35.0)
    coverage_factor = _clamp01(
        (total_length_diagonals - config.min_total_length_diagonals) / 5.0
    )
    confidence = _clamp01(0.48 + 0.28 * count_factor + 0.24 * coverage_factor)
    status = _status(violation_score, config)

    bundle_payloads: list[dict[str, object]] = []
    for bundle_index, bundle in enumerate(fit.bundles):
        member_mask = fit.assignments == bundle_index
        global_support = float(np.sum(weights[member_mask]) / max(total_weight, 1e-12))
        bundle_payloads.append(
            {
                "bundle_index": bundle_index,
                "vanishing_point": denormalize_hpoint(
                    bundle.point, center_x, center_y, scale
                ),
                "line_count": int(np.count_nonzero(member_mask)),
                "support_fraction": global_support,
                "median_error_degrees": float(np.median(fit.errors_degrees[member_mask])),
            }
        )

    segment_evidence: list[dict[str, object]] = []
    for index, segment in enumerate(segments):
        assignment = int(fit.assignments[index])
        segment_evidence.append(
            {
                "kind": "line_segment",
                "index": index,
                "xyxy": [round(float(value), 3) for value in segment],
                "length_pixels": round(segment_length(segment), 3),
                "bundle_index": assignment,
                "role": "outlier" if assignment < 0 else "inlier",
                "error_degrees": (
                    None if assignment < 0 else round(float(fit.errors_degrees[index]), 4)
                ),
            }
        )

    if status == "consistent":
        summary = (
            f"{explained_fraction:.0%} of weighted structural-line evidence is explained "
            f"by {len(fit.bundles)} coherent vanishing-point bundle(s)."
        )
    elif status == "inconsistent":
        summary = (
            f"Only {explained_fraction:.0%} of weighted structural-line evidence fits the "
            f"strongest {config.max_vanishing_points} vanishing-point bundles."
        )
    else:
        summary = (
            f"Perspective evidence is mixed: {explained_fraction:.0%} of weighted lines "
            "fit coherent vanishing-point bundles."
        )

    measurements = dict(diagnostics)
    measurements.update(
        {
            "frame_aspect_ratio": frame_aspect_ratio,
            "projection_safety_gate": {
                "passed": True,
                "maximum_frame_aspect_ratio": config.max_frame_aspect_ratio,
            },
            "bundle_count": len(fit.bundles),
            "explained_weight_fraction": explained_fraction,
            "unexplained_weight_fraction": unexplained_fraction,
            "median_assigned_error_degrees": median_error,
            "inlier_threshold_degrees": config.inlier_threshold_degrees,
            "vanishing_points": bundle_payloads,
            "structural_applicability_gate": {
                "passed": True,
                "orientation_peak_fraction": orientation_peak,
                "minimum_orientation_peak_fraction": config.min_orientation_peak_fraction,
                "orientation_entropy": orientation_entropy,
                "maximum_orientation_entropy": config.max_orientation_entropy,
            },
        }
    )

    return CueResult(
        cue="perspective",
        applicable=True,
        status=status,
        violation_score=violation_score,
        confidence=confidence,
        summary=summary,
        assumptions=assumptions,
        measurements=measurements,
        evidence=segment_evidence,
        limitations=limitations,
    )


def _center_crop_box(image: Image.Image, fraction: float) -> tuple[int, int, int, int]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Perspective stability crop fractions must lie within (0, 1]")
    width, height = image.size
    crop_width = max(1, int(round(width * fraction)))
    crop_height = max(1, int(round(height * fraction)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return left, top, left + crop_width, top + crop_height


def _center_crop(image: Image.Image, fraction: float) -> Image.Image:
    return image.crop(_center_crop_box(image, fraction))


def _crop_regions(
    regions: list[Region], crop_box: tuple[int, int, int, int]
) -> list[Region]:
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    mapped: list[Region] = []
    for left, top, right, bottom in regions:
        intersection_left = max(left, crop_left)
        intersection_top = max(top, crop_top)
        intersection_right = min(right, crop_right)
        intersection_bottom = min(bottom, crop_bottom)
        if intersection_right <= intersection_left or intersection_bottom <= intersection_top:
            continue
        mapped.append(
            (
                intersection_left - crop_left,
                intersection_top - crop_top,
                intersection_right - crop_left,
                intersection_bottom - crop_top,
            )
        )
    return mapped


def analyze_perspective(
    image: Image.Image,
    config: PerspectiveConfig | None = None,
    regions: list[Region] | None = None,
) -> CueResult:
    """Analyze perspective and suppress crop-sensitive suspicious explanations."""

    config = config or PerspectiveConfig()
    base = _analyze_single_view(image, config, regions)
    if (
        not config.enable_stability_gate
        or not base.applicable
        or base.violation_score is None
        or base.status == "consistent"
    ):
        return base

    views: list[tuple[str, CueResult]] = [("full", base)]
    for fraction in config.stability_crop_fractions:
        crop_box = _center_crop_box(image, fraction)
        cropped_regions = (
            _crop_regions(regions, crop_box) if regions is not None else None
        )
        views.append(
            (
                f"center_crop_{fraction:.2f}",
                _analyze_single_view(image.crop(crop_box), config, cropped_regions),
            )
        )

    applicable = [
        result
        for _name, result in views
        if result.applicable and result.violation_score is not None
    ]
    scores = [float(result.violation_score) for result in applicable]
    score_range = max(scores) - min(scores) if len(scores) >= 2 else None
    statuses = {result.status for result in applicable}
    hard_disagreement = "consistent" in statuses and "inconsistent" in statuses
    unstable = (
        len(applicable) < config.stability_min_applicable_views
        or score_range is None
        or score_range > config.stability_max_score_range
        or hard_disagreement
    )
    stability = {
        "gate_applied": True,
        "passed": not unstable,
        "minimum_applicable_views": config.stability_min_applicable_views,
        "maximum_score_range": config.stability_max_score_range,
        "applicable_view_count": len(applicable),
        "score_range": score_range,
        "hard_status_disagreement": hard_disagreement,
        "views": [
            {
                "view": name,
                "applicable": result.applicable,
                "status": result.status,
                "violation_score": result.violation_score,
            }
            for name, result in views
        ],
    }
    base.measurements["multi_view_stability"] = stability
    if unstable:
        base.measurements["ungated_status"] = base.status
        base.measurements["ungated_violation_score"] = base.violation_score
        base.status = "indeterminate"
        base.violation_score = 0.5
        base.confidence = min(base.confidence, 0.35)
        base.summary = (
            "Perspective evidence changed materially across nearby center views, so "
            "the result is withheld as crop-sensitive and indeterminate."
        )
        base.limitations.append(
            "The multi-view safety gate suppressed a crop-sensitive perspective claim."
        )
        return base

    stable_score = float(np.median(np.asarray(scores, dtype=np.float64)))
    base.measurements["full_view_violation_score"] = base.violation_score
    base.violation_score = stable_score
    base.status = _status(stable_score, config)
    base.confidence *= max(0.0, 1.0 - (score_range or 0.0))
    base.summary = (
        f"Across {len(applicable)} nearby views, the median perspective-violation "
        f"score is {stable_score:.3f}; the stable result is {base.status}."
    )
    return base
