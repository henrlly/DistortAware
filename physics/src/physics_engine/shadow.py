"""Cast-shadow consistency from reviewed object-contact/shadow-tip pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .annotations import ShadowPair
from .geometry import (
    Segment,
    denormalize_hpoint,
    fit_line_bundle,
    normalize_segments,
    segment_length,
)
from .schema import CueResult, not_applicable


@dataclass(slots=True)
class ShadowConfig:
    min_pairs: int = 3
    inlier_threshold_degrees: float = 5.0
    consistent_threshold: float = 0.34
    inconsistent_threshold: float = 0.62


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def analyze_cast_shadows(
    pairs: list[ShadowPair], width: int, height: int, config: ShadowConfig | None = None
) -> CueResult:
    config = config or ShadowConfig()
    assumptions = [
        "Each supplied pair correctly associates an object contact with its cast-shadow tip.",
        "One dominant point-like or distant directional light explains the selected shadows.",
        "The selected points represent comparable object-ground shadow geometry.",
    ]
    limitations = [
        "Multiple lights, diffuse illumination, uneven terrain, and soft shadows invalidate this test.",
        "Incorrectly proposed or reviewed correspondences can dominate the geometric result.",
        "A consistent result does not establish that an image is authentic.",
    ]
    if len(pairs) < config.min_pairs:
        result = not_applicable(
            "cast_shadow",
            f"{len(pairs)} object-shadow pair(s) were supplied; at least {config.min_pairs} are required.",
            limitations=limitations,
        )
        result.assumptions = assumptions
        result.measurements = {
            "reviewed_pair_count": len(pairs),
            "required_pair_count": config.min_pairs,
        }
        return result

    segments: list[Segment] = [
        (
            pair.object_contact[0],
            pair.object_contact[1],
            pair.shadow_tip[0],
            pair.shadow_tip[1],
        )
        for pair in pairs
    ]
    valid = [index for index, segment in enumerate(segments) if segment_length(segment) >= 3.0]
    if len(valid) < config.min_pairs:
        result = not_applicable(
            "cast_shadow",
            "Too many annotated shadow vectors are degenerate or shorter than three pixels.",
            limitations=limitations,
        )
        result.assumptions = assumptions
        result.measurements = {
            "reviewed_pair_count": len(pairs),
            "usable_pair_count": len(valid),
        }
        return result

    pairs = [pairs[index] for index in valid]
    segments = [segments[index] for index in valid]
    normalized, center_x, center_y, scale = normalize_segments(segments, width, height)
    weights = np.asarray(
        [segment_length(segment) * max(pair.confidence, 0.05) for segment, pair in zip(segments, pairs)],
        dtype=np.float64,
    )
    min_inliers = 3
    fit = fit_line_bundle(
        normalized,
        weights=weights,
        threshold_degrees=config.inlier_threshold_degrees,
        min_inliers=min_inliers,
    )

    if fit is None:
        inliers = np.zeros(len(segments), dtype=bool)
        errors = np.full(len(segments), 90.0)
        explained_fraction = 0.0
        median_error = 90.0
        source_payload: dict[str, object] | None = None
    else:
        inliers = fit.inliers
        errors = fit.errors_degrees
        explained_fraction = fit.support_fraction
        median_error = fit.median_inlier_error
        source_payload = denormalize_hpoint(fit.point, center_x, center_y, scale)

    unexplained_component = _clamp01((1.0 - explained_fraction) / 0.55)
    residual_component = _clamp01((median_error - 0.75) / 10.0)
    violation_score = _clamp01(0.76 * unexplained_component + 0.24 * residual_component)
    confidence = _clamp01(
        0.48
        + 0.07 * min(len(segments) - config.min_pairs, 4)
        + 0.24 * float(np.mean([pair.confidence for pair in pairs]))
    )

    if violation_score >= config.inconsistent_threshold:
        status = "inconsistent"
        summary = (
            f"Only {explained_fraction:.0%} of shadow-vector evidence agrees "
            "with one projected light source or direction."
        )
    elif violation_score <= config.consistent_threshold:
        status = "consistent"
        summary = (
            f"{explained_fraction:.0%} of shadow-vector evidence agrees with "
            "one projected light source or direction."
        )
    else:
        status = "indeterminate"
        summary = (
            f"Shadow evidence is mixed: {explained_fraction:.0%} of weighted vectors "
            "fit one projected light constraint."
        )

    evidence: list[dict[str, object]] = []
    for index, (pair, segment) in enumerate(zip(pairs, segments)):
        evidence.append(
            {
                "kind": "shadow_vector",
                "index": index,
                "object_contact": [float(value) for value in pair.object_contact],
                "shadow_tip": [float(value) for value in pair.shadow_tip],
                "confidence": pair.confidence,
                "role": "inlier" if bool(inliers[index]) else "outlier",
                "error_degrees": None if fit is None else round(float(errors[index]), 4),
                "length_pixels": round(segment_length(segment), 3),
            }
        )

    return CueResult(
        cue="cast_shadow",
        applicable=True,
        status=status,
        violation_score=violation_score,
        confidence=confidence,
        summary=summary,
        assumptions=assumptions,
        measurements={
            "reviewed_pair_count": len(pairs),
            "explained_weight_fraction": explained_fraction,
            "median_inlier_error_degrees": median_error,
            "inlier_threshold_degrees": config.inlier_threshold_degrees,
            "estimated_projected_light": source_payload,
        },
        evidence=evidence,
        limitations=limitations,
    )
