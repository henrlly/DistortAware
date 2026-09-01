from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image

from physics_engine.line_detection import LineDetectionResult
from physics_engine.perspective import (
    _analyze_single_view,
    _center_crop,
    _crop_regions,
    analyze_perspective,
)
from physics_engine.schema import CueResult


def cue(status: str, score: float) -> CueResult:
    return CueResult(
        cue="perspective",
        applicable=True,
        status=status,
        violation_score=score,
        confidence=0.8,
        summary="fixture",
        measurements={},
        evidence=[],
        limitations=[],
    )


class PerspectiveSafetyTests(unittest.TestCase):
    def test_panorama_like_frame_is_not_applicable(self) -> None:
        image = Image.new("RGB", (800, 120), "gray")

        result = analyze_perspective(image)

        self.assertFalse(result.applicable)
        self.assertEqual(
            result.measurements["projection_safety_gate"]["reason"],
            "panorama_like_frame_aspect_ratio",
        )

    def test_orientation_diffuse_line_field_is_not_structural_evidence(self) -> None:
        segments = [
            (10.0, float(10 + index * 8), 210.0, float(20 + index * 8))
            for index in range(12)
        ]
        detection = LineDetectionResult(
            segments=segments,
            diagnostics={
                "retained_count": 12,
                "total_length_diagonals": 3.2,
                "orientation_peak_fraction": 0.34,
                "orientation_entropy": 0.95,
            },
        )
        with patch(
            "physics_engine.perspective.detect_line_segments", return_value=detection
        ):
            result = _analyze_single_view(
                Image.new("RGB", (320, 240), "white"),
                regions=[(20.0, 30.0, 280.0, 210.0)],
            )

        self.assertFalse(result.applicable)
        self.assertEqual(result.status, "not_applicable")
        gate = result.measurements["structural_applicability_gate"]
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["reason"], "orientation_diffuse_line_field")
        self.assertEqual(
            result.measurements["reviewed_regions_xyxy"],
            [[20.0, 30.0, 280.0, 210.0]],
        )

    def test_crop_sensitive_suspicious_result_is_downgraded(self) -> None:
        views = [cue("inconsistent", 0.78), cue("consistent", 0.18), cue("indeterminate", 0.48)]
        with patch("physics_engine.perspective._analyze_single_view", side_effect=views):
            result = analyze_perspective(Image.new("RGB", (320, 240), "white"))

        self.assertEqual(result.status, "indeterminate")
        self.assertEqual(result.violation_score, 0.5)
        self.assertLessEqual(result.confidence, 0.35)
        stability = result.measurements["multi_view_stability"]
        self.assertFalse(stability["passed"])
        self.assertTrue(stability["hard_status_disagreement"])
        self.assertAlmostEqual(stability["score_range"], 0.60)

    def test_stable_views_use_median_score(self) -> None:
        views = [cue("indeterminate", 0.48), cue("indeterminate", 0.52), cue("indeterminate", 0.50)]
        with patch("physics_engine.perspective._analyze_single_view", side_effect=views):
            result = analyze_perspective(Image.new("RGB", (320, 240), "white"))

        self.assertEqual(result.status, "indeterminate")
        self.assertAlmostEqual(result.violation_score or 0.0, 0.50)
        self.assertTrue(result.measurements["multi_view_stability"]["passed"])

    def test_stable_median_updates_status_and_summary_together(self) -> None:
        views = [cue("indeterminate", 0.34), cue("consistent", 0.30), cue("consistent", 0.28)]
        with patch("physics_engine.perspective._analyze_single_view", side_effect=views):
            result = analyze_perspective(Image.new("RGB", (320, 240), "white"))

        self.assertEqual(result.status, "consistent")
        self.assertAlmostEqual(result.violation_score or 0.0, 0.30)
        self.assertIn("stable result is consistent", result.summary)

    def test_center_crop_preserves_requested_fraction(self) -> None:
        cropped = _center_crop(Image.new("RGB", (100, 80)), 0.8)
        self.assertEqual(cropped.size, (80, 64))
        with self.assertRaisesRegex(ValueError, "fractions"):
            _center_crop(Image.new("RGB", (10, 10)), 0.0)

    def test_reviewed_regions_are_clipped_into_stability_crop(self) -> None:
        self.assertEqual(
            _crop_regions([(10.0, 10.0, 90.0, 90.0)], (20, 20, 80, 80)),
            [(0.0, 0.0, 60.0, 60.0)],
        )
        self.assertEqual(
            _crop_regions([(0.0, 0.0, 10.0, 10.0)], (20, 20, 80, 80)), []
        )


if __name__ == "__main__":
    unittest.main()
