from __future__ import annotations

import unittest

from physics_engine.annotations import ImageAnnotations, PerspectiveRegion
from physics_engine.robustness import (
    TRANSFORMS,
    _crop_point,
    _map_annotations,
    summarize_rows,
)


class TransformTests(unittest.TestCase):
    def test_suite_matches_official_transform_names(self) -> None:
        self.assertEqual(
            [transform.name for transform in TRANSFORMS],
            [
                "clean",
                "jpeg90",
                "jpeg70",
                "jpeg50",
                "jpeg30",
                "blur0.5",
                "blur1.0",
                "blur2.0",
                "resize0.5",
                "resize0.25",
                "noise0.02",
                "noise0.05",
                "noise0.10",
                "jitter",
                "crop80",
            ],
        )

    def test_crop_mapping_rescales_visible_points_and_drops_removed_points(self) -> None:
        self.assertEqual(_crop_point((50.0, 50.0), 100, 100), (50.0, 50.0))
        self.assertIsNone(_crop_point((5.0, 50.0), 100, 100))
        self.assertEqual(_crop_point((10.0, 10.0), 100, 100), (0.0, 0.0))

    def test_reviewed_perspective_regions_are_preserved_for_transforms(self) -> None:
        annotations = ImageAnnotations(
            perspective_regions=[
                PerspectiveRegion((20.0, 15.0, 80.0, 75.0), confidence=0.8)
            ],
            shadow_applicability="uncertain",
        )

        payload = _map_annotations(annotations, TRANSFORMS[0], 100, 90)

        self.assertEqual(
            payload["perspective"]["regions"],
            [{"xyxy": [20.0, 15.0, 80.0, 75.0], "confidence": 0.8}],
        )
        self.assertEqual(payload["cast_shadow"]["applicability"], "uncertain")


class SummaryTests(unittest.TestCase):
    def test_summary_counts_hard_flips_and_applicability_loss(self) -> None:
        rows = [
            {
                "source_image": "a.png",
                "transform": "clean",
                "cue": "perspective",
                "applicable": True,
                "status": "consistent",
                "violation_score": 0.1,
            },
            {
                "source_image": "a.png",
                "transform": "jpeg30",
                "cue": "perspective",
                "applicable": True,
                "status": "inconsistent",
                "violation_score": 0.8,
            },
            {
                "source_image": "a.png",
                "transform": "blur2.0",
                "cue": "perspective",
                "applicable": False,
                "status": "not_applicable",
                "violation_score": None,
            },
        ]

        summary = summarize_rows(rows)

        cue = summary["per_cue"]["perspective"]
        self.assertEqual(cue["evaluable_transform_cases"], 2)
        self.assertEqual(cue["applicability_retained_cases"], 1)
        self.assertEqual(cue["hard_status_flips"], 1)
        self.assertAlmostEqual(cue["mean_absolute_score_drift"], 0.7)
        self.assertFalse(summary["acceptance"]["passed"])


if __name__ == "__main__":
    unittest.main()
