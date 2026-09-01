from __future__ import annotations

import unittest

import numpy as np

from physics_engine.spatial import (
    SpatialEvidenceError,
    grid_association,
    physics_outlier_mask,
    rasterize_segments,
    validate_score_grid,
)


class SpatialEvidenceTests(unittest.TestCase):
    def test_outlier_geometry_selects_nearby_patch_scores(self) -> None:
        cue = {
            "status": "inconsistent",
            "evidence": [
                {
                    "kind": "line_segment",
                    "xyxy": [0, 10, 100, 10],
                    "role": "outlier",
                },
                {
                    "kind": "line_segment",
                    "xyxy": [0, 90, 100, 90],
                    "role": "inlier",
                },
            ],
        }
        mask, diagnostics = physics_outlier_mask(
            cue, image_width=100, image_height=100, grid_shape=(8, 8)
        )
        scores = np.full((8, 8), 0.1)
        scores[mask] = 0.9
        association = grid_association(scores, mask, top_fraction=0.2)

        self.assertEqual(diagnostics["outlier_segments"], 1)
        self.assertTrue(association["applicable"])
        self.assertEqual(association["association_label"], "positive")
        self.assertGreater(association["selected_minus_background"], 0.7)

    def test_no_outlier_geometry_is_explicitly_not_applicable(self) -> None:
        cue = {
            "status": "consistent",
            "evidence": [
                {
                    "kind": "shadow_vector",
                    "object_contact": [10, 20],
                    "shadow_tip": [30, 40],
                    "role": "inlier",
                }
            ],
        }
        mask, diagnostics = physics_outlier_mask(
            cue, image_width=100, image_height=100, grid_shape=(4, 4)
        )
        association = grid_association(np.full((4, 4), 0.5), mask)

        self.assertEqual(diagnostics["outlier_segments"], 0)
        self.assertFalse(association["applicable"])

    def test_rasterization_rejects_invalid_dimensions(self) -> None:
        with self.assertRaises(SpatialEvidenceError):
            rasterize_segments([], image_width=0, image_height=10, grid_shape=(4, 4))

    def test_score_grid_validation_rejects_logits_and_nonfinite_values(self) -> None:
        with self.assertRaises(SpatialEvidenceError):
            validate_score_grid([[0.2, 1.2]])
        with self.assertRaises(SpatialEvidenceError):
            validate_score_grid([[0.2, float("nan")]])


if __name__ == "__main__":
    unittest.main()
