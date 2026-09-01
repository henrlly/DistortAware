from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest
import uuid

from physics_engine.dino_integration import (
    DinoIntegrationError,
    compute_dino_physics_alignment,
    main as dino_merge_main,
    merge_dino_and_physics,
)


TEST_TEMP_ROOT = Path(__file__).resolve().parent / ".tmp"


def physics_image(path: Path, *, outlier: bool = True) -> dict[str, object]:
    role = "outlier" if outlier else "inlier"
    status = "inconsistent" if outlier else "consistent"
    score = 0.8 if outlier else 0.1
    return {
        "image_path": str(path),
        "width": 100,
        "height": 100,
        "physics": {
            "score_kind": "physics_violation_not_aigc_probability",
            "status": status,
            "violation_score": score,
            "confidence": 0.8,
            "applicable_cues": ["perspective"],
            "summary": "Physics evidence only.",
        },
        "cues": {
            "perspective": {
                "applicable": True,
                "status": status,
                "violation_score": score,
                "confidence": 0.8,
                "summary": "Geometry result.",
                "measurements": {},
                "evidence": [
                    {
                        "kind": "line_segment",
                        "xyxy": [0, 10, 100, 10],
                        "role": role,
                    }
                ],
            },
            "cast_shadow": {"applicable": False, "status": "not_applicable", "evidence": []},
            "reflection": {"applicable": False, "status": "not_applicable", "evidence": []},
        },
        "errors": [],
    }


def dino_record(path: str) -> dict[str, object]:
    return {
        "image_path": path,
        "aigc_score": 0.91,
        "is_aigc": True,
        "patch_evidence": {
            "grid_shape": [4, 4],
            "coordinate_space": "normalized_full_frame",
            "value_kind": "sigmoid_of_per_patch_aigc_logit_uncalibrated",
            "values": [
                [0.95, 0.95, 0.95, 0.95],
                [0.20, 0.20, 0.20, 0.20],
                [0.10, 0.10, 0.10, 0.10],
                [0.05, 0.05, 0.05, 0.05],
            ],
        },
    }


class DinoIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.case_dir = TEST_TEMP_ROOT / uuid.uuid4().hex
        self.case_dir.mkdir(parents=True)
        self.image_path = self.case_dir / "images" / "scene.png"
        self.physics = {
            "schema_version": "0.1.0",
            "engine_version": "0.3.0",
            "input_root": str(self.case_dir / "images"),
            "images": [physics_image(self.image_path)],
        }

    def tearDown(self) -> None:
        if self.case_dir.exists():
            shutil.rmtree(self.case_dir)

    def test_merge_preserves_dino_score_and_adds_alignment(self) -> None:
        payload = {
            "schema_version": "0.1.0",
            "input_root": str(self.case_dir),
            "detector": {"arch": "patchhead-dinov3-vitl16"},
            "images": [dino_record("images/scene.png")],
        }
        merged, summary = merge_dino_and_physics(
            payload, self.physics, path_root=self.case_dir
        )

        record = merged["images"][0]
        self.assertEqual(record["aigc_score"], 0.91)
        self.assertIs(record["is_aigc"], True)
        self.assertEqual(record["physics_evidence"]["engine_version"], "0.3.0")
        self.assertTrue(record["dino_physics_alignment"]["applicable"])
        self.assertEqual(summary["records_with_spatial_alignment"], 1)
        self.assertIs(merged["physics_integration"]["physics_affects_detector_score"], False)

    def test_consistent_geometry_has_no_suspicious_alignment_target(self) -> None:
        image = physics_image(self.image_path, outlier=False)
        alignment = compute_dino_physics_alignment(dino_record(str(self.image_path)), image)

        self.assertFalse(alignment["applicable"])
        self.assertFalse(alignment["per_cue"]["perspective"]["applicable"])

    def test_consistent_cue_hides_even_isolated_fit_outliers(self) -> None:
        image = physics_image(self.image_path, outlier=True)
        image["cues"]["perspective"]["status"] = "consistent"
        image["cues"]["perspective"]["violation_score"] = 0.2

        alignment = compute_dino_physics_alignment(dino_record(str(self.image_path)), image)

        self.assertFalse(alignment["applicable"])
        self.assertIn(
            "globally consistent",
            alignment["per_cue"]["perspective"]["reason"],
        )

    def test_missing_patch_map_still_allows_non_spatial_physics_merge(self) -> None:
        record = {"image_path": str(self.image_path), "aigc_score": 0.3}
        merged, summary = merge_dino_and_physics([record], self.physics)

        self.assertIsNotNone(merged[0]["physics_evidence"])
        self.assertFalse(merged[0]["dino_physics_alignment"]["applicable"])
        self.assertEqual(summary["records_with_spatial_alignment"], 0)

    def test_unknown_coordinate_mapping_fails_closed(self) -> None:
        record = dino_record(str(self.image_path))
        record["patch_evidence"]["coordinate_space"] = "letterboxed_model_input"
        with self.assertRaisesRegex(DinoIntegrationError, "inverse transform"):
            compute_dino_physics_alignment(record, physics_image(self.image_path))

    def test_cli_writes_enriched_export_shape(self) -> None:
        dino_path = self.case_dir / "dino.json"
        physics_path = self.case_dir / "physics.json"
        output_path = self.case_dir / "enriched.json"
        dino_path.write_text(
            json.dumps(
                {
                    "input_root": str(self.case_dir),
                    "images": [dino_record("images/scene.png")],
                }
            ),
            encoding="utf-8",
        )
        physics_path.write_text(json.dumps(self.physics), encoding="utf-8")

        code = dino_merge_main(
            [
                "--dino-results",
                str(dino_path),
                "--physics-results",
                str(physics_path),
                "--output",
                str(output_path),
                "--path-root",
                str(self.case_dir),
                "--pretty",
            ]
        )

        self.assertEqual(code, 0)
        output = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertIn("dino_physics_alignment", output["images"][0])


if __name__ == "__main__":
    unittest.main()
