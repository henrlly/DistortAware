from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest
import uuid

from physics_engine.integration import (
    IntegrationError,
    main as merge_main,
    merge_detector_and_physics,
)


TEST_TEMP_ROOT = Path(__file__).resolve().parent / ".tmp"


def physics_image(path: Path) -> dict[str, object]:
    return {
        "image_path": str(path),
        "width": 100,
        "height": 80,
        "physics": {
            "score_kind": "physics_violation_not_aigc_probability",
            "status": "consistent",
            "violation_score": 0.1,
            "confidence": 0.8,
            "applicable_cues": ["perspective"],
            "summary": "Physics evidence only.",
        },
        "cues": {
            "perspective": {
                "applicable": True,
                "status": "consistent",
                "violation_score": 0.1,
                "confidence": 0.8,
                "summary": "Geometrically coherent.",
                "measurements": {"bundle_count": 2},
                "evidence": [{"kind": "line_segment"}],
            }
        },
        "errors": [],
    }


class MergeTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.case_dir = TEST_TEMP_ROOT / uuid.uuid4().hex
        self.case_dir.mkdir(parents=True)
        self.image_path = self.case_dir / "images" / "scene.png"
        self.physics = {
            "schema_version": "0.1.0",
            "engine_version": "0.2.0",
            "input_root": str(self.case_dir / "images"),
            "images": [physics_image(self.image_path)],
        }

    def tearDown(self) -> None:
        if self.case_dir.exists():
            shutil.rmtree(self.case_dir)

    def test_merge_preserves_primary_prediction_and_compacts_evidence(self) -> None:
        detector = [
            {"image_path": "images/scene.png", "pred": 0.83, "is_aigc": True}
        ]

        merged, summary = merge_detector_and_physics(
            detector, self.physics, path_root=self.case_dir
        )

        self.assertEqual(merged[0]["pred"], 0.83)
        self.assertIs(merged[0]["is_aigc"], True)
        evidence = merged[0]["physics_evidence"]
        self.assertEqual(evidence["engine_version"], "0.2.0")
        self.assertEqual(evidence["match_method"], "canonical_path")
        self.assertEqual(evidence["aggregate"]["score_kind"], "physics_violation_not_aigc_probability")
        self.assertNotIn("evidence", evidence["cues"]["perspective"])
        self.assertEqual(summary["matched_records"], 1)

    def test_missing_match_fails_closed_by_default(self) -> None:
        detector = [{"image_path": "missing.png", "pred": 0.2, "is_aigc": False}]

        with self.assertRaisesRegex(IntegrationError, "No unambiguous physics result"):
            merge_detector_and_physics(detector, self.physics, path_root=self.case_dir)

    def test_allow_missing_uses_explicit_null(self) -> None:
        detector = [{"image_path": "missing.png", "pred": 0.2, "is_aigc": False}]

        merged, summary = merge_detector_and_physics(
            detector, self.physics, path_root=self.case_dir, allow_missing=True
        )

        self.assertIsNone(merged[0]["physics_evidence"])
        self.assertEqual(summary["missing_records"], 1)

    def test_unique_basename_fallback_is_disclosed(self) -> None:
        detector = [
            {
                "image_path": "/different-machine/upload/scene.png",
                "pred": 0.4,
                "is_aigc": False,
            }
        ]

        merged, _summary = merge_detector_and_physics(
            detector, self.physics, path_root=self.case_dir
        )

        self.assertEqual(
            merged[0]["physics_evidence"]["match_method"], "unique_basename"
        )

    def test_ambiguous_basename_does_not_silently_join(self) -> None:
        duplicate = physics_image(self.case_dir / "other" / "scene.png")
        self.physics["images"].append(duplicate)
        detector = [
            {"image_path": "unknown/scene.png", "pred": 0.5, "is_aigc": False}
        ]

        with self.assertRaisesRegex(IntegrationError, "No unambiguous physics result"):
            merge_detector_and_physics(detector, self.physics, path_root=self.case_dir)

    def test_cli_writes_drop_in_json_array(self) -> None:
        detector_path = self.case_dir / "preds.json"
        physics_path = self.case_dir / "physics.json"
        output_path = self.case_dir / "enriched.json"
        detector_path.write_text(
            json.dumps(
                [{"image_path": "images/scene.png", "pred": 0.7, "is_aigc": True}]
            ),
            encoding="utf-8",
        )
        physics_path.write_text(json.dumps(self.physics), encoding="utf-8")

        exit_code = merge_main(
            [
                "--detector-results",
                str(detector_path),
                "--physics-results",
                str(physics_path),
                "--output",
                str(output_path),
                "--path-root",
                str(self.case_dir),
                "--pretty",
            ]
        )

        self.assertEqual(exit_code, 0)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertIsInstance(payload, list)
        self.assertEqual(payload[0]["pred"], 0.7)
        self.assertIn("physics_evidence", payload[0])


if __name__ == "__main__":
    unittest.main()
