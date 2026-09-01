from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from PIL import Image

from physics_engine.annotations import AnnotationStore
from physics_engine.cli import main as cli_main
from physics_engine.engine import PhysicsEngine


TEST_TEMP_ROOT = Path(__file__).resolve().parent / ".tmp"


class AnnotationStoreTests(unittest.TestCase):
    def test_normalized_coordinates_are_scaled(self) -> None:
        store = AnnotationStore(
            {
                "coordinate_space": "normalized",
                "images": {
                    "scene.png": {
                        "perspective": {
                            "regions": [
                                {"xyxy": [0.1, 0.2, 0.9, 0.8], "confidence": 0.9}
                            ]
                        },
                        "cast_shadow": {
                            "applicability": "applicable",
                            "pairs": [
                                {
                                    "object_contact": [0.25, 0.5],
                                    "shadow_tip": [0.75, 1.0],
                                    "confidence": 0.8,
                                }
                            ]
                        },
                        "reflection": {
                            "applicability": "applicable",
                            "pairs": [
                                {
                                    "object_point": [0.1, 0.2],
                                    "reflection_point": [0.9, 0.2],
                                }
                            ]
                        },
                    }
                },
            }
        )

        result = store.for_image(
            Path("/arbitrary/input/scene.png"),
            Path("/arbitrary/input"),
            width=400,
            height=200,
        )

        self.assertEqual(result.source_key, "scene.png")
        self.assertEqual(result.perspective_regions[0].xyxy, (40.0, 40.0, 360.0, 160.0))
        self.assertAlmostEqual(result.perspective_regions[0].confidence, 0.9)
        self.assertEqual(result.shadow_pairs[0].object_contact, (100.0, 100.0))
        self.assertEqual(result.shadow_pairs[0].shadow_tip, (300.0, 200.0))
        self.assertEqual(result.reflection_pairs[0].object_point, (40.0, 40.0))
        self.assertAlmostEqual(result.shadow_pairs[0].confidence, 0.8)
        self.assertEqual(result.shadow_applicability, "applicable")
        self.assertEqual(result.reflection_applicability, "applicable")

    def test_annotation_directory_merges_per_image_exports(self) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        case_dir = TEST_TEMP_ROOT / uuid.uuid4().hex
        case_dir.mkdir(parents=True)
        first_payload = {
            "coordinate_space": "normalized",
            "images": {
                "first.png": {
                    "cast_shadow": {
                        "pairs": [
                            {
                                "object_contact": [0.1, 0.2],
                                "shadow_tip": [0.3, 0.4],
                            }
                        ]
                    }
                }
            },
        }
        second_payload = {
            "coordinate_space": "pixels",
            "images": {
                "second.png": {
                    "reflection": {
                        "pairs": [
                            {
                                "object_point": [20, 30],
                                "reflection_point": [80, 30],
                            }
                        ]
                    }
                }
            },
        }
        try:
            (case_dir / "first.json").write_text(
                json.dumps(first_payload), encoding="utf-8"
            )
            (case_dir / "second.json").write_text(
                json.dumps(second_payload), encoding="utf-8"
            )

            store = AnnotationStore.from_path(case_dir)
            first = store.for_image(
                case_dir / "first.png", case_dir, width=200, height=100
            )
            second = store.for_image(
                case_dir / "second.png", case_dir, width=200, height=100
            )

            self.assertEqual(first.shadow_pairs[0].object_contact, (20.0, 20.0))
            self.assertEqual(second.reflection_pairs[0].object_point, (20.0, 30.0))
        finally:
            shutil.rmtree(case_dir)

    def test_annotation_directory_rejects_duplicate_image_keys(self) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        case_dir = TEST_TEMP_ROOT / uuid.uuid4().hex
        case_dir.mkdir(parents=True)
        duplicate = {
            "coordinate_space": "normalized",
            "images": {"same.png": {}},
        }
        try:
            (case_dir / "a.json").write_text(json.dumps(duplicate), encoding="utf-8")
            (case_dir / "b.json").write_text(json.dumps(duplicate), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Duplicate annotation key"):
                AnnotationStore.from_path(case_dir)
        finally:
            shutil.rmtree(case_dir)

    def test_pixel_coordinates_must_stay_inside_image(self) -> None:
        store = AnnotationStore(
            {
                "coordinate_space": "pixels",
                "images": {
                    "scene.png": {
                        "cast_shadow": {
                            "pairs": [
                                {
                                    "object_contact": [10, 10],
                                    "shadow_tip": [101, 20],
                                }
                            ]
                        }
                    }
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "within the image bounds"):
            store.for_image("scene.png", ".", width=100, height=100)


class EngineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        if TEST_TEMP_ROOT.exists():
            shutil.rmtree(TEST_TEMP_ROOT)

    def setUp(self) -> None:
        self.case_dir = TEST_TEMP_ROOT / uuid.uuid4().hex
        self.case_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.case_dir.exists():
            shutil.rmtree(self.case_dir)

    def test_blank_image_has_no_applicable_cues(self) -> None:
        image_path = self.case_dir / "blank.png"
        Image.new("RGB", (320, 240), (120, 120, 120)).save(image_path)

        result = PhysicsEngine().run(self.case_dir)

        self.assertEqual(result.summary["processed_images"], 1)
        self.assertEqual(result.summary["images_with_errors"], 0)
        image_result = result.images[0]
        self.assertEqual(image_result.physics.status, "indeterminate")
        self.assertIsNone(image_result.physics.violation_score)
        self.assertEqual(
            image_result.physics.score_kind,
            "physics_violation_not_aigc_probability",
        )
        self.assertTrue(
            all(not cue.applicable for cue in image_result.cues.values())
        )

    def test_reviewed_perspective_region_reaches_analyzer(self) -> None:
        image_path = self.case_dir / "region.png"
        annotation_path = self.case_dir / "annotations.json"
        Image.new("RGB", (200, 100), "white").save(image_path)
        annotation_path.write_text(
            json.dumps(
                {
                    "coordinate_space": "normalized",
                    "images": {
                        "region.png": {
                            "perspective": {
                                "regions": [
                                    {"xyxy": [0.1, 0.2, 0.8, 0.9]},
                                    {
                                        "xyxy": [0.0, 0.0, 0.2, 0.2],
                                        "confidence": 0.2,
                                    },
                                ]
                            }
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        with patch("physics_engine.engine.analyze_perspective") as analyzer:
            from physics_engine.schema import not_applicable

            analyzer.return_value = not_applicable("perspective", "fixture")
            PhysicsEngine().run(image_path, annotations_path=annotation_path)

        self.assertEqual(analyzer.call_args.args[2], [(20.0, 20.0, 160.0, 90.0)])

    def test_explicit_not_applicable_overrides_retained_shadow_pairs(self) -> None:
        image_path = self.case_dir / "reviewed.png"
        annotation_path = self.case_dir / "reviewed.json"
        Image.new("RGB", (200, 100), "white").save(image_path)
        pairs = [
            {
                "object_contact": [0.1, y],
                "shadow_tip": [0.8, 1.0 - y],
            }
            for y in (0.2, 0.5, 0.8)
        ]
        annotation_path.write_text(
            json.dumps(
                {
                    "images": {
                        "reviewed.png": {
                            "cast_shadow": {
                                "applicability": "not_applicable",
                                "pairs": pairs,
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        result = PhysicsEngine().run(image_path, annotations_path=annotation_path)

        shadow = result.images[0].cues["cast_shadow"]
        self.assertFalse(shadow.applicable)
        self.assertEqual(shadow.status, "not_applicable")
        self.assertEqual(
            shadow.measurements["review_applicability"], "not_applicable"
        )

    def test_cli_writes_machine_readable_result(self) -> None:
        image_path = self.case_dir / "blank.png"
        output_path = self.case_dir / "result.json"
        Image.new("RGB", (180, 120), (80, 100, 120)).save(image_path)

        exit_code = cli_main(
            [str(image_path), "--output", str(output_path), "--pretty", "--strict"]
        )

        self.assertEqual(exit_code, 0)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "0.1.0")
        self.assertEqual(payload["summary"]["processed_images"], 1)
        self.assertEqual(
            payload["images"][0]["physics"]["score_kind"],
            "physics_violation_not_aigc_probability",
        )

    def test_exif_orientation_is_applied_before_analysis(self) -> None:
        image_path = self.case_dir / "rotated.jpg"
        image = Image.new("RGB", (80, 40), (100, 120, 140))
        exif = Image.Exif()
        exif[274] = 6  # display as 90 degrees clockwise
        image.save(image_path, exif=exif)

        result = PhysicsEngine().run(image_path)

        self.assertEqual(result.images[0].width, 40)
        self.assertEqual(result.images[0].height, 80)

    def test_recursive_run_excludes_existing_overlay_directory(self) -> None:
        image_path = self.case_dir / "scene.png"
        overlay_dir = self.case_dir / "overlays"
        overlay_dir.mkdir()
        Image.new("RGB", (160, 120), (100, 100, 100)).save(image_path)
        Image.new("RGB", (160, 120), (200, 20, 20)).save(
            overlay_dir / "stale_overlay.png"
        )

        result = PhysicsEngine().run(
            self.case_dir, overlays_dir=overlay_dir, recursive=True
        )

        self.assertEqual(result.summary["processed_images"], 1)
        self.assertEqual(Path(result.images[0].image_path).name, "scene.png")

    def test_overlay_directory_cannot_equal_input_directory(self) -> None:
        Image.new("RGB", (80, 80), (100, 100, 100)).save(self.case_dir / "scene.png")

        with self.assertRaisesRegex(ValueError, "cannot be the input directory"):
            PhysicsEngine().run(self.case_dir, overlays_dir=self.case_dir)

    def test_corrupt_image_is_isolated_as_an_error_result(self) -> None:
        image_path = self.case_dir / "corrupt.png"
        image_path.write_bytes(b"this is not a png")

        result = PhysicsEngine().run(image_path)

        self.assertEqual(result.summary["processed_images"], 1)
        self.assertEqual(result.summary["images_with_errors"], 1)
        self.assertEqual(result.images[0].cues["perspective"].status, "error")
        self.assertIsNone(result.images[0].physics.violation_score)

    def test_tiny_rgba_image_degrades_to_not_applicable_without_error(self) -> None:
        image_path = self.case_dir / "tiny.png"
        Image.new("RGBA", (1, 1), (10, 20, 30, 40)).save(image_path)

        result = PhysicsEngine().run(image_path)

        self.assertEqual(result.summary["images_with_errors"], 0)
        self.assertEqual(result.images[0].width, 1)
        self.assertEqual(result.images[0].cues["perspective"].status, "not_applicable")

    def test_cli_rejects_empty_image_directory(self) -> None:
        output_path = self.case_dir / "empty.json"

        exit_code = cli_main(
            [str(self.case_dir), "--output", str(output_path), "--strict"]
        )

        self.assertEqual(exit_code, 2)
        self.assertFalse(output_path.exists())

    def test_cli_reports_invalid_proposal_threshold_without_traceback(self) -> None:
        image_path = self.case_dir / "blank.png"
        output_path = self.case_dir / "invalid.json"
        Image.new("RGB", (80, 80), "white").save(image_path)

        exit_code = cli_main(
            [
                str(image_path),
                "--auto-proposals",
                "--proposal-shadow-threshold",
                "1.5",
                "--output",
                str(output_path),
            ]
        )

        self.assertEqual(exit_code, 2)
        self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
