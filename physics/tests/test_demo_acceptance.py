from __future__ import annotations

from pathlib import Path
import shutil
import unittest
import uuid

from examples.generate_demo_scenes import (
    annotated_scene,
    no_geometry,
    perspective_consistent,
    perspective_inconsistent,
)
from physics_engine.engine import PhysicsEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TEMP_ROOT = Path(__file__).resolve().parent / ".tmp"


class DemoAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.case_dir = TEST_TEMP_ROOT / uuid.uuid4().hex
        self.case_dir.mkdir(parents=True)
        perspective_consistent(self.case_dir / "perspective_consistent.png")
        perspective_inconsistent(self.case_dir / "perspective_inconsistent.png")
        annotated_scene(self.case_dir / "annotated_consistent.png", inconsistent=False)
        annotated_scene(self.case_dir / "annotated_inconsistent.png", inconsistent=True)
        no_geometry(self.case_dir / "no_geometry.png")

    def tearDown(self) -> None:
        if self.case_dir.exists():
            shutil.rmtree(self.case_dir)

    def test_demo_scenes_exercise_expected_decision_paths(self) -> None:
        overlays = self.case_dir / "overlays"
        result = PhysicsEngine().run(
            self.case_dir,
            annotations_path=PROJECT_ROOT / "examples" / "demo_annotations.json",
            overlays_dir=overlays,
        )
        by_name = {Path(image.image_path).name: image for image in result.images}

        self.assertEqual(result.summary["processed_images"], 5)
        self.assertEqual(result.summary["images_with_errors"], 0)

        consistent = by_name["annotated_consistent.png"]
        self.assertEqual(consistent.cues["cast_shadow"].status, "consistent")
        self.assertEqual(consistent.cues["reflection"].status, "consistent")

        inconsistent = by_name["annotated_inconsistent.png"]
        self.assertEqual(inconsistent.cues["cast_shadow"].status, "inconsistent")
        self.assertEqual(inconsistent.cues["reflection"].status, "inconsistent")

        perspective_ok = by_name["perspective_consistent.png"].cues["perspective"]
        self.assertEqual(perspective_ok.status, "consistent")
        self.assertLessEqual(perspective_ok.violation_score or 0.0, 0.1)

        perspective_mixed = by_name["perspective_inconsistent.png"].cues["perspective"]
        self.assertIn(perspective_mixed.status, {"indeterminate", "inconsistent"})
        self.assertIsNotNone(perspective_mixed.violation_score)
        assert perspective_mixed.violation_score is not None
        self.assertGreater(perspective_mixed.violation_score, 0.3)

        no_structure = by_name["no_geometry.png"]
        self.assertEqual(no_structure.cues["perspective"].status, "not_applicable")
        self.assertIsNone(no_structure.physics.violation_score)

        rendered = list(overlays.glob("*.png"))
        self.assertEqual(len(rendered), 8)


if __name__ == "__main__":
    unittest.main()
