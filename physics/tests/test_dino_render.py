from __future__ import annotations

from pathlib import Path
import shutil
import unittest
import uuid

from PIL import Image

from physics_engine.dino_render import DinoRenderError, render_dino_physics_panel


TEST_TEMP_ROOT = Path(__file__).resolve().parent / ".tmp"


class DinoRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.case_dir = TEST_TEMP_ROOT / uuid.uuid4().hex
        self.case_dir.mkdir(parents=True)
        self.image_path = self.case_dir / "scene.png"
        Image.new("RGB", (80, 60), "white").save(self.image_path)
        self.dino = {
            "patch_evidence": {
                "coordinate_space": "normalized_full_frame",
                "grid_shape": [2, 2],
                "values": [[0.1, 0.9], [0.2, 0.8]],
            }
        }
        self.physics = {
            "width": 80,
            "height": 60,
            "cues": {
                "perspective": {
                    "status": "inconsistent",
                    "evidence": [
                        {
                            "kind": "line_segment",
                            "xyxy": [5, 5, 75, 55],
                            "role": "outlier",
                        }
                    ],
                }
            },
        }

    def tearDown(self) -> None:
        if self.case_dir.exists():
            shutil.rmtree(self.case_dir)

    def test_three_panel_visual_is_written(self) -> None:
        output = self.case_dir / "panel.png"
        report = render_dino_physics_panel(
            self.image_path, self.dino, self.physics, output
        )

        self.assertTrue(output.is_file())
        with Image.open(output) as image:
            self.assertEqual(image.size, (80 * 3 + 16, 60 + 34))
        self.assertEqual(report["physics_outlier_segments_drawn"], 1)

    def test_unknown_coordinate_space_is_rejected(self) -> None:
        self.dino["patch_evidence"]["coordinate_space"] = "center_crop"
        with self.assertRaisesRegex(DinoRenderError, "normalized_full_frame"):
            render_dino_physics_panel(
                self.image_path, self.dino, self.physics, self.case_dir / "bad.png"
            )


if __name__ == "__main__":
    unittest.main()
