from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from physics_engine.line_detection import LineDetectionConfig, detect_line_segments


class LineDetectionRegionTests(unittest.TestCase):
    def test_reviewed_region_filters_detected_lines(self) -> None:
        image = Image.new("RGB", (240, 180), "white")
        draw = ImageDraw.Draw(image)
        for y in (30, 60, 90, 120, 150):
            draw.line((10, y, 105, y), fill="black", width=3)
            draw.line((135, y, 230, y), fill="black", width=3)

        result = detect_line_segments(
            image,
            LineDetectionConfig(min_length_ratio=0.02, denoise_sigma=0.0),
            regions=[(0.0, 0.0, 120.0, 180.0)],
        )

        self.assertTrue(result.segments)
        self.assertTrue(result.diagnostics["reviewed_region_filter_applied"])
        self.assertEqual(result.diagnostics["reviewed_region_count"], 1)
        for x1, _y1, x2, _y2 in result.segments:
            self.assertLessEqual((x1 + x2) / 2.0, 120.0)


if __name__ == "__main__":
    unittest.main()
