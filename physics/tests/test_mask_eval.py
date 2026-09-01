from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from physics_engine.mask_eval import (
    _sample_indices,
    _sbu_zip_entries,
    _selected_sbu_rows,
    binary_mask_metrics,
)


class MaskEvaluationTests(unittest.TestCase):
    def test_binary_metrics_distinguish_false_positive_and_miss(self) -> None:
        target = np.asarray([[1, 1], [0, 0]], dtype=bool)
        probability = np.asarray([[0.9, 0.2], [0.8, 0.1]], dtype=np.float32)

        metrics = binary_mask_metrics(probability, target, 0.5)

        self.assertEqual(metrics.intersection, 1)
        self.assertEqual(metrics.union, 3)
        self.assertAlmostEqual(metrics.iou, 1 / 3)
        self.assertAlmostEqual(metrics.dice, 0.5)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)

    def test_seeded_sampling_is_bounded_sorted_and_repeatable(self) -> None:
        first = _sample_indices(571, 24, 2026)
        second = _sample_indices(571, 24, 2026)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(set(first)), 24)

    def test_sbu_zip_reader_selects_pairs_without_extraction(self) -> None:
        image_bytes = BytesIO()
        mask_bytes = BytesIO()
        Image.new("RGB", (4, 3), "white").save(image_bytes, format="JPEG")
        Image.new("L", (4, 3), 255).save(mask_bytes, format="PNG")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sbu.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "SBU-shadow/SBU-Test/ShadowImages/example.jpg",
                    image_bytes.getvalue(),
                )
                archive.writestr(
                    "SBU-shadow/SBU-Test/ShadowMasks/example.png",
                    mask_bytes.getvalue(),
                )

            entries = _sbu_zip_entries(path)
            rows = list(_selected_sbu_rows(path, entries, [0]))

            self.assertEqual(entries[0][0], "example")
            self.assertEqual(rows[0]["image_id"], "example")
            self.assertTrue(rows[0]["image"]["bytes"].startswith(b"\xff\xd8"))


if __name__ == "__main__":
    unittest.main()
