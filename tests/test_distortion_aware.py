import random
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "patchhead"))

from model import (  # noqa: E402
    ANALYTIC_FEATURE_DIM,
    DistortionHead,
    DistortionThresholdAdapter,
    analytic_distortion_features,
)
from data import ImageDataset  # noqa: E402
from transforms import (  # noqa: E402
    DISTORTION_TARGET_DIM,
    MAGNITUDE_NAMES,
    TYPE_NAMES,
    apply_with_spec,
    random_train_transform,
)


class DistortionTransformTests(unittest.TestCase):
    def setUp(self):
        grid = np.indices((64, 64)).sum(axis=0) % 2
        self.image = Image.fromarray(np.repeat((grid * 255).astype(np.uint8)[..., None], 3, axis=2))

    def test_named_noise_is_pixelwise_and_deterministic(self):
        first, first_spec = apply_with_spec(self.image, "noise0.10", rng=random.Random(7))
        second, second_spec = apply_with_spec(self.image, "noise0.10", rng=random.Random(7))
        delta = np.asarray(first, dtype=np.float32) - np.asarray(self.image, dtype=np.float32)
        self.assertTrue(np.array_equal(np.asarray(first), np.asarray(second)))
        self.assertGreater(float(delta.std()), 5.0)
        self.assertEqual(first_spec.types["noise"], 1.0)
        self.assertAlmostEqual(first_spec.magnitudes["noise_sigma"], 1.0)
        np.testing.assert_array_equal(first_spec.target(), second_spec.target())

    def test_jpeg_and_clean_metadata(self):
        _, jpeg = apply_with_spec(self.image, "jpeg30")
        _, clean = apply_with_spec(self.image, "clean")
        self.assertEqual(jpeg.types["jpeg"], 1.0)
        self.assertAlmostEqual(jpeg.magnitudes["jpeg_severity"], 1.0)
        self.assertEqual(clean.target().shape, (DISTORTION_TARGET_DIM,))
        self.assertFalse(clean.target().any())

    def test_random_policy_is_reproducible(self):
        image_a, spec_a = random_train_transform(self.image, random.Random(123))
        image_b, spec_b = random_train_transform(self.image, random.Random(123))
        self.assertTrue(np.array_equal(np.asarray(image_a), np.asarray(image_b)))
        np.testing.assert_array_equal(spec_a.target(), spec_b.target())
        self.assertEqual(len(TYPE_NAMES) + len(MAGNITUDE_NAMES), DISTORTION_TARGET_DIM)

    def test_dataset_can_return_known_distortion_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "sample.png"
            manifest_path = root / "samples.csv"
            self.image.save(image_path)
            with manifest_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "label", "group_id"])
                writer.writeheader()
                writer.writerow({"image_path": image_path.name, "label": 0, "group_id": "sample"})
            dataset = ImageDataset(
                roots=[], split="test", transform_name="jpeg30",
                manifest=str(manifest_path), return_distortion=True,
                canon=64, size=64,
            )
            image, label, key, target = dataset[0]
            self.assertEqual(image.shape, (3, 64, 64))
            self.assertEqual(float(label), 0.0)
            self.assertEqual(key, "sample")
            self.assertEqual(float(target[TYPE_NAMES.index("jpeg")]), 1.0)


class DistortionModelTests(unittest.TestCase):
    def test_analytic_features_are_finite(self):
        features = analytic_distortion_features(torch.rand(3, 3, 64, 64))
        self.assertEqual(features.shape, (3, ANALYTIC_FEATURE_DIM))
        self.assertTrue(torch.isfinite(features).all())

    def test_estimator_shapes_and_zero_initial_threshold_shift(self):
        estimator = DistortionHead(dim=32, hidden=16).eval()
        types, magnitudes = estimator(
            torch.rand(4, 32), torch.rand(4, ANALYTIC_FEATURE_DIM))
        self.assertEqual(types.shape, (4, len(TYPE_NAMES)))
        self.assertEqual(magnitudes.shape, (4, len(MAGNITUDE_NAMES)))
        adapter = DistortionThresholdAdapter(hidden=8).eval()
        shift = adapter(torch.rand(4, DISTORTION_TARGET_DIM))
        torch.testing.assert_close(shift, torch.zeros_like(shift))


if __name__ == "__main__":
    unittest.main()
