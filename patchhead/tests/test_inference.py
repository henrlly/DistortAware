from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from patchhead.inference import (
    PatchHeadInferenceError,
    _require_pooled_checkpoint,
    _preflight_checkpoint_payload,
    run_patchhead_inference,
    write_json_atomic,
)


class FakeRuntime:
    def __init__(self, *, malformed: bool = False) -> None:
        self.metadata = {
            "family": "fake_patchhead_for_contract_tests",
            "arch": "fake",
            "threshold": 0.60,
            "model_input_size": 32,
            "score_formula": "test fixture",
        }
        self.malformed = malformed
        self.seen_shapes: list[tuple[int, ...]] = []

    def infer(self, batch: np.ndarray):
        self.seen_shapes.append(tuple(batch.shape))
        count = batch.shape[0]
        if self.malformed:
            return np.zeros(count + 1), np.zeros(count), np.zeros((count, 2, 3))
        patch = np.tile(
            np.asarray([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64),
            (count, 1, 1),
        )
        image = patch.reshape(count, -1).mean(axis=1)
        cls = np.full(count, 1.0, dtype=np.float64)
        return image, cls, patch


class FailingRuntime(FakeRuntime):
    def infer(self, batch: np.ndarray):
        raise RuntimeError("fixture failure")


class WrongArityRuntime(FakeRuntime):
    def infer(self, batch: np.ndarray):
        count = batch.shape[0]
        return np.zeros(count), np.zeros(count)


class FakeFeatureRuntime(FakeRuntime):
    def infer_with_features(self, batch: np.ndarray):
        image, cls, patch = self.infer(batch)
        count = batch.shape[0]
        features = np.zeros((count, 2, 3, 5), dtype=np.float32)
        features[..., 0] = 1.0
        features[..., 1] = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
        return image, cls, patch, features


class PatchHeadInferenceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="patchhead-contract-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _image(self, relative: str, *, size: tuple[int, int] = (80, 60)) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (40, 90, 150)).save(path)
        return path

    def test_compact_default_uses_official_score_formula(self) -> None:
        self._image("scene.png")
        runtime = FakeRuntime()

        payload = run_patchhead_inference(self.root, runtime=runtime, batch_size=1)

        record = payload["images"][0]
        patch_score = 1.0 / (1.0 + np.exp(-0.5))
        cls_score = 1.0 / (1.0 + np.exp(-1.0))
        self.assertAlmostEqual(record["aigc_score"], 0.5 * (patch_score + cls_score))
        self.assertTrue(record["is_aigc"])
        self.assertNotIn("patch_evidence", record)
        self.assertEqual(runtime.seen_shapes, [(1, 3, 32, 32)])

    def test_optional_patch_map_supports_non_square_future_grids(self) -> None:
        self._image("scene.png")

        payload = run_patchhead_inference(
            self.root,
            runtime=FakeRuntime(),
            export_patch_evidence=True,
        )

        evidence = payload["images"][0]["patch_evidence"]
        self.assertEqual(evidence["grid_shape"], [2, 3])
        self.assertEqual(evidence["coordinate_space"], "normalized_full_frame")
        self.assertEqual(len(evidence["values"]), 2)
        self.assertEqual(len(evidence["values"][0]), 3)

    def test_same_pass_dense_features_are_forwarded_in_memory_only(self) -> None:
        self._image("scene.png")
        received: dict[str, np.ndarray] = {}

        payload = run_patchhead_inference(
            self.root,
            runtime=FakeFeatureRuntime(),
            dense_feature_sink=lambda path, values: received.setdefault(path, values),
        )

        self.assertEqual(received["scene.png"].shape, (2, 3, 5))
        self.assertTrue(payload["detector"]["dense_features_forwarded_in_memory"])
        self.assertNotIn("dense_features", payload["images"][0])

    def test_dense_feature_request_fails_if_runtime_cannot_expose_grid(self) -> None:
        self._image("scene.png")
        with self.assertRaisesRegex(PatchHeadInferenceError, "cannot expose"):
            run_patchhead_inference(
                self.root,
                runtime=FakeRuntime(),
                dense_feature_sink=lambda _path, _values: None,
            )

    def test_recursive_paths_keep_duplicate_basenames_unambiguous(self) -> None:
        self._image("first/scene.png")
        self._image("second/scene.png")

        payload = run_patchhead_inference(self.root, runtime=FakeRuntime(), batch_size=2)

        self.assertEqual(
            [record["image_path"] for record in payload["images"]],
            ["first/scene.png", "second/scene.png"],
        )

    def test_preprocessing_flushes_in_bounded_batches(self) -> None:
        for index in range(5):
            self._image(f"scene-{index}.png")
        runtime = FakeRuntime()

        run_patchhead_inference(self.root, runtime=runtime, batch_size=2)

        self.assertEqual(
            runtime.seen_shapes,
            [(2, 3, 32, 32), (2, 3, 32, 32), (1, 3, 32, 32)],
        )

    def test_corrupt_image_isolated_and_valid_image_still_scores(self) -> None:
        (self.root / "bad.png").write_bytes(b"not a png")
        self._image("good.png")

        payload = run_patchhead_inference(self.root, runtime=FakeRuntime())

        by_name = {record["image_path"]: record for record in payload["images"]}
        self.assertIsNone(by_name["bad.png"]["aigc_score"])
        self.assertIn("Could not decode", by_name["bad.png"]["error"])
        self.assertNotIn("patch_evidence", by_name["bad.png"])
        self.assertIsInstance(by_name["good.png"]["aigc_score"], float)
        self.assertEqual(payload["summary"]["decode_failures"], 1)

    def test_exif_orientation_is_recorded_in_display_coordinates(self) -> None:
        path = self.root / "rotated.jpg"
        image = Image.new("RGB", (80, 40), (10, 20, 30))
        exif = Image.Exif()
        exif[274] = 6
        image.save(path, exif=exif)

        payload = run_patchhead_inference(path, runtime=FakeRuntime())

        self.assertEqual(payload["images"][0]["width"], 40)
        self.assertEqual(payload["images"][0]["height"], 80)

    def test_invalid_runtime_shape_fails_closed(self) -> None:
        self._image("scene.png")
        with self.assertRaisesRegex(PatchHeadInferenceError, "Image logits"):
            run_patchhead_inference(self.root, runtime=FakeRuntime(malformed=True))

    def test_runtime_failure_is_wrapped_with_batch_context(self) -> None:
        self._image("scene.png")
        with self.assertRaisesRegex(PatchHeadInferenceError, "batch of 1 image"):
            run_patchhead_inference(self.root, runtime=FailingRuntime())

    def test_runtime_must_return_all_three_logit_tensors(self) -> None:
        self._image("scene.png")
        with self.assertRaisesRegex(PatchHeadInferenceError, "exactly image, CLS"):
            run_patchhead_inference(self.root, runtime=WrongArityRuntime())

    def test_checkpoint_requirement_is_explicit(self) -> None:
        self._image("scene.png")
        with self.assertRaisesRegex(PatchHeadInferenceError, "checkpoint is required"):
            run_patchhead_inference(self.root)

    def test_primary_runtime_rejects_non_pooled_checkpoint_metadata(self) -> None:
        self.assertEqual(_require_pooled_checkpoint({"ds": "pooled"}), "pooled")
        with self.assertRaisesRegex(PatchHeadInferenceError, "requires the pooled"):
            _require_pooled_checkpoint({"ds": "wildfake"})

    def test_checkpoint_preflight_rejects_incomplete_metadata(self) -> None:
        valid = _preflight_checkpoint_payload(
            {
                "ds": "pooled",
                "model": {"head.weight": object()},
                "threshold": 0.42,
                "size": 256,
            }
        )
        self.assertEqual(valid["dataset"], "pooled")
        self.assertEqual(valid["model_size"], 256)
        binary = _preflight_checkpoint_payload(
            {
                "ds": "pooled",
                "model": {
                    "head.patch_logit.weight": np.zeros((1, 2, 1, 1))
                },
                "threshold": 0.42,
                "size": 256,
            }
        )
        three_class = _preflight_checkpoint_payload(
            {
                "ds": "pooled",
                "model": {
                    "head.patch_logit.weight": np.zeros((3, 2, 1, 1))
                },
                "threshold": 0.42,
                "size": 256,
            }
        )
        self.assertEqual(binary["output_classes"], 1)
        self.assertEqual(three_class["output_classes"], 3)
        with self.assertRaisesRegex(PatchHeadInferenceError, "no trainable model state"):
            _preflight_checkpoint_payload(
                {"ds": "pooled", "model": {}, "threshold": 0.5, "size": 256}
            )

    def test_atomic_writer_round_trips_payload(self) -> None:
        output = self.root / "nested" / "result.json"
        write_json_atomic(output, {"ok": True}, pretty=True)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"ok": True})
        self.assertFalse(output.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
