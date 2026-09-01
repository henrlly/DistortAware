from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from browser_product.backend import (
    BackendConfig,
    BrowserInferenceBackend,
    BrowserInferenceError,
    DemoFixtureBrowserInferenceBackend,
    PrismGuardBrowserInferenceBackend,
)
from patchhead.tests.test_inference import FakeFeatureRuntime


def _png_bytes(size: tuple[int, int] = (96, 72)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, (90, 110, 140)).save(output, format="PNG")
    return output.getvalue()


class FakeArtifactPredictor:
    def predict_path(self, _path):
        return (
            {
                "prediction": "ai_tampered",
                "binary_prediction": "ai",
                "confidence": 0.72,
                "ai_probability": 0.91,
                "class_probabilities": {
                    "authentic": 0.09,
                    "fully_synthetic": 0.19,
                    "ai_tampered": 0.72,
                },
            },
            np.pad(np.ones((4, 4), dtype=np.float32), 2),
        )


class FakePrismGuardDetector:
    payload = {
        "extractor": {
            "type": "frozen_vfm",
            "registry_name": "dinov3_vitl16",
            "checkpoint_sha256": "a" * 64,
            "input_size": 384,
        },
        "metadata": {"training_manifest_sha256": "b" * 64},
    }
    prediction_contract = {"score_source": "calibrated_dino_logit_only"}

    def score_images_with_trace(self, images, *, batch_size):
        assert len(images) == 1
        assert batch_size == 1
        return SimpleNamespace(
            raw_logit=np.asarray([1.5]),
            calibrated_logit=np.asarray([0.75]),
            pred=np.asarray([0.679178699175393]),
            score_source="calibrated_dino_logit_only",
        )


class BrowserInferenceBackendTests(unittest.TestCase):
    def test_prismguard_adapter_rejects_engineering_smoke_bundle(self) -> None:
        smoke = FakePrismGuardDetector()
        smoke.payload = {
            "extractor": {"type": "numpy_smoke"},
            "metadata": {"scientific_status": "plumbing_only_no_aigc_performance_claim"},
        }

        with self.assertRaisesRegex(ValueError, "smoke bundles are prohibited"):
            PrismGuardBrowserInferenceBackend(
                detector=smoke,
                config=BackendConfig(physics_profile="off", artifact_profile="off"),
            )

    def test_demo_fixture_is_explicit_and_deterministic(self) -> None:
        backend = DemoFixtureBrowserInferenceBackend()

        first = backend.analyze(_png_bytes(), media_type="image/png")
        second = backend.analyze(
            _png_bytes(),
            media_type="image/png",
            include_physics=True,
            include_artifacts=True,
        )

        self.assertEqual(first["verdict"], second["verdict"])
        self.assertEqual(
            first["scientific_status"],
            "plumbing_only_no_aigc_performance_claim",
        )
        self.assertIn("WIRING DEMO ONLY", first["limitations"][0])
        self.assertFalse(second["explanation"]["physics_affects_detector_score"])
        self.assertFalse(second["explanation"]["artifact_affects_detector_score"])

    def test_prismguard_adapter_keeps_diagnostics_out_of_prediction(self) -> None:
        backend = PrismGuardBrowserInferenceBackend(
            detector=FakePrismGuardDetector(),
            config=BackendConfig(
                physics_profile="off", artifact_profile="off", cache_entries=0
            ),
        )

        primary = backend.analyze(_png_bytes(), media_type="image/png")
        challenged = backend.analyze(
            _png_bytes(),
            media_type="image/png",
            include_physics=True,
            include_artifacts=True,
        )

        self.assertEqual(primary["verdict"], challenged["verdict"])
        self.assertEqual(
            primary["verdict"]["aigc_score"], 0.679178699175393
        )
        self.assertEqual(
            primary["score_trace"]["prediction_source"],
            "calibrated_dino_logit_only",
        )
        self.assertIsNone(challenged["explanation"]["physics"])
        self.assertIsNone(challenged["explanation"]["artifact"])
        self.assertFalse(
            challenged["explanation"]["physics_affects_detector_score"]
        )
        self.assertFalse(
            challenged["explanation"]["artifact_affects_detector_score"]
        )

    def test_prismguard_adapter_rejects_coupled_diagnostics_profiles(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate from prediction"):
            PrismGuardBrowserInferenceBackend(
                detector=FakePrismGuardDetector(),
                config=BackendConfig(physics_profile="heuristic"),
            )

    def test_primary_result_and_cache_preserve_score_contract(self) -> None:
        backend = BrowserInferenceBackend(
            runtime=FakeFeatureRuntime(),
            config=BackendConfig(physics_profile="off", cache_entries=4),
        )

        first = backend.analyze(
            _png_bytes(), media_type="image/png", source_kind="test_fixture"
        )
        second = backend.analyze(
            _png_bytes(), media_type="image/png", source_kind="test_fixture"
        )

        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["verdict"], second["verdict"])
        self.assertTrue(first["verdict"]["is_aigc"])
        self.assertEqual(first["image"]["source_kind"], "test_fixture")
        self.assertEqual(
            first["explanation"]["patch_evidence"]["grid_shape"], [2, 3]
        )
        self.assertIsNone(first["explanation"]["physics"])

    def test_opt_in_physics_never_changes_primary_verdict(self) -> None:
        backend = BrowserInferenceBackend(
            runtime=FakeFeatureRuntime(),
            config=BackendConfig(physics_profile="heuristic", cache_entries=0),
        )

        primary = backend.analyze(_png_bytes(), media_type="image/png")
        explained = backend.analyze(
            _png_bytes(), media_type="image/png", include_physics=True
        )

        self.assertEqual(primary["verdict"], explained["verdict"])
        self.assertIsNotNone(explained["explanation"]["physics"])
        self.assertIsNotNone(
            explained["explanation"]["dino_physics_alignment"]
        )
        self.assertNotIn(
            "details_image_path", explained["explanation"]["physics"]
        )
        for cue in explained["explanation"]["physics"]["cues"].values():
            self.assertNotIn("overlay_path", cue)
            for item in cue.get("evidence", []):
                if isinstance(item, dict):
                    self.assertNotIn("contour", item)
        self.assertFalse(
            explained["explanation"]["physics_affects_detector_score"]
        )

    def test_invalid_type_and_decoded_pixel_limit_fail_closed(self) -> None:
        backend = BrowserInferenceBackend(
            runtime=FakeFeatureRuntime(),
            config=BackendConfig(max_pixels=100, cache_entries=0),
        )

        with self.assertRaisesRegex(BrowserInferenceError, "Content-Type"):
            backend.analyze(_png_bytes(), media_type="application/octet-stream")
        with self.assertRaisesRegex(BrowserInferenceError, "pixel limit"):
            backend.analyze(_png_bytes((20, 20)), media_type="image/png")

    def test_physics_request_is_not_hidden_by_detector_only_cache(self) -> None:
        backend = BrowserInferenceBackend(
            runtime=FakeFeatureRuntime(),
            config=BackendConfig(physics_profile="off", cache_entries=4),
        )

        backend.analyze(_png_bytes(), media_type="image/png")
        requested = backend.analyze(
            _png_bytes(), media_type="image/png", include_physics=True
        )

        self.assertTrue(requested["explanation"]["physics_requested"])
        self.assertTrue(
            any("physics_profile=off" in item for item in requested["limitations"])
        )

    def test_artifact_sidecar_is_explanation_only_and_cache_separated(self) -> None:
        backend = BrowserInferenceBackend(
            runtime=FakeFeatureRuntime(),
            artifact_predictor=FakeArtifactPredictor(),
            config=BackendConfig(artifact_profile="residual", cache_entries=4),
        )

        primary = backend.analyze(_png_bytes(), media_type="image/png")
        explained = backend.analyze(
            _png_bytes(), media_type="image/png", include_artifacts=True
        )

        self.assertEqual(primary["verdict"], explained["verdict"])
        artifact = explained["explanation"]["artifact"]
        self.assertEqual(artifact["status"], "available")
        self.assertEqual(artifact["predicted_class"], "ai_tampered")
        self.assertAlmostEqual(artifact["ai_signal_score"], 0.91)
        self.assertEqual(artifact["evidence_mask"]["shape"], [8, 8])
        self.assertEqual(len(artifact["evidence_mask"]["coarse_grid_8x8"]), 8)
        self.assertFalse(
            explained["explanation"]["artifact_affects_detector_score"]
        )
        self.assertFalse(explained["cache_hit"])


if __name__ == "__main__":
    unittest.main()
