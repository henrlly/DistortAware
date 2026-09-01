from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
from PIL import Image

from physics_engine.automatic import (
    AutomaticCueProposals,
    AutomaticProposalConfig,
    AutomaticProposalEngine,
    AutomaticProposalError,
    DenseFeatures,
    ObjectDetection,
    SemanticMasks,
    _fuse_semantic_and_physical_masks,
)
from physics_engine.annotations import ShadowPair
from physics_engine.engine import PhysicsEngine, PhysicsEngineConfig
from physics_engine.reflection import analyze_reflections
from physics_engine.schema import CueResult
from physics_engine.shadow import analyze_cast_shadows


TEST_TEMP_ROOT = Path(__file__).resolve().parent / ".tmp"


class StaticMaskProvider:
    def __init__(self, shadow: np.ndarray, mirror: np.ndarray) -> None:
        self.shadow = shadow
        self.mirror = mirror

    def predict(self, image: Image.Image) -> SemanticMasks:
        return SemanticMasks(
            shadow=self.shadow,
            mirror=self.mirror,
            backend="test_semantic_masks",
            model="deterministic-fixture",
            metadata={"learned": True},
        )


class StaticFeatureProvider:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def extract(self, image: Image.Image) -> DenseFeatures:
        return DenseFeatures(
            values=self.values,
            backend="test_dense_features",
            model="deterministic-fixture",
            metadata={"shared_primary_forward": False},
        )


class StaticObjectProvider:
    def __init__(self, detections: list[ObjectDetection]) -> None:
        self.detections = detections

    def detect(self, image: Image.Image):
        return self.detections, {"backend": "test_object_boxes", "learned": True}


def _fixture_masks(width: int = 320, height: int = 220):
    shadow = np.zeros((height, width), dtype=np.float32)
    objects: list[ObjectDetection] = []
    for index, (x, y) in enumerate(((35, 145), (95, 140), (155, 150), (215, 142))):
        polygon = np.asarray(
            [[x, y], [x + 12, y], [x + 62, y + 34], [x + 50, y + 39]],
            dtype=np.int32,
        )
        import cv2

        cv2.fillConvexPoly(shadow, polygon, 0.96)
        objects.append(
            ObjectDetection(
                xyxy=(x - 8.0, y - 45.0, x + 14.0, y + 2.0),
                confidence=0.94,
                label=f"fixture-{index}",
            )
        )
    mirror = np.zeros((height, width), dtype=np.float32)
    mirror[20:200, 180:305] = 0.97
    return shadow, mirror, objects


def _fixture_features() -> np.ndarray:
    values = np.zeros((11, 16, 12), dtype=np.float32)
    pairs = [((2, 3), (2, 11)), ((4, 4), (4, 12)), ((6, 3), (6, 12)), ((8, 4), (8, 11))]
    for index, (outside, inside) in enumerate(pairs):
        descriptor = np.zeros(12, dtype=np.float32)
        descriptor[index] = 3.0
        descriptor[4 + index] = 1.3
        values[outside] = descriptor
        values[inside] = descriptor + 0.001
    return values


class AutomaticProposalTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.case_dir = TEST_TEMP_ROOT / uuid.uuid4().hex
        self.case_dir.mkdir(parents=True)
        self.image = Image.new("RGB", (320, 220), (175, 185, 195))
        self.shadow, self.mirror, self.objects = _fixture_masks()
        self.config = AutomaticProposalConfig(
            enabled=True,
            shadow_threshold=0.5,
            mirror_threshold=0.5,
            min_shadow_pair_confidence=0.25,
            min_reflection_pair_confidence=0.25,
            min_feature_similarity=0.8,
            min_feature_margin=0.08,
        )

    def tearDown(self) -> None:
        if self.case_dir.exists():
            shutil.rmtree(self.case_dir)

    def _engine(self) -> AutomaticProposalEngine:
        return AutomaticProposalEngine(
            self.config,
            mask_provider=StaticMaskProvider(self.shadow, self.mirror),
            feature_provider=StaticFeatureProvider(_fixture_features()),
            object_provider=StaticObjectProvider(self.objects),
        )

    def test_semantic_mask_fusion_retains_strong_physical_priors(self) -> None:
        semantic_shadow = np.asarray([[0.2, 0.8]], dtype=np.float32)
        semantic_mirror = np.asarray([[0.1, 0.9]], dtype=np.float32)
        photometric_shadow = np.asarray([[0.7, 0.1]], dtype=np.float32)
        geometric_mirror = np.asarray([[0.8, 0.1]], dtype=np.float32)

        shadow, mirror = _fuse_semantic_and_physical_masks(
            semantic_shadow,
            semantic_mirror,
            photometric_shadow,
            geometric_mirror,
        )

        np.testing.assert_array_less(photometric_shadow - 1e-7, shadow)
        np.testing.assert_array_less(0.8 * geometric_mirror - 1e-7, mirror)
        self.assertTrue(np.all((0.0 <= shadow) & (shadow <= 1.0)))
        self.assertTrue(np.all((0.0 <= mirror) & (mirror <= 1.0)))

    def test_custom_model_ids_do_not_inherit_unrelated_default_revisions(self) -> None:
        config = AutomaticProposalConfig(
            mask_model="example/custom-mask",
            dino_model="custom_dino_model",
        )

        self.assertIsNone(config.mask_revision)
        self.assertIsNone(config.dino_revision)

    def test_shadow_components_become_object_contact_tip_pairs(self) -> None:
        proposals = self._engine().propose(self.image).shadow

        self.assertTrue(proposals.applicable)
        self.assertGreaterEqual(len(proposals.pairs), 3)
        result = analyze_cast_shadows(proposals.pairs, 320, 220)
        self.assertEqual(result.status, "consistent")
        self.assertTrue(
            any(item["kind"] == "shadow_region" for item in proposals.evidence)
        )

    def test_mirror_mask_and_dense_features_become_reflection_pairs(self) -> None:
        proposals = self._engine().propose(self.image).reflection

        self.assertTrue(proposals.applicable)
        self.assertGreaterEqual(len(proposals.pairs), 3)
        result = analyze_reflections(proposals.pairs, 320, 220)
        self.assertEqual(result.status, "consistent")
        matches = [
            item
            for item in proposals.evidence
            if item["kind"] == "reflection_pair_proposal"
        ]
        self.assertTrue(all(item["mutual_nearest_neighbour"] for item in matches))

    def test_insufficient_external_reflection_features_can_use_appearance_fallback(self) -> None:
        config = AutomaticProposalConfig(
            enabled=True,
            shadow_threshold=0.5,
            mirror_threshold=0.5,
            min_shadow_pair_confidence=0.25,
            min_reflection_pair_confidence=0.25,
            min_feature_similarity=0.8,
            min_feature_margin=0.08,
            feature_backend="appearance",
            appearance_fallback_on_insufficient_external=True,
        )
        engine = AutomaticProposalEngine(
            config,
            mask_provider=StaticMaskProvider(self.shadow, self.mirror),
            feature_provider=StaticFeatureProvider(_fixture_features()),
            object_provider=StaticObjectProvider(self.objects),
        )

        reflection = engine.propose(
            self.image,
            external_features=np.zeros((11, 16, 12), dtype=np.float32),
            include_shadow=False,
        ).reflection

        self.assertTrue(reflection.applicable)
        selection = reflection.measurements["feature_selection"]
        self.assertEqual(selection["primary_accepted_pairs"], 0)
        self.assertGreaterEqual(selection["fallback_accepted_pairs"], 3)
        self.assertIn("fallback", reflection.warnings[0].lower())

    def test_engine_uses_automatic_pairs_and_records_provenance(self) -> None:
        path = self.case_dir / "scene.png"
        self.image.save(path)
        engine = PhysicsEngine(
            PhysicsEngineConfig(automatic=self.config), proposal_engine=self._engine()
        )

        result = engine.run(path, overlays_dir=self.case_dir / "overlays")

        shadow = result.images[0].cues["cast_shadow"]
        reflection = result.images[0].cues["reflection"]
        self.assertTrue(shadow.applicable)
        self.assertTrue(reflection.applicable)
        self.assertEqual(shadow.measurements["evidence_origin"], "automatic_proposal")
        self.assertEqual(
            shadow.measurements["geometry_pair_count"],
            shadow.measurements["proposed_pair_count"],
        )
        self.assertNotIn("reviewed_pair_count", shadow.measurements)
        self.assertEqual(
            reflection.measurements["feature_backend"]["backend"],
            "test_dense_features",
        )
        self.assertTrue(Path(shadow.overlay_path or "").is_file())
        self.assertTrue(Path(reflection.overlay_path or "").is_file())

    def test_reviewed_decision_still_takes_precedence(self) -> None:
        class FailingObjectProvider:
            def detect(self, image: Image.Image):
                raise AssertionError("reviewed shadow evidence should skip object inference")

        path = self.case_dir / "scene.png"
        annotations = self.case_dir / "annotations.json"
        self.image.save(path)
        annotations.write_text(
            json.dumps(
                {
                    "images": {
                        "scene.png": {
                            "cast_shadow": {"applicability": "not_applicable"}
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        proposal_engine = AutomaticProposalEngine(
            self.config,
            mask_provider=StaticMaskProvider(self.shadow, self.mirror),
            feature_provider=StaticFeatureProvider(_fixture_features()),
            object_provider=FailingObjectProvider(),
        )
        engine = PhysicsEngine(
            PhysicsEngineConfig(automatic=self.config), proposal_engine=proposal_engine
        )

        result = engine.run(path, annotations_path=annotations)

        shadow = result.images[0].cues["cast_shadow"]
        reflection = result.images[0].cues["reflection"]
        self.assertFalse(shadow.applicable)
        self.assertEqual(shadow.measurements["review_applicability"], "not_applicable")
        self.assertTrue(reflection.applicable)

    def test_reviewed_reflection_skips_dense_feature_inference(self) -> None:
        class FailingFeatureProvider:
            def extract(self, image: Image.Image):
                raise AssertionError(
                    "reviewed reflection evidence should skip feature inference"
                )

        path = self.case_dir / "scene.png"
        annotations = self.case_dir / "annotations.json"
        self.image.save(path)
        annotations.write_text(
            json.dumps(
                {
                    "images": {
                        "scene.png": {
                            "reflection": {"applicability": "not_applicable"}
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        proposal_engine = AutomaticProposalEngine(
            self.config,
            mask_provider=StaticMaskProvider(self.shadow, self.mirror),
            feature_provider=FailingFeatureProvider(),
            object_provider=StaticObjectProvider(self.objects),
        )
        engine = PhysicsEngine(
            PhysicsEngineConfig(automatic=self.config), proposal_engine=proposal_engine
        )

        result = engine.run(path, annotations_path=annotations)

        shadow = result.images[0].cues["cast_shadow"]
        reflection = result.images[0].cues["reflection"]
        self.assertTrue(shadow.applicable)
        self.assertFalse(reflection.applicable)
        self.assertEqual(
            reflection.measurements["review_applicability"], "not_applicable"
        )

    def test_external_feature_grid_replaces_feature_provider(self) -> None:
        class FailingFeatureProvider:
            def extract(self, image: Image.Image):
                raise AssertionError("external grid should bypass this provider")

        engine = AutomaticProposalEngine(
            AutomaticProposalConfig(
                enabled=True,
                feature_backend="external",
                shadow_threshold=0.5,
                mirror_threshold=0.5,
                min_shadow_pair_confidence=0.25,
                min_reflection_pair_confidence=0.25,
                min_feature_similarity=0.8,
                min_feature_margin=0.08,
            ),
            mask_provider=StaticMaskProvider(self.shadow, self.mirror),
            feature_provider=FailingFeatureProvider(),
            object_provider=StaticObjectProvider(self.objects),
        )

        proposals = engine.propose(
            self.image, external_features=_fixture_features()
        ).reflection

        self.assertTrue(proposals.applicable)
        self.assertTrue(
            proposals.measurements["feature_backend"]["shared_primary_forward"]
        )

    def test_structured_external_features_preserve_checkpoint_provenance(self) -> None:
        engine = AutomaticProposalEngine(
            AutomaticProposalConfig(
                enabled=True,
                feature_backend="external",
                shadow_threshold=0.5,
                mirror_threshold=0.5,
                min_reflection_pair_confidence=0.25,
                min_feature_similarity=0.8,
                min_feature_margin=0.08,
            ),
            mask_provider=StaticMaskProvider(self.shadow, self.mirror),
            feature_provider=None,
            object_provider=StaticObjectProvider(self.objects),
        )

        proposals = engine.propose(
            self.image,
            external_features={
                "values": _fixture_features().astype(np.float16),
                "backend": "shared_patchhead_dinov3_tokens",
                "model": "vit_large_patch16_dinov3.lvd1689m",
                "metadata": {
                    "source_detector_family": "dino_patchhead",
                    "source_checkpoint_sha256": "abc123",
                    "feature_dtype": "float16",
                    "score_independent": True,
                },
            },
        ).reflection

        metadata = proposals.measurements["feature_backend"]
        self.assertEqual(metadata["model"], "vit_large_patch16_dinov3.lvd1689m")
        self.assertEqual(metadata["source_checkpoint_sha256"], "abc123")
        self.assertEqual(metadata["feature_dtype"], "float16")
        self.assertTrue(metadata["score_independent"])
        self.assertTrue(metadata["shared_primary_forward"])

    def test_malformed_structured_external_features_are_rejected(self) -> None:
        engine = self._engine()

        with self.assertRaisesRegex(AutomaticProposalError, "must contain `values`"):
            engine.propose(self.image, external_features={"metadata": {}})

    def test_three_automatic_pairs_cannot_assert_definitive_inconsistency(self) -> None:
        config = AutomaticProposalConfig(
            enabled=True,
            min_pairs=3,
            min_pairs_for_definitive_inconsistency=4,
        )
        engine = PhysicsEngine(PhysicsEngineConfig(automatic=config))
        pairs = [
            ShadowPair((20.0 + index * 20.0, 80.0), (25.0, 150.0 - index * 10.0))
            for index in range(3)
        ]
        proposal = AutomaticCueProposals(
            cue="cast_shadow",
            pairs=pairs,
            applicable=True,
            confidence=0.75,
            reason="deterministic gate fixture",
        )
        result = CueResult(
            cue="cast_shadow",
            applicable=True,
            status="inconsistent",
            violation_score=0.91,
            confidence=0.8,
            summary="fixture inconsistency",
        )

        decorated = engine._decorate_automatic_result(result, proposal)

        self.assertEqual(decorated.status, "indeterminate")
        self.assertEqual(decorated.violation_score, 0.5)
        self.assertEqual(decorated.measurements["proposed_pair_count"], 3)
        gate = decorated.measurements["automatic_definitive_inconsistency_gate"]
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["required_pairs"], 4)

    def test_four_automatic_pairs_can_retain_geometric_inconsistency(self) -> None:
        config = AutomaticProposalConfig(
            enabled=True,
            min_pairs=3,
            min_pairs_for_definitive_inconsistency=4,
        )
        engine = PhysicsEngine(PhysicsEngineConfig(automatic=config))
        proposal = AutomaticCueProposals(
            cue="cast_shadow",
            pairs=[
                ShadowPair(
                    (20.0 + index * 20.0, 80.0),
                    (25.0, 150.0 - index * 10.0),
                )
                for index in range(4)
            ],
            applicable=True,
            confidence=0.75,
            reason="deterministic gate fixture",
        )
        result = CueResult(
            cue="cast_shadow",
            applicable=True,
            status="inconsistent",
            violation_score=0.91,
            confidence=0.8,
            summary="fixture inconsistency",
        )

        decorated = engine._decorate_automatic_result(result, proposal)

        self.assertEqual(decorated.status, "inconsistent")
        self.assertEqual(decorated.violation_score, 0.91)
        self.assertAlmostEqual(decorated.confidence, 0.6)
        gate = decorated.measurements["automatic_definitive_inconsistency_gate"]
        self.assertTrue(gate["passed"])


if __name__ == "__main__":
    unittest.main()
