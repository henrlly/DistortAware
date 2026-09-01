from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import infer as unified_infer
from patchhead.tests.test_inference import FakeFeatureRuntime, FakeRuntime


class UnifiedInferenceTests(unittest.TestCase):
    def test_fake_primary_runtime_and_real_physics_merge_in_one_flow(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unified-infer-") as temporary:
            root = Path(temporary)
            Image.new("RGB", (96, 72), (80, 90, 100)).save(root / "scene.png")
            args = SimpleNamespace(
                image_dir=str(root),
                ckpt=None,
                device=None,
                batch=2,
                max_images=None,
                export_patch_evidence=False,
                with_physics=True,
                physics_annotations=None,
                physics_overlays_dir=None,
                allow_missing_physics=False,
            )

            payload = unified_infer._run_patchhead(args, runtime=FakeRuntime())

            record = payload["images"][0]
            self.assertIn("patch_evidence", record)
            self.assertIn("physics_evidence", record)
            self.assertIn("dino_physics_alignment", record)
            self.assertEqual(record["physics_evidence"]["aggregate"]["status"], "indeterminate")
            self.assertFalse(payload["physics_integration"]["physics_affects_detector_score"])
            self.assertEqual(payload["summary"]["physics_integration"]["matched_records"], 1)

    def test_duplicate_basenames_join_by_relative_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unified-infer-duplicates-") as temporary:
            root = Path(temporary)
            for folder in ("first", "second"):
                (root / folder).mkdir()
                Image.new("RGB", (64, 48), (50, 60, 70)).save(
                    root / folder / "scene.png"
                )
            args = SimpleNamespace(
                image_dir=str(root),
                ckpt=None,
                device=None,
                batch=2,
                max_images=None,
                export_patch_evidence=False,
                with_physics=True,
                physics_annotations=None,
                physics_overlays_dir=None,
                allow_missing_physics=False,
            )

            payload = unified_infer._run_patchhead(args, runtime=FakeRuntime())

            self.assertEqual(
                [record["image_path"] for record in payload["images"]],
                ["first/scene.png", "second/scene.png"],
            )
            self.assertTrue(
                all(
                    record["physics_evidence"]["match_method"] == "canonical_path"
                    for record in payload["images"]
                )
            )

    def test_same_pass_dino_grid_can_feed_automatic_physics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="unified-infer-auto-") as temporary:
            root = Path(temporary)
            Image.new("RGB", (96, 72), (110, 120, 130)).save(root / "scene.png")
            args = SimpleNamespace(
                image_dir=str(root),
                ckpt=None,
                device=None,
                batch=1,
                max_images=None,
                export_patch_evidence=False,
                with_physics=True,
                physics_annotations=None,
                physics_overlays_dir=None,
                allow_missing_physics=False,
                physics_auto_proposals=True,
                physics_proposal_mask_backend="heuristic",
                physics_proposal_feature_backend="patchhead",
                physics_proposal_object_backend="edges",
                physics_proposal_cache_dir=None,
                physics_proposal_device=None,
                physics_proposal_offline=False,
                physics_strict_proposal_models=False,
                physics_feature_memory_mib=8,
            )

            original_run_physics = unified_infer._run_physics
            captured_features = {}

            def capture_run_physics(args, *, dense_feature_maps=None):
                captured_features.update(dense_feature_maps or {})
                return original_run_physics(
                    args, dense_feature_maps=dense_feature_maps
                )

            with patch.object(
                unified_infer, "_run_physics", side_effect=capture_run_physics
            ):
                payload = unified_infer._run_patchhead(
                    args, runtime=FakeFeatureRuntime()
                )

            self.assertTrue(payload["detector"]["dense_features_forwarded_in_memory"])
            automatic = payload["summary"]["physics_integration"]
            self.assertEqual(automatic["matched_records"], 1)
            evidence = payload["images"][0]["physics_evidence"]
            self.assertEqual(evidence["cues"]["reflection"]["status"], "not_applicable")
            transferred = captured_features["scene.png"]
            self.assertEqual(transferred["backend"], "shared_patchhead_dinov3_tokens")
            self.assertEqual(transferred["metadata"]["feature_dtype"], "float16")
            self.assertEqual(
                transferred["metadata"]["source_detector_family"],
                "fake_patchhead_for_contract_tests",
            )
            self.assertTrue(transferred["metadata"]["score_independent"])


if __name__ == "__main__":
    unittest.main()
