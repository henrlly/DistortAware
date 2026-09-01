from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from physics_engine.checkpoint_eval import (
    CheckpointEvaluationError,
    evaluate_checkpoint_predictions,
    render_markdown,
)


def _cue(status: str, *, shared: bool = False) -> dict:
    applicable = status != "not_applicable"
    measurements = {}
    if shared:
        measurements["feature_backend"] = {
            "backend": "shared_patchhead_dinov3_tokens",
            "shared_primary_forward": True,
            "source_checkpoint_sha256": "checkpoint-hash",
        }
    return {
        "applicable": applicable,
        "status": status,
        "violation_score": 0.1 if applicable else None,
        "measurements": measurements,
    }


class CheckpointEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="checkpoint-eval-")
        self.root = Path(self.temporary.name)
        images = self.root / "images"
        masks = self.root / "masks" / "tampered"
        for label in ("real", "full_synthetic", "tampered"):
            (images / label).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (50, 80, 100)).save(images / label / f"{label}.png")
        masks.mkdir(parents=True)
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[:8, :8] = 255
        Image.fromarray(mask).save(masks / "tampered.png")
        self.manifest = {
            "images": [
                {
                    "label": 0,
                    "label_name": "real",
                    "image_path": str(images / "real" / "real.png"),
                    "mask_path": None,
                },
                {
                    "label": 1,
                    "label_name": "full_synthetic",
                    "image_path": str(images / "full_synthetic" / "full_synthetic.png"),
                    "mask_path": None,
                },
                {
                    "label": 2,
                    "label_name": "tampered",
                    "image_path": str(images / "tampered" / "tampered.png"),
                    "mask_path": str(masks / "tampered.png"),
                },
            ]
        }
        self.predictions = {
            "detector": {
                "family": "dino_patchhead",
                "arch": "patchhead-dinov3-vitl16",
                "backbone": "fixture-dino",
                "checkpoint_dataset": "pooled",
                "checkpoint_sha256": "checkpoint-hash",
                "threshold": 0.5,
                "score_formula": "fixture",
            },
            "physics_integration": {
                "primary_detector_fields_preserved": True,
                "physics_affects_detector_score": False,
            },
            "images": [],
        }
        for path, score, verdict in (
            ("real/real.png", 0.1, False),
            ("full_synthetic/full_synthetic.png", 0.9, True),
            ("tampered/tampered.png", 0.8, True),
        ):
            self.predictions["images"].append(
                {
                    "image_path": path,
                    "aigc_score": score,
                    "is_aigc": verdict,
                    "component_scores": {"patch_head": score, "cls_head": score},
                    "patch_evidence": {
                        "values": [[0.9, 0.1], [0.1, 0.1]],
                        "grid_shape": [2, 2],
                    },
                    "physics_evidence": {
                        "errors": [],
                        "cues": {
                            "perspective": _cue("consistent"),
                            "cast_shadow": _cue("not_applicable"),
                            "reflection": _cue("not_applicable", shared=True),
                        },
                    },
                    "dino_physics_alignment": {"applicable": path.startswith("tampered")},
                }
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_sid_metrics_keep_tampered_separate_and_verify_exact_parity(self) -> None:
        result = evaluate_checkpoint_predictions(
            self.predictions,
            manifest=self.manifest,
            baseline=deepcopy(self.predictions),
            dataset_name="fixture SID",
            backbone_revision="fixture-revision",
        )

        binary = result["primary_detector"]["binary"]
        self.assertEqual(binary["count"], 2)
        self.assertEqual(binary["accuracy"], 1.0)
        self.assertEqual(binary["roc_auc"], 1.0)
        self.assertEqual(
            result["primary_detector"]["per_label"]["tampered"]["aigc_alert_rate"],
            1.0,
        )
        self.assertTrue(
            result["integration_contract"]["detector_only_parity"][
                "exact_primary_parity"
            ]
        )
        tamper = result["tamper_patch_localization"]
        self.assertEqual(tamper["evaluable_tampered_images"], 1)
        self.assertEqual(tamper["metrics"]["patch_roc_auc"]["mean"], 1.0)
        self.assertEqual(tamper["metrics"]["top_area_iou"]["mean"], 1.0)
        self.assertEqual(
            result["physics_sidecar"]["same_pass_reflection_feature_attempts"], 3
        )
        self.assertIn("Binary metrics include only", render_markdown(result))

    def test_parent_directory_labels_support_small_wildfake_samples(self) -> None:
        predictions = deepcopy(self.predictions)
        predictions["images"] = predictions["images"][:2]
        predictions["images"][0]["image_path"] = "real/coco.png"
        predictions["images"][1]["image_path"] = "fake/adm.png"

        result = evaluate_checkpoint_predictions(predictions, dataset_name="WildFake pilot")

        self.assertEqual(result["primary_detector"]["binary"]["accuracy"], 1.0)

    def test_non_pooled_checkpoint_is_rejected(self) -> None:
        predictions = deepcopy(self.predictions)
        predictions["detector"]["checkpoint_dataset"] = "sid_set"

        with self.assertRaisesRegex(CheckpointEvaluationError, "pooled"):
            evaluate_checkpoint_predictions(predictions, manifest=self.manifest)

    def test_mismatched_baseline_paths_are_rejected(self) -> None:
        baseline = deepcopy(self.predictions)
        baseline["images"].pop()

        with self.assertRaisesRegex(CheckpointEvaluationError, "do not match"):
            evaluate_checkpoint_predictions(
                self.predictions, manifest=self.manifest, baseline=baseline
            )


if __name__ == "__main__":
    unittest.main()
