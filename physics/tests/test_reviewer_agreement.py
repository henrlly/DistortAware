from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from physics_engine.reviewer_agreement import (
    _cohen_kappa,
    evaluate_reviewer_agreement,
    write_report,
)


def _shadow_pairs(offset: float = 0.0) -> list[dict[str, object]]:
    return [
        {
            "object_contact": [0.20 + offset, y],
            "shadow_tip": [0.40 + offset, y + 0.10],
            "confidence": 1.0,
        }
        for y in (0.20, 0.45, 0.70)
    ]


class ReviewerAgreementTests(unittest.TestCase):
    def test_kappa_for_balanced_perfect_agreement(self) -> None:
        result = _cohen_kappa(
            ["applicable", "not_applicable"],
            ["applicable", "not_applicable"],
            ["applicable", "not_applicable"],
        )
        self.assertEqual(result["observed_agreement"], 1.0)
        self.assertEqual(result["cohen_kappa"], 1.0)

    def test_independent_review_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            image_root.mkdir()
            for name in ("applicable.png", "not-applicable.png"):
                Image.new("RGB", (100, 100), "white").save(image_root / name)

            def payload(reviewer_id: str, offset: float) -> dict[str, object]:
                return {
                    "schema_version": "0.1.0",
                    "coordinate_space": "normalized",
                    "reviewer": {"id": reviewer_id},
                    "images": {
                        "applicable.png": {
                            "cast_shadow": {
                                "applicability": "applicable",
                                "pairs": _shadow_pairs(offset),
                            },
                            "reflection": {
                                "applicability": "not_applicable",
                                "pairs": [],
                            },
                        },
                        "not-applicable.png": {
                            "cast_shadow": {
                                "applicability": "not_applicable",
                                "pairs": [],
                            },
                            "reflection": {
                                "applicability": "not_applicable",
                                "pairs": [],
                            },
                        },
                    },
                }

            path_a = root / "reviewer-a.json"
            path_b = root / "reviewer-b.json"
            path_a.write_text(json.dumps(payload("reviewer-a", 0.0)))
            path_b.write_text(json.dumps(payload("reviewer-b", 0.01)))

            report = evaluate_reviewer_agreement(
                image_root, path_a, path_b, pair_tolerance=0.05
            )

            shadow = report["per_cue"]["cast_shadow"]
            self.assertEqual(report["evaluated_image_count"], 2)
            self.assertEqual(shadow["both_reviewed"], 2)
            self.assertEqual(
                shadow["applicability_decision_agreement"]["cohen_kappa"], 1.0
            )
            self.assertEqual(
                shadow["geometric_status_agreement"]["observed_agreement"], 1.0
            )
            self.assertEqual(shadow["pair_concordance"]["matched_pairs"], 3)
            self.assertAlmostEqual(
                shadow["pair_concordance"]["mean_matched_pair_distance"], 0.01
            )

            json_path, markdown_path = write_report(root / "agreement.json", report)
            self.assertTrue(json_path.is_file())
            self.assertIn("reviewer-a", markdown_path.read_text())

    def test_rejects_same_reviewer_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            image_root.mkdir()
            Image.new("RGB", (10, 10), "white").save(image_root / "image.png")
            payload = {
                "reviewer": {"id": "same-person"},
                "images": {"image.png": {}},
            }
            path_a = root / "a.json"
            path_b = root / "b.json"
            path_a.write_text(json.dumps(payload))
            path_b.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "distinct reviewer IDs"):
                evaluate_reviewer_agreement(image_root, path_a, path_b)

    def test_rejects_missing_reviewer_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            image_root.mkdir()
            Image.new("RGB", (10, 10), "white").save(image_root / "image.png")
            path_a = root / "a.json"
            path_b = root / "b.json"
            path_a.write_text(json.dumps({"images": {"image.png": {}}}))
            path_b.write_text(
                json.dumps(
                    {"reviewer": {"id": "reviewer-b"}, "images": {"image.png": {}}}
                )
            )
            with self.assertRaisesRegex(ValueError, "non-empty reviewer.id"):
                evaluate_reviewer_agreement(image_root, path_a, path_b)

    def test_pair_dice_includes_applicable_images_with_no_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            image_root.mkdir()
            Image.new("RGB", (100, 100), "white").save(image_root / "image.png")
            payload_a = {
                "reviewer": {"id": "a"},
                "images": {
                    "image.png": {
                        "cast_shadow": {
                            "applicability": "applicable",
                            "pairs": _shadow_pairs(0.0),
                        }
                    }
                },
            }
            payload_b = {
                "reviewer": {"id": "b"},
                "images": {
                    "image.png": {
                        "cast_shadow": {
                            "applicability": "applicable",
                            "pairs": _shadow_pairs(0.5),
                        }
                    }
                },
            }
            path_a = root / "a.json"
            path_b = root / "b.json"
            path_a.write_text(json.dumps(payload_a))
            path_b.write_text(json.dumps(payload_b))

            report = evaluate_reviewer_agreement(
                image_root, path_a, path_b, pair_tolerance=0.05
            )

            concordance = report["per_cue"]["cast_shadow"]["pair_concordance"]
            self.assertEqual(concordance["comparable_images_with_pairs"], 1)
            self.assertEqual(concordance["images_with_matched_pairs"], 0)
            self.assertEqual(concordance["mean_image_pair_count_dice"], 0.0)


if __name__ == "__main__":
    unittest.main()
