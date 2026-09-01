import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from harness.common import Record, TRANSFORMS, apply_transform, fingerprint, load_manifest, materialize_view, select_per_class, write_manifest
from harness.metrics import summarize
from harness.reports import write_reports
from harness.did_data import prepare
from harness.adapters import _normalize
from harness.verify_fetch import FetchVerificationError, verify


class HarnessTests(unittest.TestCase):
    def _records(self, root: Path):
        records = []
        for label in (0, 1, 2):
            for index in range(3):
                path = root / f"{label}_{index}.png"
                Image.new("RGB", (32, 24), (label * 80, index * 30, 40)).save(path)
                records.append(Record(str(path), label, f"source{label}", "category", "generator", f"group:{label}:{index}"))
        return records

    def test_select_per_class_is_deterministic_and_balanced(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = self._records(Path(temporary))
            first = select_per_class(records, 2, 42)
            second = select_per_class(records, 2, 42)
            self.assertEqual(fingerprint(first), fingerprint(second))
            self.assertEqual({label: sum(r.label == label for r in first) for label in (0, 1, 2)}, {0: 2, 1: 2, 2: 2})

    def test_manifest_round_trip_and_transforms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = self._records(root)
            manifest = root / "manifest.csv"
            write_manifest(manifest, records)
            loaded = load_manifest(manifest)
            self.assertEqual(fingerprint(records), fingerprint(loaded))
            for transform in TRANSFORMS:
                output = apply_transform(Image.open(records[0].image_path), transform, 42)
                self.assertEqual(output.mode, "RGB")

    def test_materialized_view_has_manifest_and_transform(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = self._records(root)
            view = materialize_view(records[:2], root / "view", "jpeg90", 42)
            self.assertEqual(len(view), 2)
            self.assertTrue((root / "view" / "manifest.csv").is_file())
            self.assertTrue(all(record.transform == "jpeg90" for record in view))

    def test_metrics_include_auc_and_physics_confidence(self):
        records = [
            {"model": "patchhead_baseline", "transform": "clean", "label": 0, "score": .1, "threshold": .5},
            {"model": "patchhead_baseline", "transform": "clean", "label": 1, "score": .9, "threshold": .5},
            {"model": "physics", "transform": "clean", "label": 0, "score": .2,
             "physics": {"status": "consistent", "violation_score": .2, "confidence": .8}},
        ]
        report = summarize(records)
        self.assertEqual(report["groups"]["patchhead_baseline:clean"]["roc_auc"], 1.0)
        self.assertIn("0", report["groups"]["patchhead_baseline:clean"]["by_label"])
        self.assertEqual(report["physics"]["clean"]["mean_confidence"], .8)

    def test_missing_records_are_not_counted_as_coverage(self):
        from harness.evaluate import _coverage
        report = _coverage([{"model": "physics", "transform": "clean", "image_id": "a", "missing": True}],
                           1, {"physics"}, ("clean",))
        self.assertEqual(report["physics:clean"]["returned"], 0)
        self.assertEqual(report["physics:clean"]["missing"], 1)

    def test_standard_entrypoint_output_is_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = self._records(root)[:1]

            def fake_entrypoint(_repo, _module, view, _output, _arguments, _python_path):
                return [{"method": "filter", "image_path": str(next(view.glob("*.jpg"))),
                         "score": .8, "score_kind": "ai_probability",
                         "confidence": .8, "threshold": .5, "decision": True,
                         "details": {"prediction": "fully_synthetic"}, "errors": []}]

            with patch("harness.adapters._run_entrypoint", side_effect=fake_entrypoint):
                result = _normalize(root, records, "clean", root / "results",
                                    "filter", "filter_based_approach.entrypoint", [])
            self.assertEqual(result[0]["model"], "filter")
            self.assertEqual(result[0]["score"], .8)
            self.assertEqual(result[0]["filter"]["prediction"], "fully_synthetic")

    def test_patchhead_harness_does_not_use_legacy_infer_cli(self):
        source = (Path(__file__).resolve().parents[1] / "adapters.py").read_text()
        self.assertNotIn('"patchhead/infer.py"', source)

    def test_filter_slurm_uses_shared_manifest_not_archives(self):
        source = (Path(__file__).resolve().parents[2] / "slurm/evaluate_filter.sh").read_text()
        self.assertIn("--models filter", source)
        self.assertNotIn("celebahq.zip", source)
        self.assertNotIn("DDIM.zip", source)

    def test_reports_write_combined_and_per_model_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"
            records = [
                {"model": "patchhead_baseline", "transform": "clean", "image_id": "a",
                 "image_path": "/a.jpg", "source": "sid", "category": "real", "label": 0,
                 "score": .1, "score_kind": "aigc", "confidence": None, "threshold": .5,
                 "decision": False, "missing": False, "errors": []},
            ]
            report = {"data_dir": "/data", "manifest": "test.csv", "records": 1,
                      "models": ["patchhead_baseline"],
                      "coverage": {"patchhead_baseline:clean": {"expected": 1, "returned": 1,
                                                                    "missing": 0, "duplicates": 0,
                                                                    "errors": 0}},
                      "metrics": {"groups": {"patchhead_baseline:clean": {
                          "n": 1, "accuracy": 1.0, "balanced_accuracy": 1.0,
                          "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": None}},
                                   "physics": {}}}
            write_reports(output, records, report)
            self.assertTrue((output / "records.csv").is_file())
            self.assertTrue((output / "report.md").is_file())
            self.assertTrue((output / "models/patchhead_baseline/records.csv").is_file())
            self.assertTrue((output / "models/patchhead_baseline/report.md").is_file())

    def test_did_data_prepares_binary_symlink_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = self._records(root)
            data = root / "data"
            data.mkdir()
            write_manifest(data / "train.csv", records)
            write_manifest(data / "test.csv", records)
            output = root / "did"
            prepare(data, output)
            self.assertEqual(len(list((output / "train" / "real").iterdir())), 3)
            self.assertEqual(len(list((output / "train" / "fake").iterdir())), 6)
            self.assertTrue(next((output / "test" / "fake").iterdir()).is_symlink())

    def test_verify_fetch_checks_manifests_and_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = self._records(root)
            large = root / "large"
            quick = root / "quick"
            for directory in (large, quick):
                directory.mkdir()
                (directory / "fetch_config.json").write_text("{}")
                (directory / "dataset_report.json").write_text("{}")
            (quick / "quick_report.json").write_text("{}")
            for name in ("train.csv", "validation.csv", "calibration.csv", "matched_test.csv"):
                write_manifest(large / name, records)
            for name in ("train.csv", "validation.csv", "calibration.csv", "test.csv"):
                write_manifest(quick / name, records)
            result = verify(large, quick, 3)
            self.assertEqual(result["quick_per_class"], 3)
            (quick / "test.csv").unlink()
            with self.assertRaises(FetchVerificationError):
                verify(large, quick, 3)


if __name__ == "__main__":
    unittest.main()
