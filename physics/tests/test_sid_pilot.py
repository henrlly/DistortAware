from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import unittest
import uuid

from PIL import Image, ImageDraw

from physics_engine.sid_pilot import (
    GIB,
    SidPilotError,
    _wilson_interval,
    evaluate_sid_pilot,
    extract_sid_sample,
    main as sid_main,
)


TEST_TEMP_ROOT = Path(__file__).resolve().parent / ".tmp"


def png_bytes(*, mask: bool = False, offset: int = 0) -> bytes:
    image = Image.new("L" if mask else "RGB", (96, 72), 0 if mask else "white")
    draw = ImageDraw.Draw(image)
    if mask:
        draw.rectangle((25 + offset, 20, 65 + offset, 55), fill=255)
    else:
        draw.line((5, 10 + offset, 90, 10 + offset), fill="black", width=3)
        draw.line((10, 65, 85, 15 + offset), fill="black", width=3)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@unittest.skipUnless(
    __import__("importlib").util.find_spec("pyarrow") is not None,
    "pyarrow optional dependency is not installed",
)
class SidPilotTests(unittest.TestCase):
    def setUp(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.case_dir = TEST_TEMP_ROOT / uuid.uuid4().hex
        self.case_dir.mkdir(parents=True)
        self.shard = self.case_dir / "validation.parquet"
        rows = []
        for label in (0, 1, 2):
            for index in range(4):
                rows.append(
                    {
                        "img_id": f"label-{label}-{index}",
                        "image": {"bytes": png_bytes(offset=index), "path": None},
                        "mask": {
                            "bytes": png_bytes(mask=True, offset=index) if label == 2 else None,
                            "path": None,
                        },
                        "width": 96,
                        "height": 72,
                        "label": label,
                    }
                )
        pq.write_table(pa.Table.from_pylist(rows), self.shard)

    def tearDown(self) -> None:
        if self.case_dir.exists():
            shutil.rmtree(self.case_dir)

    def test_capped_sample_extracts_each_label_and_runs_physics(self) -> None:
        workspace = self.case_dir / "pilot"
        manifest = extract_sid_sample(
            [self.shard],
            workspace=workspace,
            per_label=2,
            seed=7,
            max_source_bytes=1 * GIB,
            max_extracted_bytes=1 * GIB,
            dataset_revision="test-revision",
        )
        report = evaluate_sid_pilot(manifest, workspace=workspace)

        self.assertEqual(len(manifest["images"]), 6)
        self.assertEqual(
            manifest["sampling"]["selected_by_label"],
            {"real": 2, "full_synthetic": 2, "tampered": 2},
        )
        self.assertLess(manifest["storage"]["extracted_bytes"], 1 * GIB)
        self.assertEqual(report["sample_size"], 6)
        self.assertTrue((workspace / "physics_results.json").is_file())
        self.assertTrue((workspace / "review_queue.json").is_file())
        review_queue = json.loads((workspace / "review_queue.json").read_text())
        self.assertEqual(review_queue["review_queue_version"], "0.2.0")
        self.assertEqual(
            sum(item["independent_double_review_target"] for item in review_queue["images"]),
            3,
        )
        self.assertEqual(report["independent_double_review_target_count"], 3)
        self.assertEqual(report["by_label"]["real"]["shadow_applicable"], 0)
        self.assertEqual(report["source_shard_count"], 1)
        interval = report["by_label"]["real"]["perspective_applicability_ci95"]
        self.assertEqual(interval["total"], 2)
        self.assertEqual(
            manifest["sampling"]["selected_by_source_file"], {str(self.shard): 6}
        )

        self.assertEqual(
            sid_main(["--workspace", str(workspace), "--evaluate-existing"]), 0
        )

    def test_nonempty_workspace_fails_without_overwriting(self) -> None:
        workspace = self.case_dir / "pilot"
        workspace.mkdir()
        (workspace / "keep.txt").write_text("user data", encoding="utf-8")

        with self.assertRaisesRegex(SidPilotError, "not empty"):
            extract_sid_sample([self.shard], workspace=workspace, per_label=1)
        self.assertEqual((workspace / "keep.txt").read_text(encoding="utf-8"), "user data")

    def test_existing_workspace_rejects_paths_outside_cap_root(self) -> None:
        workspace = self.case_dir / "pilot"
        manifest = extract_sid_sample(
            [self.shard], workspace=workspace, per_label=1, seed=9
        )
        manifest["images"][0]["image_path"] = str(self.shard)
        (workspace / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        self.assertEqual(
            sid_main(["--workspace", str(workspace), "--evaluate-existing"]), 2
        )

    def test_wilson_interval_reports_uncertainty_at_zero_successes(self) -> None:
        interval = _wilson_interval(0, 50)
        assert interval is not None
        self.assertEqual(interval["rate"], 0.0)
        self.assertEqual(interval["lower"], 0.0)
        self.assertGreater(interval["upper"], 0.0)
        self.assertLess(interval["upper"], 0.1)


if __name__ == "__main__":
    unittest.main()
