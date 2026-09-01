"""Select the reproducible SID-only evaluation manifest from a gathered set."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def select(test_manifest: Path, output: Path, count: int, seed: int) -> None:
    with test_manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("source") == "sid"]
    selected = random.Random(seed).sample(rows, min(count, len(rows)))
    if len(selected) != count:
        raise RuntimeError(f"Found only {len(selected)} SID test records; need {count}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["image_path", "label", "category", "source"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({field: str((test_manifest.parent / row[field]).resolve()) if field == "image_path" else row.get(field, "") for field in fields})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    select(args.test_manifest, args.output, args.count, args.seed)
