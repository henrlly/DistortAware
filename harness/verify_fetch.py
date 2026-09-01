"""Verify that a harness fetch produced usable large and quick datasets."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .common import load_manifest


class FetchVerificationError(ValueError):
    pass


def _check_manifest(path: Path, *, expected_per_class: int | None = None) -> dict[str, int]:
    if not path.is_file():
        raise FetchVerificationError(f"missing manifest: {path}")
    records = load_manifest(path)
    counts = Counter(record.label for record in records)
    if set(counts) != {0, 1, 2}:
        raise FetchVerificationError(f"{path.name} must contain labels 0, 1, and 2: {dict(counts)}")
    if expected_per_class is not None and any(counts[label] != expected_per_class for label in (0, 1, 2)):
        raise FetchVerificationError(
            f"{path.name} has unexpected class counts: {dict(counts)}; "
            f"expected {expected_per_class} each"
        )
    for record in records:
        image = Path(record.image_path)
        if not image.is_file():
            raise FetchVerificationError(f"missing image referenced by {path.name}: {image}")
        if record.mask_path and not Path(record.mask_path).is_file():
            raise FetchVerificationError(f"missing mask referenced by {path.name}: {record.mask_path}")
    return {str(label): counts[label] for label in (0, 1, 2)}


def verify(data_dir: str | Path, quick_data_dir: str | Path, quick_per_class: int = 200) -> dict:
    large = Path(data_dir).expanduser().resolve()
    quick = Path(quick_data_dir).expanduser().resolve()
    for name in ("fetch_config.json", "dataset_report.json"):
        if not (large / name).is_file():
            raise FetchVerificationError(f"missing fetch metadata: {large / name}")
    for name in ("train.csv", "validation.csv", "calibration.csv"):
        _check_manifest(large / name)
    large_test = large / "test.csv"
    if not large_test.is_file():
        large_test = large / "matched_test.csv"
    large_counts = _check_manifest(large_test)
    for name in ("train.csv", "validation.csv", "calibration.csv", "test.csv"):
        _check_manifest(quick / name, expected_per_class=quick_per_class)
    if not (quick / "quick_report.json").is_file():
        raise FetchVerificationError(f"missing quick metadata: {quick / 'quick_report.json'}")
    return {"large": str(large), "large_test_counts": large_counts,
            "quick": str(quick), "quick_per_class": quick_per_class}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/harness_large"))
    parser.add_argument("--quick-data-dir", type=Path, default=Path("data/harness_quick"))
    parser.add_argument("--quick-per-class", type=int, default=200)
    args = parser.parse_args()
    try:
        result = verify(args.data_dir, args.quick_data_dir, args.quick_per_class)
    except (FetchVerificationError, OSError, ValueError) as exc:
        print(f"FETCH FAILED: {exc}")
        return 1
    print("FETCH OK")
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
