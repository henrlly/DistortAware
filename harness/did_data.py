"""Build a DID-compatible binary image tree from the shared harness manifests."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import load_manifest


def prepare(data_dir: str | Path, output_dir: str | Path, test_manifest: str | Path | None = None) -> None:
    data = Path(data_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    for split in ("train", "test"):
        manifest_name = "train.csv"
        if split == "test":
            manifest_name = test_manifest or ("test.csv" if (data / "test.csv").is_file() else "matched_test.csv")
        manifest_path = Path(manifest_name)
        records = load_manifest(manifest_path if manifest_path.is_absolute() else data / manifest_path)
        for label_name in ("real", "fake"):
            destination = output / split / label_name
            destination.mkdir(parents=True, exist_ok=True)
            for existing in destination.iterdir():
                if existing.is_symlink():
                    existing.unlink()
                elif existing.is_file():
                    raise RuntimeError(f"refusing to overwrite non-generated file: {existing}")
        for index, record in enumerate(records):
            source = Path(record.image_path)
            if not source.is_file():
                raise FileNotFoundError(source)
            label_name = "real" if record.label == 0 else "fake"
            link = output / split / label_name / f"{index:07d}{source.suffix.lower()}"
            link.symlink_to(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path,
                        help="Optional test manifest name/path instead of test.csv or matched_test.csv")
    args = parser.parse_args()
    prepare(args.data_dir, args.output_dir, args.test_manifest)
    print(f"prepared DID binary tree: {args.output_dir}")


if __name__ == "__main__":
    main()
