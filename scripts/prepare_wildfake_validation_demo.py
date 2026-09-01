#!/usr/bin/env python3
"""Prepare a tiny, reproducible WildFake validation-only browser demo.

The script uses HTTP range reads against the official ModelScope archives. It
therefore downloads only selected image members, not either multi-gigabyte ZIP.
The generated data directory is ignored by Git and must never train a model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from remote_zip import open_remote_zip, wildfake_url  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "data" / "wildfake_validation_demo"
DEFAULT_SEED = "tiktok-techjam-2026-validation-demo-v1"
MAX_PER_CLASS = 24
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
FORMAT_SUFFIXES = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


@dataclass(frozen=True)
class Source:
    key: str
    ground_truth: str
    display_name: str
    archive_path: str
    member_prefix: str
    expected_count: int


SOURCES = (
    Source(
        key="coco_val2017",
        ground_truth="real",
        display_name="COCO val2017",
        archive_path="Images/Real/coco.zip",
        member_prefix="coco/coco2017/val2017/",
        expected_count=4_998,
    ),
    Source(
        key="dalle_advanced",
        ground_truth="aigc",
        display_name="DALL-E Advanced (DALL-E 3)",
        archive_path="Images/Diffusion_based/DALLE.zip",
        member_prefix="DALLE/Advanced/DALLE3/dalle3/",
        expected_count=8_843,
    ),
)


def digest_rank(seed: str, namespace: str, value: str) -> bytes:
    material = f"{seed}\0{namespace}\0{value}".encode("utf-8")
    return hashlib.sha256(material).digest()


def choose_members(names: list[str], count: int, seed: str, source: Source) -> list[str]:
    """Choose a stable pseudo-random subset without depending on list order."""
    return sorted(names, key=lambda name: digest_rank(seed, source.key, name))[:count]


def inspect_image(payload: bytes, member: str) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        image_format = (image.format or "").upper()
        width, height = image.size
    suffix = FORMAT_SUFFIXES.get(image_format)
    if suffix is None:
        raise ValueError(f"Unsupported browser-demo image format {image_format!r}: {member}")
    if width < 32 or height < 32:
        raise ValueError(f"Image is unexpectedly small ({width}x{height}): {member}")
    return width, height, suffix


def read_source(source: Source, count: int, seed: str) -> list[dict[str, Any]]:
    print(f"Indexing {source.display_name} via HTTP range reads...", flush=True)
    with open_remote_zip(wildfake_url(source.archive_path)) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.startswith(source.member_prefix)
            and name.lower().endswith(IMAGE_SUFFIXES)
            and not name.endswith("/")
        ]
        if len(names) != source.expected_count:
            raise RuntimeError(
                f"{source.display_name} count mismatch: expected "
                f"{source.expected_count:,}, found {len(names):,}. Refusing to build "
                "a page that may not match the declared validation benchmark."
            )

        selected = choose_members(names, count, seed, source)
        # Reading in archive order improves locality; display order is shuffled later.
        selected.sort(key=lambda name: archive.getinfo(name).header_offset)
        records: list[dict[str, Any]] = []
        for index, member in enumerate(selected, start=1):
            payload = archive.read(member)
            width, height, suffix = inspect_image(payload, member)
            records.append(
                {
                    "ground_truth": source.ground_truth,
                    "source_key": source.key,
                    "source_display": source.display_name,
                    "source_archive": source.archive_path,
                    "source_member": member,
                    "source_population": source.expected_count,
                    "width": width,
                    "height": height,
                    "suffix": suffix,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "byte_size": len(payload),
                    "payload": payload,
                }
            )
            print(f"  fetched {index}/{count}", flush=True)
    return records


def remove_previous_output(output: Path) -> None:
    images = output / "images"
    for path in images.glob("wf_eval_*.*") if images.exists() else ():
        path.unlink()
    manifest = output / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def prepare(output: Path, count_per_class: int, seed: str, force: bool) -> Path:
    manifest_path = output / "manifest.json"
    images_path = output / "images"
    has_previous = manifest_path.exists() or any(images_path.glob("wf_eval_*.*"))
    if has_previous and not force:
        raise FileExistsError(
            f"Generated demo already exists at {output}. Use --force to replace only "
            "its wf_eval_* images and manifest."
        )
    if force:
        remove_previous_output(output)
    images_path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for source in SOURCES:
        records.extend(read_source(source, count_per_class, seed))

    records.sort(key=lambda item: digest_rank(seed, "display-order", item["source_member"]))
    items: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        sample_id = f"{index:03d}"
        filename = f"wf_eval_{sample_id}{record.pop('suffix')}"
        payload = record.pop("payload")
        (images_path / filename).write_bytes(payload)
        items.append({"id": sample_id, "file": f"images/{filename}", **record})

    manifest = {
        "schema_version": 1,
        "title": "WildFake validation-only browser demonstration subset",
        "intended_use": "demonstration_and_evaluation_only",
        "training_allowed": False,
        "seed": seed,
        "count_per_class": count_per_class,
        "source_url": "https://modelscope.cn/datasets/hy2628982280/WildFake/summary",
        "detector_blinding": {
            "image_filenames_are_label_neutral": True,
            "image_alt_text_is_label_neutral": True,
            "ground_truth_is_rendered_in_a_sibling_caption_outside_the_image_crop": True,
        },
        "source_populations": {source.key: source.expected_count for source in SOURCES},
        "items": items,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Generated data directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--count-per-class",
        type=int,
        default=6,
        help="Images from each benchmark class (default: 6; maximum: 24)",
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only a prior generated manifest and wf_eval_* image files",
    )
    args = parser.parse_args()
    if not 1 <= args.count_per_class <= MAX_PER_CLASS:
        parser.error(f"--count-per-class must be between 1 and {MAX_PER_CLASS}")
    return args


def main() -> None:
    args = parse_args()
    manifest = prepare(
        output=args.output.resolve(),
        count_per_class=args.count_per_class,
        seed=args.seed,
        force=args.force,
    )
    size = sum(path.stat().st_size for path in manifest.parent.rglob("*") if path.is_file())
    print(f"Created {manifest}")
    print(f"Local demo footprint: {size / (1024 * 1024):.2f} MiB")
    print("Evaluation/demo only: do not copy these files into a training split.")


if __name__ == "__main__":
    main()
