"""Shared manifest contract for the PatchHead workflow."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

FIELDS = ["image_path", "label", "source", "category", "generator", "group_id", "mask_path", "transform"]


@dataclass(frozen=True)
class ImageRecord:
    image_path: str
    label: int
    source: str = ""
    category: str = ""
    generator: str = ""
    group_id: str = ""
    mask_path: str = ""
    transform: str = "original"


def load_manifest(path: str | Path) -> list[ImageRecord]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image_path", "label"}.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain image_path and label: {path}")
        records = []
        for row in reader:
            image = Path(row["image_path"])
            if not image.is_absolute():
                image = (path.parent / image).resolve()
            label = int(row["label"])
            if label not in (0, 1, 2):
                raise ValueError(f"Invalid label {label} in {path}")
            records.append(ImageRecord(
                image_path=str(image), label=label,
                source=row.get("source", ""), category=row.get("category", ""),
                generator=row.get("generator", ""),
                group_id=row.get("group_id", "") or str(image),
                mask_path=row.get("mask_path", ""),
                transform=row.get("transform", "original") or "original"))
    if not records:
        raise ValueError(f"Manifest contains no records: {path}")
    return records


def write_manifest(path: str | Path, records: list[ImageRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.parent.resolve()

    def relative(value: str) -> str:
        if not value:
            return ""
        candidate = Path(value).resolve()
        try:
            return str(candidate.relative_to(base))
        except ValueError:
            return str(candidate)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for record in records:
            row = {name: getattr(record, name) for name in FIELDS}
            row["image_path"] = relative(record.image_path)
            row["mask_path"] = relative(record.mask_path)
            writer.writerow(row)


def split_records(records: list[ImageRecord], *, seed: int = 42,
                  validation_fraction: float = .15,
                  calibration_fraction: float = .15,
                  test_fraction: float = .20) -> dict[str, list[ImageRecord]]:
    """Split by group, preserving related records in one split."""
    groups: dict[str, list[ImageRecord]] = {}
    for record in records:
        groups.setdefault(record.group_id or record.image_path, []).append(record)
    ids = list(groups)
    random.Random(seed).shuffle(ids)
    targets = {
        "validation": len(records) * validation_fraction,
        "calibration": len(records) * calibration_fraction,
        "test": len(records) * test_fraction,
    }
    result = {"train": [], "validation": [], "calibration": [], "test": []}
    assigned = {key: 0 for key in targets}
    for group_id in ids:
        destination = min(targets, key=lambda name: assigned[name] / max(targets[name], 1))
        if assigned[destination] >= targets[destination]:
            destination = "train"
        else:
            assigned[destination] += len(groups[group_id])
        result[destination].extend(groups[group_id])
    return result


def split_train_only(records: list[ImageRecord], *, seed: int = 42) -> dict[str, list[ImageRecord]]:
    """Split records that may join training but must never enter final test."""
    groups: dict[str, list[ImageRecord]] = {}
    for record in records:
        groups.setdefault(record.group_id or record.image_path, []).append(record)
    ids = list(groups)
    random.Random(seed).shuffle(ids)
    result = {"train": [], "validation": [], "calibration": []}
    targets = {"validation": len(records) * .15, "calibration": len(records) * .15}
    assigned = {key: 0 for key in targets}
    for group_id in ids:
        destination = min(targets, key=lambda name: assigned[name] / max(targets[name], 1))
        if assigned[destination] >= targets[destination]:
            destination = "train"
        else:
            assigned[destination] += len(groups[group_id])
        result[destination].extend(groups[group_id])
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_fingerprint(records: list[ImageRecord]) -> str:
    payload = [record.__dict__ for record in records]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
