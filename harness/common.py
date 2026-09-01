"""Manifest, transform, and unified-record helpers for the harness."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


FIELDS = ["image_path", "label", "source", "category", "generator", "group_id", "mask_path", "transform"]
TRANSFORMS = ("clean", "jpeg90", "blur1.0", "resize0.5", "noise0.05", "jitter", "crop80")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Record:
    image_path: str
    label: int
    source: str = ""
    category: str = ""
    generator: str = ""
    group_id: str = ""
    mask_path: str = ""
    transform: str = "original"


def load_manifest(path: str | Path) -> list[Record]:
    path = Path(path).expanduser().resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image_path", "label"}.issubset(reader.fieldnames):
            raise ValueError(f"manifest must contain image_path and label: {path}")
        records = []
        for row in reader:
            image = Path(row["image_path"]).expanduser()
            if not image.is_absolute():
                image = path.parent / image
            label = int(row["label"])
            if label not in (0, 1, 2):
                raise ValueError(f"invalid label {label} in {path}")
            mask = row.get("mask_path", "")
            mask_path = ""
            if mask:
                mask_candidate = Path(mask).expanduser()
                mask_path = str((mask_candidate if mask_candidate.is_absolute() else path.parent / mask_candidate).resolve())
            records.append(Record(
                image_path=str(image.resolve()), label=label,
                source=row.get("source", ""), category=row.get("category", ""),
                generator=row.get("generator", ""), group_id=row.get("group_id", "") or str(image),
                mask_path=mask_path, transform=row.get("transform", "original") or "original",
            ))
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def write_manifest(path: str | Path, records: Iterable[Record]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            for field in ("image_path", "mask_path"):
                value = row[field]
                if value:
                    candidate = Path(value).resolve()
                    try:
                        row[field] = str(candidate.relative_to(path.parent))
                    except ValueError:
                        row[field] = str(candidate)
            writer.writerow(row)


def fingerprint(records: Iterable[Record]) -> str:
    payload = []
    for record in records:
        row = asdict(record)
        # Paths are deliberately excluded so relocation does not alter the fingerprint.
        row.pop("image_path", None)
        row.pop("mask_path", None)
        payload.append(row)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def select_per_class(records: list[Record], count: int, seed: int) -> list[Record]:
    if count <= 0:
        raise ValueError("per-class count must be positive")
    selected: list[Record] = []
    for label in (0, 1, 2):
        candidates = [record for record in records if record.label == label]
        candidates.sort(key=lambda record: hashlib.sha256(
            f"{seed}:{label}:{record.group_id}:{record.image_path}".encode()).digest())
        if len(candidates) < count:
            raise ValueError(f"label {label} has {len(candidates)} records, need {count}")
        selected.extend(candidates[:count])
    return selected


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def apply_transform(image: Image.Image, name: str, seed: int = 42) -> Image.Image:
    """Apply one deterministic representative transform."""
    if name == "clean":
        return image.convert("RGB")
    if name == "jpeg90":
        return _jpeg(image.convert("RGB"), 90)
    if name == "blur1.0":
        return image.convert("RGB").filter(ImageFilter.GaussianBlur(1.0))
    if name == "resize0.5":
        image = image.convert("RGB")
        small = image.resize((max(1, image.width // 2), max(1, image.height // 2)), Image.Resampling.BICUBIC)
        return small.resize(image.size, Image.Resampling.BICUBIC)
    if name == "noise0.05":
        import numpy as np
        import random as _random
        rng = _random.Random(seed)
        array = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 0.05, array.shape)
        return Image.fromarray((np.clip(array + noise, 0, 1) * 255).astype("uint8"), "RGB")
    if name == "jitter":
        rng = random.Random(seed)
        output = ImageEnhance.Brightness(image.convert("RGB")).enhance(rng.uniform(.8, 1.2))
        output = ImageEnhance.Contrast(output).enhance(rng.uniform(.8, 1.2))
        return ImageEnhance.Color(output).enhance(rng.uniform(.8, 1.2))
    if name == "crop80":
        image = image.convert("RGB")
        width, height = int(image.width * .8), int(image.height * .8)
        left, top = (image.width - width) // 2, (image.height - height) // 2
        return image.crop((left, top, left + width, top + height)).resize(image.size, Image.Resampling.BICUBIC)
    raise ValueError(f"unknown transform: {name}")


def materialize_view(records: list[Record], output_dir: str | Path, transform: str, seed: int) -> list[Record]:
    """Create a self-contained transformed image view and matching manifest."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.iterdir():
        if stale.is_file() and (stale.suffix.lower() in SUPPORTED_EXTENSIONS or stale.name == "manifest.csv"):
            stale.unlink()
    result: list[Record] = []
    for index, record in enumerate(records):
        source = Path(record.image_path)
        name = f"{index:06d}_{hashlib.sha256(record.group_id.encode()).hexdigest()[:12]}.jpg"
        destination = output / name
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            apply_transform(image, transform, seed + index * 1000003).save(destination, format="JPEG", quality=95)
        result.append(replace(record, image_path=str(destination), transform=transform))
    write_manifest(output / "manifest.csv", result)
    return result
