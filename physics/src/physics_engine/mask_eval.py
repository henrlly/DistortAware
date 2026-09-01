"""Storage-bounded evaluation for automatic shadow or mirror masks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from io import BytesIO
import json
from pathlib import Path
import random
import time
from typing import Any, Iterable
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageOps

from .automatic import ClipSegMaskProvider, HeuristicMaskProvider
from .automatic_config import AutomaticProposalConfig, DEFAULT_CLIPSEG_REVISION


HARD_SOURCE_LIMIT_GIB = 50.0
RECOMMENDED_SOURCE_LIMIT_GIB = 10.0


@dataclass(slots=True)
class BinaryMaskMetrics:
    intersection: int
    union: int
    predicted_pixels: int
    target_pixels: int
    iou: float
    dice: float
    precision: float
    recall: float
    soft_mae: float


def binary_mask_metrics(
    probability: np.ndarray, target: np.ndarray, threshold: float
) -> BinaryMaskMetrics:
    prediction = np.asarray(probability, dtype=np.float32) >= threshold
    truth = np.asarray(target, dtype=bool)
    if prediction.shape != truth.shape:
        prediction = cv2.resize(
            prediction.astype(np.uint8),
            (truth.shape[1], truth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        probability = cv2.resize(
            np.asarray(probability, dtype=np.float32),
            (truth.shape[1], truth.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    intersection = int(np.logical_and(prediction, truth).sum())
    union = int(np.logical_or(prediction, truth).sum())
    predicted = int(prediction.sum())
    target_pixels = int(truth.sum())
    iou = 1.0 if union == 0 else intersection / union
    denominator = predicted + target_pixels
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    precision = (
        1.0 if predicted == 0 and target_pixels == 0 else intersection / max(predicted, 1)
    )
    recall = (
        1.0
        if target_pixels == 0 and predicted == 0
        else intersection / max(target_pixels, 1)
    )
    soft_mae = float(
        np.abs(np.clip(probability, 0.0, 1.0) - truth.astype(np.float32)).mean()
    )
    return BinaryMaskMetrics(
        intersection=intersection,
        union=union,
        predicted_pixels=predicted,
        target_pixels=target_pixels,
        iou=float(iou),
        dice=float(dice),
        precision=float(precision),
        recall=float(recall),
        soft_mae=soft_mae,
    )


def _sample_indices(count: int, maximum: int, seed: int) -> list[int]:
    if count <= 0 or maximum <= 0:
        return []
    if maximum >= count:
        return list(range(count))
    return sorted(random.Random(seed).sample(range(count), maximum))


def _payload_bytes(value: Any, name: str) -> bytes:
    if not isinstance(value, dict) or not isinstance(value.get("bytes"), bytes):
        raise ValueError(f"Parquet column {name!r} must contain Hugging Face image bytes")
    return value["bytes"]


def _selected_rows(parquet_path: Path, indices: list[int]) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("Mask evaluation requires the optional `eval` dependencies") from exc
    selected = set(indices)
    parquet = pq.ParquetFile(parquet_path)
    for index, batch in enumerate(
        parquet.iter_batches(batch_size=1, columns=["image_id", "image", "mask"])
    ):
        if index not in selected:
            continue
        yield batch.to_pylist()[0]


def _sbu_zip_entries(archive_path: Path) -> list[tuple[str, str, str]]:
    image_prefix = "SBU-shadow/SBU-Test/ShadowImages/"
    mask_prefix = "SBU-shadow/SBU-Test/ShadowMasks/"
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    entries: list[tuple[str, str, str]] = []
    for image_name in sorted(
        name
        for name in names
        if name.startswith(image_prefix) and not name.endswith("/")
    ):
        image_id = Path(image_name).stem
        mask_name = f"{mask_prefix}{image_id}.png"
        if mask_name not in names:
            raise ValueError(f"SBU image has no matching mask in archive: {image_name}")
        entries.append((image_id, image_name, mask_name))
    if not entries:
        raise ValueError("No SBU test image/mask pairs were found in the ZIP archive")
    return entries


def _selected_sbu_rows(
    archive_path: Path,
    entries: list[tuple[str, str, str]],
    indices: list[int],
) -> Iterable[dict[str, Any]]:
    with zipfile.ZipFile(archive_path) as archive:
        for index in indices:
            image_id, image_name, mask_name = entries[index]
            yield {
                "image_id": image_id,
                "image": {"bytes": archive.read(image_name), "path": image_name},
                "mask": {"bytes": archive.read(mask_name), "path": mask_name},
            }


def _overlay(
    image: Image.Image,
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    truth = np.asarray(target, dtype=bool)
    if rgb.shape[:2] != truth.shape:
        rgb = cv2.resize(rgb, (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_AREA)
    prediction = cv2.resize(
        (np.asarray(probability) >= threshold).astype(np.uint8),
        (truth.shape[1], truth.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    canvas = rgb.astype(np.float32)
    colors = (
        (np.logical_and(prediction, truth), np.asarray((38, 210, 88))),
        (np.logical_and(prediction, ~truth), np.asarray((245, 72, 72))),
        (np.logical_and(~prediction, truth), np.asarray((55, 130, 245))),
    )
    for mask, color in colors:
        canvas[mask] = 0.48 * canvas[mask] + 0.52 * color
    canvas_u8 = np.clip(canvas, 0, 255).astype(np.uint8)
    cv2.putText(
        canvas_u8,
        "green=correct  red=false positive  blue=missed",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas_u8,
        "green=correct  red=false positive  blue=missed",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (15, 15, 15),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas_u8).save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physics-mask-eval",
        description=(
            "Evaluate automatic shadow/mirror region proposals directly from a "
            "Hugging Face image Parquet or the official SBU ZIP without unpacking "
            "the dataset."
        ),
    )
    parser.add_argument("source", help="Local Parquet or official SBU-shadow ZIP")
    parser.add_argument(
        "--source-format",
        choices=("auto", "hf-parquet", "sbu-zip"),
        default="auto",
    )
    parser.add_argument("--cue", choices=("shadow", "mirror"), required=True)
    parser.add_argument("--backend", choices=("heuristic", "clipseg"), default="clipseg")
    parser.add_argument("--output", default="outputs/mask_eval.json")
    parser.add_argument("--overlays-dir")
    parser.add_argument("--max-overlays", type=int, default=12)
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--max-source-gib", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_CLIPSEG_REVISION,
        help="Pinned CLIPSeg Hub revision",
    )
    parser.add_argument("--device")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dataset-name", default="unidentified-local-parquet")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--dataset-license", default="unknown")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.source).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Evaluation source does not exist: {path}")
    if args.max_images <= 0 or args.max_overlays < 0:
        raise SystemExit("--max-images must be positive and --max-overlays cannot be negative")
    if not 0.0 < args.max_source_gib <= HARD_SOURCE_LIMIT_GIB:
        raise SystemExit(f"--max-source-gib must lie within (0, {HARD_SOURCE_LIMIT_GIB:g}]")
    source_gib = path.stat().st_size / 1024**3
    if source_gib > args.max_source_gib:
        raise SystemExit(
            f"Source is {source_gib:.3f} GiB, above the configured "
            f"{args.max_source_gib:.3f} GiB source cap"
        )
    source_format = args.source_format
    if source_format == "auto":
        source_format = "sbu-zip" if path.suffix.lower() == ".zip" else "hf-parquet"
    if source_format == "sbu-zip":
        entries = _sbu_zip_entries(path)
        row_count = len(entries)

        def selected_rows(selected: list[int]) -> Iterable[dict[str, Any]]:
            return _selected_sbu_rows(path, entries, selected)

    else:
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise SystemExit("Install with `python -m pip install -e '.[eval]'`") from exc
        row_count = pq.ParquetFile(path).metadata.num_rows

        def selected_rows(selected: list[int]) -> Iterable[dict[str, Any]]:
            return _selected_rows(path, selected)

    indices = _sample_indices(row_count, args.max_images, args.seed)
    threshold = args.threshold
    if threshold is None:
        threshold = 0.40 if args.cue == "shadow" else 0.54
    if not 0.0 <= threshold <= 1.0:
        raise SystemExit("--threshold must lie within [0, 1]")
    config = AutomaticProposalConfig(
        enabled=True,
        mask_backend=args.backend,
        mask_revision=args.model_revision or None,
        cache_dir=args.cache_dir,
        device=args.device,
        local_files_only=args.offline,
        allow_model_fallback=False,
    )
    provider = ClipSegMaskProvider(config) if args.backend == "clipseg" else HeuristicMaskProvider()
    records: list[dict[str, Any]] = []
    overlays = Path(args.overlays_dir).expanduser().resolve() if args.overlays_dir else None
    start = time.perf_counter()
    for sample_offset, row in enumerate(selected_rows(indices)):
        image = ImageOps.exif_transpose(
            Image.open(BytesIO(_payload_bytes(row["image"], "image")))
        ).convert("RGB")
        mask_image = Image.open(BytesIO(_payload_bytes(row["mask"], "mask"))).convert("L")
        target = np.asarray(mask_image, dtype=np.uint8) >= 128
        masks = provider.predict(image)
        probability = masks.shadow if args.cue == "shadow" else masks.mirror
        metrics = binary_mask_metrics(probability, target, threshold)
        record = {
            "sample_index": indices[sample_offset],
            "image_id": str(row["image_id"]),
            **asdict(metrics),
        }
        if overlays is not None and sample_offset < args.max_overlays:
            overlay_path = overlays / f"{sample_offset:03d}_{row['image_id']}.png"
            _overlay(image, probability, target, threshold, overlay_path)
            record["overlay_path"] = str(overlay_path)
        records.append(record)
    elapsed = time.perf_counter() - start
    macro_fields = ("iou", "dice", "precision", "recall", "soft_mae")
    macro = {
        field: float(np.mean([record[field] for record in records])) if records else None
        for field in macro_fields
    }
    payload = {
        "report_version": "0.1.0",
        "purpose": "proposal_mask_quality_not_aigc_detection_accuracy",
        "dataset": {
            "name": args.dataset_name,
            "revision": args.dataset_revision,
            "license": args.dataset_license,
            "source_path": str(path),
            "source_format": source_format,
            "source_size_gib": source_gib,
            "row_count": row_count,
        },
        "storage_safety": {
            "streamed_without_extraction": True,
            "configured_source_cap_gib": args.max_source_gib,
            "recommended_source_limit_gib": RECOMMENDED_SOURCE_LIMIT_GIB,
            "hard_source_limit_gib": HARD_SOURCE_LIMIT_GIB,
        },
        "evaluation": {
            "cue": args.cue,
            "backend": args.backend,
            "model": config.mask_model if args.backend == "clipseg" else None,
            "model_revision": (
                config.mask_revision if args.backend == "clipseg" else None
            ),
            "threshold": threshold,
            "seed": args.seed,
            "sample_indices": indices,
            "processed_images": len(records),
            "elapsed_seconds": elapsed,
            "seconds_per_image": elapsed / max(len(records), 1),
            "macro_metrics": macro,
        },
        "records": records,
        "limitations": [
            "This evaluates semantic region masks, not object-shadow association or reflection geometry.",
            "The bounded sample is a smoke test and is not a substitute for a held-out calibrated benchmark.",
            "A dataset's license applies to local evaluation artifacts and must be reviewed before redistribution.",
        ],
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2 if args.pretty else None) + "\n",
        encoding="utf-8",
    )
    print(
        f"Evaluated {len(records)} {args.cue} mask(s) without extraction; "
        f"macro IoU={macro['iou']:.3f}. Report: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
