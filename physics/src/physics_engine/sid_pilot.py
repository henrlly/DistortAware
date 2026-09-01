"""Storage-capped SID_Set sampling and physics-engine evaluation.

SID_Set is roughly 140 GB in full.  This tool streams selected Parquet shards,
keeps only a deterministic per-label reservoir in memory, and refuses source or
extracted byte budgets above explicit safety caps.  It never expands the full
dataset and does not modify the source clone.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .engine import PhysicsEngine
from .integration import _write_json_atomic
from .spatial import SpatialEvidenceError, physics_outlier_mask


GIB = 1024**3
DEFAULT_MAX_SOURCE_BYTES = 10 * GIB
HARD_MAX_SOURCE_BYTES = 50 * GIB
DEFAULT_MAX_EXTRACTED_BYTES = 1 * GIB
HARD_MAX_EXTRACTED_BYTES = 10 * GIB
LABEL_NAMES = {0: "real", 1: "full_synthetic", 2: "tampered"}
FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "TIFF": ".tiff",
}


class SidPilotError(RuntimeError):
    """Raised when a SID pilot cannot proceed safely or reproducibly."""


@dataclass(slots=True)
class SelectedRow:
    source_path: Path
    source_row_index: int
    img_id: str
    label: int
    width: int
    height: int
    image_bytes: bytes
    mask_bytes: bytes | None


def _field_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, dict):
        raw = value.get("bytes")
        return bytes(raw) if isinstance(raw, (bytes, bytearray, memoryview)) else None
    return None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_extension(value: bytes) -> str:
    try:
        with Image.open(io.BytesIO(value)) as image:
            image.verify()
            image_format = image.format
    except (UnidentifiedImageError, OSError) as exc:
        raise SidPilotError(f"SID row contains undecodable image bytes: {exc}") from exc
    extension = FORMAT_EXTENSIONS.get(str(image_format).upper())
    if extension is None:
        raise SidPilotError(f"Unsupported embedded image format: {image_format}")
    return extension


def _safe_identifier(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "image"
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{safe[:48]}_{digest}"


def _validate_source_files(
    shards: list[Path], max_source_bytes: int
) -> tuple[int, list[dict[str, Any]]]:
    if max_source_bytes <= 0 or max_source_bytes > HARD_MAX_SOURCE_BYTES:
        raise SidPilotError(
            f"max_source_bytes must be in (0, {HARD_MAX_SOURCE_BYTES}]"
        )
    if not shards:
        raise SidPilotError("At least one Parquet shard is required")
    metadata: list[dict[str, Any]] = []
    total = 0
    for shard in shards:
        if not shard.is_file():
            raise SidPilotError(f"SID shard does not exist: {shard}")
        size = shard.stat().st_size
        if size < 1024:
            try:
                prefix = shard.read_text(encoding="utf-8")[:80]
            except (OSError, UnicodeDecodeError):
                prefix = ""
            if "git-lfs.github.com/spec" in prefix:
                raise SidPilotError(
                    f"{shard} is only a Git LFS pointer. Download that single shard "
                    "before running the pilot; do not pull the full dataset."
                )
        total += size
        metadata.append({"path": str(shard), "bytes": size})
    if total > max_source_bytes:
        raise SidPilotError(
            f"Selected Parquet sources total {total / GIB:.2f} GiB, exceeding the "
            f"configured {max_source_bytes / GIB:.2f} GiB cap"
        )
    return total, metadata


def reservoir_sample_sid(
    shards: list[Path],
    *,
    per_label: int,
    seed: int,
) -> tuple[dict[int, list[SelectedRow]], dict[str, Any]]:
    """Stream Parquet batches and retain a deterministic reservoir per SID label."""

    if per_label <= 0:
        raise SidPilotError("per_label must be positive")
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SidPilotError(
            "pyarrow is required for the SID pilot. Install `pip install -e '.[sid]'`."
        ) from exc

    reservoirs: dict[int, list[SelectedRow]] = {label: [] for label in LABEL_NAMES}
    seen = Counter()
    decoded_rows = 0
    rngs = {label: random.Random(seed + 1009 * label) for label in LABEL_NAMES}
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        row_offset = 0
        columns = ["img_id", "image", "mask", "width", "height", "label"]
        for batch in parquet.iter_batches(batch_size=32, columns=columns):
            rows = batch.to_pylist()
            for batch_index, raw in enumerate(rows):
                label = int(raw["label"])
                if label not in LABEL_NAMES:
                    continue
                seen[label] += 1
                image_bytes = _field_bytes(raw.get("image"))
                if not image_bytes:
                    continue
                decoded_rows += 1
                candidate = SelectedRow(
                    source_path=shard,
                    source_row_index=row_offset + batch_index,
                    img_id=str(raw.get("img_id") or f"row-{row_offset + batch_index}"),
                    label=label,
                    width=int(raw.get("width") or 0),
                    height=int(raw.get("height") or 0),
                    image_bytes=image_bytes,
                    mask_bytes=_field_bytes(raw.get("mask")) if label == 2 else None,
                )
                reservoir = reservoirs[label]
                if len(reservoir) < per_label:
                    reservoir.append(candidate)
                else:
                    replacement = rngs[label].randrange(seen[label])
                    if replacement < per_label:
                        reservoir[replacement] = candidate
            row_offset += len(rows)
    return reservoirs, {
        "rows_seen_by_label": {
            LABEL_NAMES[label]: int(seen[label]) for label in LABEL_NAMES
        },
        "rows_with_image_bytes": decoded_rows,
    }


def extract_sid_sample(
    shards: list[str | Path],
    *,
    workspace: str | Path,
    per_label: int = 20,
    seed: int = 2026,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    max_extracted_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES,
    dataset_revision: str | None = None,
) -> dict[str, Any]:
    """Extract only the selected in-Parquet images and return a manifest."""

    if max_extracted_bytes <= 0 or max_extracted_bytes > HARD_MAX_EXTRACTED_BYTES:
        raise SidPilotError(
            f"max_extracted_bytes must be in (0, {HARD_MAX_EXTRACTED_BYTES}]"
        )
    paths = [Path(path).expanduser().resolve() for path in shards]
    total_source_bytes, source_metadata = _validate_source_files(paths, max_source_bytes)
    reservoirs, scan = reservoir_sample_sid(paths, per_label=per_label, seed=seed)
    selected = [row for label in sorted(reservoirs) for row in reservoirs[label]]
    if not selected:
        raise SidPilotError("No SID rows with decodable image bytes were selected")
    selected_bytes = sum(
        len(row.image_bytes) + (len(row.mask_bytes) if row.mask_bytes else 0)
        for row in selected
    )
    if selected_bytes > max_extracted_bytes:
        raise SidPilotError(
            f"The selected sample would extract {selected_bytes / GIB:.2f} GiB, "
            f"exceeding the configured {max_extracted_bytes / GIB:.2f} GiB cap"
        )

    root = Path(workspace).expanduser().resolve(strict=False)
    if root.exists() and any(root.iterdir()):
        raise SidPilotError(f"Pilot workspace is not empty: {root}")
    image_root = root / "images"
    mask_root = root / "masks"
    image_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    written_bytes = 0

    for label in sorted(reservoirs):
        label_name = LABEL_NAMES[label]
        label_dir = image_root / label_name
        label_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(reservoirs[label]):
            identifier = _safe_identifier(row.img_id)
            extension = _image_extension(row.image_bytes)
            image_path = label_dir / f"{index:03d}_{identifier}{extension}"
            image_path.write_bytes(row.image_bytes)
            written_bytes += len(row.image_bytes)
            mask_path: Path | None = None
            if row.mask_bytes:
                mask_extension = _image_extension(row.mask_bytes)
                mask_dir = mask_root / label_name
                mask_dir.mkdir(parents=True, exist_ok=True)
                mask_path = mask_dir / f"{index:03d}_{identifier}{mask_extension}"
                mask_path.write_bytes(row.mask_bytes)
                written_bytes += len(row.mask_bytes)
            manifest_rows.append(
                {
                    "img_id": row.img_id,
                    "label": row.label,
                    "label_name": label_name,
                    "source_path": str(row.source_path),
                    "source_row_index": row.source_row_index,
                    "reported_width": row.width,
                    "reported_height": row.height,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path) if mask_path else None,
                    "image_bytes": len(row.image_bytes),
                    "mask_bytes": len(row.mask_bytes) if row.mask_bytes else 0,
                    "image_sha256": _sha256_bytes(row.image_bytes),
                    "mask_sha256": _sha256_bytes(row.mask_bytes) if row.mask_bytes else None,
                }
            )

    for item in source_metadata:
        item["sha256"] = _sha256_file(Path(item["path"]))
    manifest = {
        "manifest_version": "0.2.0",
        "dataset": "saberzl/SID_Set",
        "dataset_revision": dataset_revision,
        "license": "CC-BY-4.0 (per dataset card)",
        "sampling": {
            "method": "deterministic_per_label_reservoir",
            "seed": seed,
            "requested_per_label": per_label,
            "selected_by_label": {
                LABEL_NAMES[label]: len(reservoirs[label]) for label in LABEL_NAMES
            },
            "selected_by_source_file": dict(
                sorted(Counter(item["source_path"] for item in manifest_rows).items())
            ),
        },
        "storage": {
            "source_bytes": total_source_bytes,
            "extracted_bytes": written_bytes,
            "configured_max_source_bytes": max_source_bytes,
            "configured_max_extracted_bytes": max_extracted_bytes,
            "hard_max_source_bytes": HARD_MAX_SOURCE_BYTES,
            "hard_max_extracted_bytes": HARD_MAX_EXTRACTED_BYTES,
        },
        "source_files": source_metadata,
        "scan": scan,
        "images": manifest_rows,
    }
    _write_json_atomic(root / "manifest.json", manifest, pretty=True)
    return manifest


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _validate_existing_workspace(
    manifest: dict[str, Any],
    workspace: Path,
    *,
    max_source_bytes: int,
    max_extracted_bytes: int,
) -> None:
    if not 0 < max_source_bytes <= HARD_MAX_SOURCE_BYTES:
        raise SidPilotError(
            f"max_source_bytes must be in (0, {HARD_MAX_SOURCE_BYTES}]"
        )
    if not 0 < max_extracted_bytes <= HARD_MAX_EXTRACTED_BYTES:
        raise SidPilotError(
            f"max_extracted_bytes must be in (0, {HARD_MAX_EXTRACTED_BYTES}]"
        )
    images = manifest.get("images")
    storage = manifest.get("storage")
    if not isinstance(images, list) or not images or not isinstance(storage, dict):
        raise SidPilotError("Existing SID manifest is malformed")
    source_bytes = int(storage.get("source_bytes", -1))
    recorded_extracted = int(storage.get("extracted_bytes", -1))
    if source_bytes < 0 or source_bytes > max_source_bytes:
        raise SidPilotError("Existing SID manifest exceeds the configured source cap")

    selected_files: set[Path] = set()
    for item in images:
        if not isinstance(item, dict):
            raise SidPilotError("Existing SID manifest contains a malformed image record")
        for field in ("image_path", "mask_path"):
            raw_path = item.get(field)
            if raw_path is None and field == "mask_path":
                continue
            if not isinstance(raw_path, str):
                raise SidPilotError(f"Existing SID manifest has an invalid {field}")
            path = Path(raw_path).expanduser().resolve()
            if not path.is_relative_to(workspace):
                raise SidPilotError(
                    f"Existing SID manifest points outside its capped workspace: {path}"
                )
            if not path.is_file():
                raise SidPilotError(f"Existing SID extracted file is missing: {path}")
            selected_files.add(path)
    actual_extracted = sum(path.stat().st_size for path in selected_files)
    if actual_extracted != recorded_extracted:
        raise SidPilotError(
            "Existing SID extracted bytes do not match the recorded manifest"
        )
    if actual_extracted > max_extracted_bytes:
        raise SidPilotError("Existing SID workspace exceeds the configured extraction cap")


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any] | None:
    """Two-sided Wilson score interval for a binomial rate."""

    if total <= 0:
        return None
    successes = max(0, min(int(successes), int(total)))
    n = float(total)
    proportion = successes / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)
        )
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - half_width)
    upper = 1.0 if successes == total else min(1.0, center + half_width)
    return {
        "successes": successes,
        "total": total,
        "rate": proportion,
        "confidence_level": 0.95,
        "lower": lower,
        "upper": upper,
        "method": "wilson_score",
    }


def _mask_overlap(
    *,
    mask_path: Path,
    perspective_cue: dict[str, Any],
    image_width: int,
    image_height: int,
    grid_shape: tuple[int, int] = (32, 32),
) -> dict[str, Any] | None:
    try:
        physics_mask, diagnostics = physics_outlier_mask(
            perspective_cue,
            image_width=image_width,
            image_height=image_height,
            grid_shape=grid_shape,
        )
    except SpatialEvidenceError:
        return None
    if not physics_mask.any():
        return None
    try:
        with Image.open(mask_path) as opened:
            resized = opened.convert("L").resize(
                (grid_shape[1], grid_shape[0]), Image.Resampling.NEAREST
            )
            tamper_mask = np.asarray(resized) > 0
    except (UnidentifiedImageError, OSError):
        return None
    if not tamper_mask.any():
        return None
    intersection = int(np.logical_and(physics_mask, tamper_mask).sum())
    union = int(np.logical_or(physics_mask, tamper_mask).sum())
    physics_count = int(physics_mask.sum())
    tamper_count = int(tamper_mask.sum())
    return {
        "intersection_patches": intersection,
        "physics_outlier_patches": physics_count,
        "tamper_patches": tamper_count,
        "precision_physics_inside_tamper": intersection / physics_count,
        "recall_tamper_covered_by_physics": intersection / tamper_count,
        "iou": intersection / union if union else None,
        "geometry": diagnostics,
    }


def evaluate_sid_pilot(
    manifest: dict[str, Any],
    *,
    workspace: str | Path,
) -> dict[str, Any]:
    """Run the physics engine and summarize explanation safety by SID label."""

    root = Path(workspace).expanduser().resolve()
    image_root = root / "images"
    batch = PhysicsEngine().run(image_root, recursive=True)
    batch_payload = batch.to_dict()
    _write_json_atomic(root / "physics_results.json", batch_payload, pretty=True)

    metadata = {str(Path(item["image_path"]).resolve()): item for item in manifest["images"]}
    review_seed = int(manifest.get("sampling", {}).get("seed", 2026))
    double_review_paths: set[str] = set()
    for label_name in LABEL_NAMES.values():
        candidates = [
            item for item in manifest["images"] if item["label_name"] == label_name
        ]
        ranked = sorted(
            candidates,
            key=lambda item: hashlib.sha256(
                (
                    f"{review_seed}\0{item['img_id']}\0{item['source_path']}\0"
                    f"{item['source_row_index']}"
                ).encode("utf-8")
            ).digest(),
        )
        target_count = math.ceil(0.20 * len(candidates))
        double_review_paths.update(
            str(Path(item["image_path"]).resolve()) for item in ranked[:target_count]
        )
    by_label: dict[str, list[dict[str, Any]]] = {name: [] for name in LABEL_NAMES.values()}
    localization: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    for image_result in batch_payload["images"]:
        item = metadata.get(str(Path(image_result["image_path"]).resolve()))
        if item is None:
            raise SidPilotError(f"Physics output was not present in the SID manifest: {image_result['image_path']}")
        by_label[item["label_name"]].append(image_result)
        perspective_cue = image_result["cues"]["perspective"]
        measurements = perspective_cue.get("measurements", {})
        review_items.append(
            {
                "img_id": item["img_id"],
                "label_name": item["label_name"],
                "image_path": item["image_path"],
                "mask_path": item.get("mask_path"),
                "source_path": item["source_path"],
                "source_row_index": item["source_row_index"],
                "independent_double_review_target": (
                    str(Path(item["image_path"]).resolve()) in double_review_paths
                ),
                "physics_prefill": {
                    "perspective_applicable": perspective_cue.get("applicable"),
                    "perspective_status": perspective_cue.get("status"),
                    "perspective_violation_score": perspective_cue.get("violation_score"),
                    "line_count": measurements.get("retained_count"),
                    "orientation_peak_fraction": measurements.get(
                        "orientation_peak_fraction"
                    ),
                    "orientation_entropy": measurements.get("orientation_entropy"),
                    "structural_applicability_gate": measurements.get(
                        "structural_applicability_gate"
                    ),
                    "multi_view_stability": measurements.get("multi_view_stability"),
                },
                "human_review": {
                    "scene_type": None,
                    "structured_geometry": None,
                    "visible_cast_shadows": None,
                    "planar_reflection": None,
                    "screenshot_or_composite": None,
                    "cgi_or_illustration": None,
                    "reviewer_notes": None,
                },
            }
        )
        if item["label_name"] == "tampered" and item.get("mask_path"):
            overlap = _mask_overlap(
                mask_path=Path(item["mask_path"]),
                perspective_cue=image_result["cues"]["perspective"],
                image_width=int(image_result["width"]),
                image_height=int(image_result["height"]),
            )
            if overlap is not None:
                localization.append({"img_id": item["img_id"], **overlap})

    label_summaries: dict[str, Any] = {}
    for label_name, results in by_label.items():
        perspective = [result["cues"]["perspective"] for result in results]
        applicable = [cue for cue in perspective if cue.get("applicable")]
        scores = [float(cue["violation_score"]) for cue in applicable if cue.get("violation_score") is not None]
        status_counts = Counter(str(cue.get("status")) for cue in perspective)
        inconsistent = sum(cue.get("status") == "inconsistent" for cue in applicable)
        indeterminate = sum(cue.get("status") == "indeterminate" for cue in applicable)
        label_summaries[label_name] = {
            "images": len(results),
            "perspective_applicable": len(applicable),
            "perspective_applicability_rate": len(applicable) / len(results) if results else None,
            "perspective_applicability_ci95": _wilson_interval(
                len(applicable), len(results)
            ),
            "perspective_status_counts": dict(status_counts),
            "mean_perspective_violation_score_when_applicable": _safe_mean(scores),
            "displayed_inconsistency_rate_when_applicable": inconsistent / len(applicable) if applicable else None,
            "displayed_inconsistency_ci95_when_applicable": _wilson_interval(
                inconsistent, len(applicable)
            ),
            "displayed_inconsistency_rate_all_images": inconsistent / len(results) if results else None,
            "displayed_inconsistency_ci95_all_images": _wilson_interval(
                inconsistent, len(results)
            ),
            "indeterminate_rate_when_applicable": indeterminate / len(applicable) if applicable else None,
            "indeterminate_ci95_when_applicable": _wilson_interval(
                indeterminate, len(applicable)
            ),
            "shadow_applicable": sum(result["cues"]["cast_shadow"]["applicable"] for result in results),
            "reflection_applicable": sum(result["cues"]["reflection"]["applicable"] for result in results),
        }

    localization_summary = {
        "evaluable_tampered_images": len(localization),
        "mean_precision_physics_inside_tamper": _safe_mean(
            [float(item["precision_physics_inside_tamper"]) for item in localization]
        ),
        "mean_recall_tamper_covered_by_physics": _safe_mean(
            [float(item["recall_tamper_covered_by_physics"]) for item in localization]
        ),
        "mean_iou": _safe_mean([float(item["iou"]) for item in localization if item["iou"] is not None]),
        "images": localization,
    }
    review_queue = {
        "review_queue_version": "0.2.0",
        "instructions": {
            "purpose": "Complete scene stratification before drawing explanation-safety conclusions.",
            "allowed_scene_types": [
                "indoor",
                "outdoor",
                "architecture",
                "portrait",
                "product",
                "food",
                "landscape",
                "document_or_text",
                "other",
            ],
            "boolean_fields": "Use true, false, or null when genuinely indeterminate.",
            "label_warning": "Do not infer review fields from the SID class label.",
            "independent_review_protocol": (
                "Two reviewers should annotate target images independently, using distinct "
                "reviewer IDs and without seeing each other's points. The deterministic 20% "
                "target is a minimum starting set; continue reviewing until at least 20% of "
                "cases judged shadow/reflection-applicable by either reviewer are double-reviewed."
            ),
            "agreement_command": "physics-review-agreement",
        },
        "images": sorted(review_items, key=lambda item: (item["label_name"], item["img_id"])),
    }
    review_queue_path = root / "review_queue.json"
    _write_json_atomic(review_queue_path, review_queue, pretty=True)

    return {
        "report_version": "0.2.0",
        "engine_version": batch.engine_version,
        "manifest_path": str(root / "manifest.json"),
        "physics_results_path": str(root / "physics_results.json"),
        "sample_size": len(manifest["images"]),
        "source_shard_count": len(manifest.get("source_files", [])),
        "review_queue_path": str(review_queue_path),
        "independent_double_review_target_count": len(double_review_paths),
        "by_label": label_summaries,
        "tamper_mask_localization": localization_summary,
        "interpretation": {
            "primary_detector_role": "none; this report evaluates physics explanations only",
            "real_image_safety_metric": "displayed_inconsistency_rate_when_applicable",
            "warning": (
                "Label separation is descriptive, not a trained accuracy estimate. "
                "Physical inconsistency and AIGC labels are not equivalent."
            ),
            "annotation_scope": (
                "No reviewed shadow/reflection correspondences were supplied, so those "
                "cues correctly remain not_applicable."
            ),
        },
    }


def write_markdown_report(path: Path, report: dict[str, Any], manifest: dict[str, Any]) -> None:
    storage = manifest["storage"]
    source_shards = len(manifest.get("source_files", []))
    lines = [
        "# SID_Set physics pilot",
        "",
        f"- Sample: **{report['sample_size']} images** from **{source_shards} streamed Parquet shard(s)**",
        f"- Source bytes used: **{storage['source_bytes'] / GIB:.3f} GiB**",
        f"- Extracted bytes: **{storage['extracted_bytes'] / (1024**2):.1f} MiB**",
        "- Physics role: **explanation evidence only; not an AIGC classifier**",
        "",
        "| SID label | Images | Perspective applicable | Inconsistent / applicable | 95% CI | Mean violation score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label_name in ("real", "full_synthetic", "tampered"):
        row = report["by_label"][label_name]
        rate = row["perspective_applicability_rate"]
        inconsistent = row["displayed_inconsistency_rate_when_applicable"]
        interval = row["displayed_inconsistency_ci95_when_applicable"]
        mean_score = row["mean_perspective_violation_score_when_applicable"]
        interval_text = (
            "n/a"
            if interval is None
            else f"{interval['lower']:.1%}–{interval['upper']:.1%}"
        )
        lines.append(
            f"| {label_name} | {row['images']} | "
            f"{'n/a' if rate is None else f'{rate:.1%}'} | "
            f"{'n/a' if inconsistent is None else f'{inconsistent:.1%}'} | "
            f"{interval_text} | "
            f"{'n/a' if mean_score is None else f'{mean_score:.3f}'} |"
        )
    localization = report["tamper_mask_localization"]
    precision = localization["mean_precision_physics_inside_tamper"]
    recall = localization["mean_recall_tamper_covered_by_physics"]
    mean_iou = localization["mean_iou"]
    precision_text = "n/a" if precision is None else f"{precision:.3f}"
    recall_text = "n/a" if recall is None else f"{recall:.3f}"
    iou_text = "n/a" if mean_iou is None else f"{mean_iou:.3f}"
    lines.extend(
        [
            "",
            "## Tamper-mask association",
            "",
            f"Physics outlier geometry and a non-empty SID tamper mask were jointly evaluable for **{localization['evaluable_tampered_images']}** images.",
            "",
            "| Diagnostic | Mean |",
            "|---|---:|",
            f"| Physics-residual patches inside mask | {precision_text} |",
            f"| Tamper mask covered by physics residuals | {recall_text} |",
            f"| Sparse-grid IoU | {iou_text} |",
            "",
            "This sparse line/mask overlap is diagnostic only; it is not segmentation IoU for a trained localizer. Globally consistent cues do not become suspicious DINO explanations.",
            "",
            "## Interpretation guardrails",
            "",
            "- A real image flagged physically inconsistent is a false *explanation* for this demo, not necessarily a detector false positive.",
            "- A generated or tampered image can be physically coherent, so a consistent result is not a detector false negative.",
            "- Shadow and reflection remain `not_applicable` without reviewed correspondences; missing cues are neutral.",
            f"- Scene-stratification fields are prepared in `{report['review_queue_path']}` and require human review before subgroup claims.",
            f"- The sample is a deterministic pilot from {source_shards} shard(s), not a representative SID_Set benchmark.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physics-sid-pilot",
        description="Stream a storage-capped SID_Set sample and evaluate physics explanations.",
    )
    parser.add_argument(
        "--shard", action="append", help="Downloaded SID Parquet shard; repeat for more"
    )
    parser.add_argument("--workspace", default="outputs/sid_pilot")
    parser.add_argument("--per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-source-gib", type=float, default=10.0)
    parser.add_argument("--max-extracted-gib", type=float, default=1.0)
    parser.add_argument("--dataset-revision")
    parser.add_argument(
        "--evaluate-existing",
        action="store_true",
        help="Re-run physics on an existing capped workspace without reading Parquet again",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        workspace = Path(args.workspace).expanduser().resolve()
        max_source = int(args.max_source_gib * GIB)
        max_extracted = int(args.max_extracted_gib * GIB)
        if args.evaluate_existing:
            if args.shard:
                raise SidPilotError("--evaluate-existing cannot be combined with --shard")
            manifest_path = workspace / "manifest.json"
            if not manifest_path.is_file():
                raise SidPilotError(
                    f"Existing workspace has no manifest.json: {workspace}"
                )
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if not isinstance(manifest, dict) or not isinstance(
                manifest.get("images"), list
            ):
                raise SidPilotError("Existing SID manifest is malformed")
            _validate_existing_workspace(
                manifest,
                workspace,
                max_source_bytes=max_source,
                max_extracted_bytes=max_extracted,
            )
        else:
            if not args.shard:
                raise SidPilotError("At least one --shard is required for extraction")
            manifest = extract_sid_sample(
                args.shard,
                workspace=workspace,
                per_label=args.per_label,
                seed=args.seed,
                max_source_bytes=max_source,
                max_extracted_bytes=max_extracted,
                dataset_revision=args.dataset_revision,
            )
        report = evaluate_sid_pilot(manifest, workspace=args.workspace)
        _write_json_atomic(workspace / "report.json", report, pretty=True)
        write_markdown_report(workspace / "report.md", report, manifest)
    except (OSError, ValueError, SidPilotError, json.JSONDecodeError) as exc:
        print(f"physics-sid-pilot: {exc}", file=sys.stderr)
        return 2
    print(
        f"SID pilot completed for {report['sample_size']} image(s); "
        f"source {manifest['storage']['source_bytes'] / GIB:.3f} GiB, "
        f"extracted {manifest['storage']['extracted_bytes'] / (1024**2):.1f} MiB. "
        f"Report: {workspace / 'report.md'}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
