"""Transformation robustness harness for physics evidence.

The transform names and severities mirror the official DID repository so both
components can be discussed against the same real-world degradation suite.
The clean result for each source image/cue is the reference; this harness
measures applicability loss, hard status flips, and score drift.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .annotations import AnnotationStore, ImageAnnotations
from .automatic_config import AutomaticProposalConfig
from .engine import PhysicsEngine, PhysicsEngineConfig, SUPPORTED_EXTENSIONS


Point = tuple[float, float]
PointMapper = Callable[[Point, int, int], Point | None]


@dataclass(frozen=True, slots=True)
class RobustnessTransform:
    name: str
    apply: Callable[[Image.Image, np.random.Generator], Image.Image]
    map_point: PointMapper


def _identity_point(point: Point, _width: int, _height: int) -> Point:
    return point


def _clean(image: Image.Image, _rng: np.random.Generator) -> Image.Image:
    return image.copy()


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def _resize_cycle(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    small = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.BICUBIC,
    )
    return small.resize((width, height), Image.Resampling.BICUBIC)


def _noise(
    image: Image.Image, sigma: float, rng: np.random.Generator
) -> Image.Image:
    values = np.asarray(image, dtype=np.float32) / 255.0
    values = values + rng.normal(0.0, sigma, values.shape)
    return Image.fromarray((np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8))


def _jitter(image: Image.Image) -> Image.Image:
    # Match the official detector suite's deterministic RandomState(0) factors.
    random_state = np.random.RandomState(0)
    result = image
    for enhancer in (
        ImageEnhance.Brightness,
        ImageEnhance.Contrast,
        ImageEnhance.Color,
    ):
        factor = 1.0 + random_state.uniform(-0.2, 0.2)
        result = enhancer(result).enhance(float(factor))
    return result


def _crop_box(width: int, height: int, fraction: float = 0.8) -> tuple[int, int, int, int]:
    crop_width = max(1, int(width * fraction))
    crop_height = max(1, int(height * fraction))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return left, top, left + crop_width, top + crop_height


def _center_crop(image: Image.Image, fraction: float = 0.8) -> Image.Image:
    width, height = image.size
    return image.crop(_crop_box(width, height, fraction)).resize(
        (width, height), Image.Resampling.BICUBIC
    )


def _crop_point(point: Point, width: int, height: int) -> Point | None:
    left, top, right, bottom = _crop_box(width, height)
    x, y = point
    if not (left <= x < right and top <= y < bottom):
        return None
    return (
        (x - left) * width / (right - left),
        (y - top) * height / (bottom - top),
    )


TRANSFORMS: tuple[RobustnessTransform, ...] = (
    RobustnessTransform("clean", _clean, _identity_point),
    *tuple(
        RobustnessTransform(
            f"jpeg{quality}",
            lambda image, _rng, quality=quality: _jpeg(image, quality),
            _identity_point,
        )
        for quality in (90, 70, 50, 30)
    ),
    *tuple(
        RobustnessTransform(
            f"blur{sigma}",
            lambda image, _rng, sigma=sigma: _blur(image, sigma),
            _identity_point,
        )
        for sigma in (0.5, 1.0, 2.0)
    ),
    *tuple(
        RobustnessTransform(
            f"resize{scale}",
            lambda image, _rng, scale=scale: _resize_cycle(image, scale),
            _identity_point,
        )
        for scale in (0.5, 0.25)
    ),
    *tuple(
        RobustnessTransform(
            f"noise{sigma:.2f}",
            lambda image, rng, sigma=sigma: _noise(image, sigma, rng),
            _identity_point,
        )
        for sigma in (0.02, 0.05, 0.10)
    ),
    RobustnessTransform("jitter", lambda image, _rng: _jitter(image), _identity_point),
    RobustnessTransform(
        "crop80", lambda image, _rng: _center_crop(image, 0.8), _crop_point
    ),
)


def _iter_source_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _stable_seed(source_key: str, transform: str) -> int:
    digest = hashlib.sha256(f"{source_key}\0{transform}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _safe_stem(relative_path: Path) -> str:
    raw = "__".join(relative_path.with_suffix("").parts)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._") or "image"
    digest = hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{safe[:72]}__{digest}"


def _map_annotations(
    annotations: ImageAnnotations,
    transform: RobustnessTransform,
    width: int,
    height: int,
) -> dict[str, Any]:
    perspective_regions: list[dict[str, Any]] = []
    for region in annotations.perspective_regions:
        left, top, right, bottom = region.xyxy
        first = transform.map_point((left, top), width, height)
        second = transform.map_point((right, bottom), width, height)
        if first is not None and second is not None:
            perspective_regions.append(
                {
                    "xyxy": [first[0], first[1], second[0], second[1]],
                    "confidence": region.confidence,
                }
            )

    shadow_pairs: list[dict[str, Any]] = []
    for pair in annotations.shadow_pairs:
        contact = transform.map_point(pair.object_contact, width, height)
        tip = transform.map_point(pair.shadow_tip, width, height)
        if contact is not None and tip is not None:
            shadow_pairs.append(
                {
                    "object_contact": list(contact),
                    "shadow_tip": list(tip),
                    "confidence": pair.confidence,
                }
            )

    reflection_pairs: list[dict[str, Any]] = []
    for pair in annotations.reflection_pairs:
        visible = transform.map_point(pair.object_point, width, height)
        reflected = transform.map_point(pair.reflection_point, width, height)
        if visible is not None and reflected is not None:
            reflection_pairs.append(
                {
                    "object_point": list(visible),
                    "reflection_point": list(reflected),
                    "confidence": pair.confidence,
                }
            )

    return {
        "coordinate_space": "pixels",
        "perspective": {"regions": perspective_regions},
        "cast_shadow": {
            "applicability": annotations.shadow_applicability,
            "pairs": shadow_pairs,
        },
        "reflection": {
            "applicability": annotations.reflection_applicability,
            "pairs": reflection_pairs,
        },
    }


def _primitive_row(
    *,
    source_image: str,
    transform: str,
    cue_name: str,
    cue: Any,
) -> dict[str, Any]:
    return {
        "source_image": source_image,
        "transform": transform,
        "cue": cue_name,
        "applicable": bool(cue.applicable),
        "status": cue.status,
        "violation_score": cue.violation_score,
        "confidence": cue.confidence,
        "measurements": dict(cue.measurements),
        "errors": [],
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    min_applicability_retention: float = 0.9,
    max_hard_flips: int = 0,
    max_mean_score_drift: float = 0.2,
    max_score_drift: float = 0.2,
) -> dict[str, Any]:
    baselines = {
        (row["source_image"], row["cue"]): row
        for row in rows
        if row["transform"] == "clean"
    }
    for row in rows:
        baseline = baselines.get((row["source_image"], row["cue"]))
        row["baseline_applicable"] = None if baseline is None else baseline["applicable"]
        row["baseline_status"] = None if baseline is None else baseline["status"]
        row["baseline_violation_score"] = (
            None if baseline is None else baseline["violation_score"]
        )
        comparable = (
            baseline is not None
            and row["transform"] != "clean"
            and bool(baseline["applicable"])
        )
        row["applicability_retained"] = (
            bool(row["applicable"]) if comparable else None
        )
        if comparable and row["applicable"]:
            definitive = {baseline["status"], row["status"]} == {
                "consistent",
                "inconsistent",
            }
            row["hard_status_flip"] = definitive
            if (
                baseline["violation_score"] is not None
                and row["violation_score"] is not None
            ):
                row["absolute_score_drift"] = abs(
                    float(row["violation_score"])
                    - float(baseline["violation_score"])
                )
            else:
                row["absolute_score_drift"] = None
        else:
            row["hard_status_flip"] = None
            row["absolute_score_drift"] = None

    cue_names = sorted({str(row["cue"]) for row in rows})
    per_cue: dict[str, dict[str, Any]] = {}
    all_evaluable: list[dict[str, Any]] = []
    for cue_name in cue_names:
        evaluable = [
            row
            for row in rows
            if row["cue"] == cue_name and row["applicability_retained"] is not None
        ]
        retained = [row for row in evaluable if row["applicability_retained"]]
        drifts = [
            float(row["absolute_score_drift"])
            for row in retained
            if row["absolute_score_drift"] is not None
        ]
        hard_flips = sum(bool(row["hard_status_flip"]) for row in retained)
        status_changes = sum(
            row["status"] != row["baseline_status"] for row in retained
        )
        per_cue[cue_name] = {
            "evaluable_transform_cases": len(evaluable),
            "applicability_retained_cases": len(retained),
            "applicability_retention_rate": (
                len(retained) / len(evaluable) if evaluable else None
            ),
            "hard_status_flips": hard_flips,
            "status_changes": status_changes,
            "mean_absolute_score_drift": float(np.mean(drifts)) if drifts else None,
            "max_absolute_score_drift": float(np.max(drifts)) if drifts else None,
        }
        all_evaluable.extend(evaluable)

    retained_all = [row for row in all_evaluable if row["applicability_retained"]]
    all_drifts = [
        float(row["absolute_score_drift"])
        for row in retained_all
        if row["absolute_score_drift"] is not None
    ]
    overall_retention = (
        len(retained_all) / len(all_evaluable) if all_evaluable else None
    )
    overall_hard_flips = sum(bool(row["hard_status_flip"]) for row in retained_all)
    overall_status_changes = sum(
        row["status"] != row["baseline_status"] for row in retained_all
    )
    overall_mean_drift = float(np.mean(all_drifts)) if all_drifts else None
    overall_max_drift = float(np.max(all_drifts)) if all_drifts else None

    checks = {
        "applicability_retention": (
            overall_retention is None
            or overall_retention >= min_applicability_retention
        ),
        "hard_status_flips": overall_hard_flips <= max_hard_flips,
        "mean_score_drift": (
            overall_mean_drift is None
            or overall_mean_drift <= max_mean_score_drift
        ),
        "maximum_score_drift": (
            overall_max_drift is None or overall_max_drift <= max_score_drift
        ),
    }
    return {
        "per_cue": per_cue,
        "overall": {
            "evaluable_transform_cases": len(all_evaluable),
            "applicability_retained_cases": len(retained_all),
            "applicability_retention_rate": overall_retention,
            "hard_status_flips": overall_hard_flips,
            "status_changes": overall_status_changes,
            "mean_absolute_score_drift": overall_mean_drift,
            "max_absolute_score_drift": overall_max_drift,
        },
        "acceptance": {
            "thresholds": {
                "min_applicability_retention": min_applicability_retention,
                "max_hard_status_flips": max_hard_flips,
                "max_mean_absolute_score_drift": max_mean_score_drift,
                "max_absolute_score_drift": max_score_drift,
            },
            "checks": checks,
            "passed": all(checks.values()),
        },
    }


def run_battle_test(
    input_path: str | Path,
    *,
    annotations_path: str | Path | None = None,
    recursive: bool = False,
    max_images: int | None = None,
    workspace: str | Path,
    min_applicability_retention: float = 0.9,
    max_hard_flips: int = 0,
    max_mean_score_drift: float = 0.2,
    max_score_drift: float = 0.2,
    engine_config: PhysicsEngineConfig | None = None,
) -> dict[str, Any]:
    root = Path(input_path).expanduser().resolve()
    sources = _iter_source_images(root, recursive)
    if max_images is not None:
        sources = sources[:max_images]
    if not sources:
        raise ValueError("No supported images were found for battle testing")

    work_root = Path(workspace)
    image_dir = work_root / "transformed_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_store = AnnotationStore.from_path(annotations_path)
    generated_annotations: dict[str, Any] = {
        "coordinate_space": "pixels",
        "images": {},
    }
    metadata: dict[str, tuple[str, str]] = {}

    for source in sources:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        source_root = root if root.is_dir() else source.parent
        relative = source.relative_to(source_root)
        source_key = relative.as_posix()
        annotations = annotation_store.for_image(
            source, source_root, width=width, height=height
        )
        safe_stem = _safe_stem(relative)
        for transform in TRANSFORMS:
            rng = np.random.default_rng(_stable_seed(source_key, transform.name))
            transformed = transform.apply(image, rng).convert("RGB")
            output_name = f"{safe_stem}__{transform.name}.png"
            transformed.save(image_dir / output_name, format="PNG")
            generated_annotations["images"][output_name] = _map_annotations(
                annotations, transform, width, height
            )
            metadata[output_name] = (source_key, transform.name)

    annotation_path = work_root / "transformed_annotations.json"
    with annotation_path.open("w", encoding="utf-8") as handle:
        json.dump(generated_annotations, handle, indent=2)
        handle.write("\n")

    started = time.perf_counter()
    batch = PhysicsEngine(engine_config).run(image_dir, annotations_path=annotation_path)
    elapsed = time.perf_counter() - started

    rows: list[dict[str, Any]] = []
    for image_result in batch.images:
        output_name = Path(image_result.image_path).name
        source_key, transform_name = metadata[output_name]
        for cue_name, cue in image_result.cues.items():
            row = _primitive_row(
                source_image=source_key,
                transform=transform_name,
                cue_name=cue_name,
                cue=cue,
            )
            row["errors"] = list(image_result.errors)
            rows.append(row)

    summary = summarize_rows(
        rows,
        min_applicability_retention=min_applicability_retention,
        max_hard_flips=max_hard_flips,
        max_mean_score_drift=max_mean_score_drift,
        max_score_drift=max_score_drift,
    )
    summary["runtime"] = {
        "seconds": elapsed,
        "transformed_images": len(batch.images),
        "images_per_second": len(batch.images) / elapsed if elapsed > 0 else None,
    }
    return {
        "report_version": "0.1.0",
        "engine_version": batch.engine_version,
        "input_root": str(root),
        "source_image_count": len(sources),
        "transform_names": [transform.name for transform in TRANSFORMS],
        "automatic_proposals": {
            "enabled": bool(engine_config and engine_config.automatic.enabled),
            "mask_backend": (
                engine_config.automatic.mask_backend if engine_config else "heuristic"
            ),
            "feature_backend": (
                engine_config.automatic.feature_backend if engine_config else "appearance"
            ),
            "object_backend": (
                engine_config.automatic.object_backend if engine_config else "edges"
            ),
        },
        "summary": summary,
        "rows": rows,
    }


def write_report(output_path: str | Path, report: dict[str, Any]) -> tuple[Path, Path, Path]:
    json_path = Path(output_path).expanduser().resolve(strict=False)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = json_path.with_suffix(".csv")
    markdown_path = json_path.with_suffix(".md")

    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(json_path)

    fieldnames = [
        "source_image",
        "transform",
        "cue",
        "applicable",
        "status",
        "violation_score",
        "confidence",
        "baseline_applicable",
        "baseline_status",
        "baseline_violation_score",
        "applicability_retained",
        "hard_status_flip",
        "absolute_score_drift",
        "errors",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["rows"]:
            output = dict(row)
            output["errors"] = " | ".join(output.get("errors", []))
            writer.writerow({field: output.get(field) for field in fieldnames})

    summary = report["summary"]
    acceptance = summary["acceptance"]
    lines = [
        "# Physics transformation battle test",
        "",
        f"- Source images: **{report['source_image_count']}**",
        f"- Transformed evaluations: **{summary['runtime']['transformed_images']}**",
        f"- Runtime: **{summary['runtime']['seconds']:.2f} s**",
        f"- Acceptance: **{'PASS' if acceptance['passed'] else 'FAIL'}**",
        "",
        "| Cue | Evaluable | Applicability retained | Status changes | Hard flips | Mean score drift | Max score drift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cue_name, cue in summary["per_cue"].items():
        retention = cue["applicability_retention_rate"]
        mean_drift = cue["mean_absolute_score_drift"]
        max_drift = cue["max_absolute_score_drift"]
        lines.append(
            f"| {cue_name} | {cue['evaluable_transform_cases']} | "
            f"{'n/a' if retention is None else f'{retention:.1%}'} | "
            f"{cue['status_changes']} | "
            f"{cue['hard_status_flips']} | "
            f"{'n/a' if mean_drift is None else f'{mean_drift:.3f}'} | "
            f"{'n/a' if max_drift is None else f'{max_drift:.3f}'} |"
        )
    lines.extend(
        [
            "",
            "A hard flip means `consistent` became `inconsistent`, or vice versa. "
            "A move to `indeterminate` is recorded through score drift but is not counted as a hard flip.",
            "",
            "These results measure stability on the supplied images. They are not an accuracy estimate on real-world AIGC data.",
            "",
            (
                "Shadow and reflection correspondences were proposed again from each "
                "transformed image; applicability loss and drift therefore include the "
                "automatic region and matching stages."
                if report.get("automatic_proposals", {}).get("enabled")
                else "Shadow and reflection scores use reviewed geometry after the points "
                "are transformed. Their zero drift demonstrates coordinate/schema "
                "stability, not automatic pixel-level shadow or reflection detection."
            ),
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physics-battle-test",
        description="Measure physics-cue stability across the official 14-transform suite.",
    )
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("--annotations", help="Optional annotation file or directory")
    parser.add_argument(
        "--output",
        default="outputs/battle_test.json",
        help="JSON report path; CSV and Markdown are written beside it",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--min-applicability-retention", type=float, default=0.9)
    parser.add_argument("--max-hard-flips", type=int, default=0)
    parser.add_argument("--max-mean-score-drift", type=float, default=0.2)
    parser.add_argument("--max-score-drift", type=float, default=0.2)
    parser.add_argument("--auto-proposals", action="store_true")
    parser.add_argument(
        "--proposal-mask-backend",
        choices=("heuristic", "clipseg"),
        default="heuristic",
    )
    parser.add_argument(
        "--proposal-feature-backend",
        choices=("appearance", "dinov3", "none"),
        default="appearance",
    )
    parser.add_argument(
        "--proposal-object-backend",
        choices=("edges", "torchvision", "none"),
        default="edges",
    )
    parser.add_argument("--proposal-cache-dir")
    parser.add_argument("--proposal-device")
    parser.add_argument("--proposal-offline", action="store_true")
    parser.add_argument("--strict-proposal-models", action="store_true")
    parser.add_argument(
        "--strict", action="store_true", help="Return status 1 if acceptance checks fail"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_images is not None and args.max_images <= 0:
        raise SystemExit("--max-images must be positive")
    if not 0.0 <= args.min_applicability_retention <= 1.0:
        raise SystemExit("--min-applicability-retention must lie within [0, 1]")
    if (
        args.max_hard_flips < 0
        or args.max_mean_score_drift < 0
        or args.max_score_drift < 0
    ):
        raise SystemExit("Battle-test thresholds cannot be negative")

    output_path = Path(args.output).expanduser().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    automatic = AutomaticProposalConfig(
        enabled=args.auto_proposals,
        mask_backend=args.proposal_mask_backend,
        feature_backend=args.proposal_feature_backend,
        object_backend=args.proposal_object_backend,
        cache_dir=args.proposal_cache_dir,
        device=args.proposal_device,
        local_files_only=args.proposal_offline,
        allow_model_fallback=not args.strict_proposal_models,
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix="physics-battle-", dir=output_path.parent
        ) as temporary:
            report = run_battle_test(
                args.input,
                annotations_path=args.annotations,
                recursive=args.recursive,
                max_images=args.max_images,
                workspace=temporary,
                min_applicability_retention=args.min_applicability_retention,
                max_hard_flips=args.max_hard_flips,
                max_mean_score_drift=args.max_mean_score_drift,
                max_score_drift=args.max_score_drift,
                engine_config=PhysicsEngineConfig(automatic=automatic),
            )
        json_path, csv_path, markdown_path = write_report(output_path, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"physics-battle-test: {exc}", file=sys.stderr)
        return 2

    passed = bool(report["summary"]["acceptance"]["passed"])
    print(
        f"Battle test {'passed' if passed else 'failed'}; reports: "
        f"{json_path}, {csv_path}, {markdown_path}"
    )
    return 1 if args.strict and not passed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
