"""Join DINO PatchHead evidence with physics results without changing verdicts."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .integration import _canonical_path, _compact_physics, _write_json_atomic
from .spatial import (
    SpatialEvidenceError,
    grid_association,
    physics_outlier_mask,
    validate_score_grid,
)


DINO_ALIGNMENT_SCHEMA_VERSION = "0.1.0"


class DinoIntegrationError(ValueError):
    """Raised when DINO and physics payloads cannot be joined safely."""


def _physics_indices(
    physics_images: list[Any], physics_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_path: dict[str, dict[str, Any]] = {}
    by_basename: dict[str, list[dict[str, Any]]] = {}
    for raw_image in physics_images:
        if not isinstance(raw_image, dict) or not isinstance(raw_image.get("image_path"), str):
            raise DinoIntegrationError("Every physics image result must have a string `image_path`")
        canonical = _canonical_path(raw_image["image_path"], physics_root)
        if canonical in by_path:
            raise DinoIntegrationError(f"Duplicate physics image path: {raw_image['image_path']}")
        by_path[canonical] = raw_image
        by_basename.setdefault(Path(raw_image["image_path"]).name, []).append(raw_image)
    return by_path, by_basename


def _match_physics(
    image_path: str,
    *,
    detector_root: Path,
    physics_root: Path,
    by_path: dict[str, dict[str, Any]],
    by_basename: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str | None]:
    candidates = [("canonical_path", _canonical_path(image_path, detector_root))]
    if not Path(image_path).expanduser().is_absolute():
        candidates.append(("physics_input_root", _canonical_path(image_path, physics_root)))
    for method, path in candidates:
        if path in by_path:
            return by_path[path], method
    basename_matches = by_basename.get(Path(image_path).name, [])
    if len(basename_matches) == 1:
        return basename_matches[0], "unique_basename"
    return None, None


def compute_dino_physics_alignment(
    dino_image: dict[str, Any],
    physics_image: dict[str, Any],
) -> dict[str, Any]:
    """Measure DINO patch-score concentration around physics outlier geometry."""

    patch = dino_image.get("patch_evidence")
    limitations = [
        "Spatial association is not causal attribution and does not prove that DINO used a physics cue.",
        "PatchHead patches inherit image-level labels during training, so the map is weak evidence rather than supervised localization.",
        "The official PatchHead score averages a patch-head score with a CLS-head score; this map describes only the patch-head component.",
        "Only indeterminate/inconsistent cues and evidence explicitly marked as an outlier participate; globally consistent geometry is not treated as suspicious evidence.",
    ]
    base: dict[str, Any] = {
        "schema_version": DINO_ALIGNMENT_SCHEMA_VERSION,
        "alignment_kind": "spatial_association_not_causal_attribution",
        "limitations": limitations,
    }
    if not isinstance(patch, dict) or "values" not in patch:
        return {
            **base,
            "applicable": False,
            "reason": (
                "No per-patch evidence map is present. Export PatchHead's existing "
                "patch_logits tensor with physics-dino-export before requesting spatial alignment."
            ),
            "per_cue": {},
            "overall": None,
        }

    try:
        grid = validate_score_grid(patch["values"])
    except SpatialEvidenceError as exc:
        raise DinoIntegrationError(str(exc)) from exc
    declared_shape = patch.get("grid_shape")
    if declared_shape is not None and list(grid.shape) != list(declared_shape):
        raise DinoIntegrationError(
            f"Declared patch grid {declared_shape!r} does not match values shape {list(grid.shape)}"
        )
    coordinate_space = patch.get("coordinate_space")
    if coordinate_space != "normalized_full_frame":
        raise DinoIntegrationError(
            "Patch evidence coordinate_space must be `normalized_full_frame`; "
            "cropped/letterboxed mappings require an explicit inverse transform"
        )

    width = physics_image.get("width")
    height = physics_image.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise DinoIntegrationError("Physics image dimensions must be positive integers")
    cues = physics_image.get("cues")
    if not isinstance(cues, dict):
        raise DinoIntegrationError("Physics image must contain a `cues` object")

    per_cue: dict[str, Any] = {}
    union = np.zeros(grid.shape, dtype=bool)
    for cue_name, raw_cue in cues.items():
        if not isinstance(raw_cue, dict):
            continue
        try:
            mask, diagnostics = physics_outlier_mask(
                raw_cue,
                image_width=width,
                image_height=height,
                grid_shape=grid.shape,
            )
            if not raw_cue.get("applicable"):
                association = {
                    "applicable": False,
                    "reason": "The physics cue did not meet its applicability requirements.",
                    "selected_patch_count": 0,
                    "total_patch_count": int(grid.size),
                }
            elif raw_cue.get("status") == "consistent":
                association = {
                    "applicable": False,
                    "reason": (
                        "The physics cue is globally consistent, so isolated line-fit "
                        "residuals are not exposed as a suspicious DINO explanation."
                    ),
                    "selected_patch_count": 0,
                    "total_patch_count": int(grid.size),
                }
            else:
                association = grid_association(grid, mask)
        except SpatialEvidenceError as exc:
            raise DinoIntegrationError(f"Cue {cue_name}: {exc}") from exc
        association["physics_status"] = raw_cue.get("status")
        association["physics_violation_score"] = raw_cue.get("violation_score")
        association["geometry"] = diagnostics
        per_cue[str(cue_name)] = association
        if association.get("applicable"):
            union |= mask

    overall = grid_association(grid, union)
    applicable = bool(overall.get("applicable"))
    return {
        **base,
        "applicable": applicable,
        "patch_value_kind": patch.get("value_kind"),
        "grid_shape": list(grid.shape),
        "per_cue": per_cue,
        "overall": overall,
    }


def merge_dino_and_physics(
    dino_payload: Any,
    physics_payload: Any,
    *,
    path_root: str | Path = ".",
    allow_missing: bool = False,
) -> tuple[Any, dict[str, int]]:
    """Attach compact physics evidence and optional spatial alignment to DINO output.

    Supported DINO shapes are a JSON array of image records or an export object
    containing an ``images`` array.  Every record must contain ``image_path``.
    All existing detector fields are deep-copied unchanged.
    """

    if isinstance(dino_payload, list):
        records = dino_payload
        wrapper: dict[str, Any] | None = None
    elif isinstance(dino_payload, dict) and isinstance(dino_payload.get("images"), list):
        records = dino_payload["images"]
        wrapper = deepcopy(dino_payload)
    else:
        raise DinoIntegrationError(
            "DINO results must be a JSON array or an object containing an `images` array"
        )
    if not isinstance(physics_payload, dict) or not isinstance(physics_payload.get("images"), list):
        raise DinoIntegrationError("Physics results must contain an `images` array")

    detector_root = Path(path_root).expanduser().resolve(strict=False)
    raw_detector_root = dino_payload.get("input_root", ".") if isinstance(dino_payload, dict) else "."
    detector_input_root = Path(str(raw_detector_root)).expanduser()
    if not detector_input_root.is_absolute():
        detector_input_root = detector_root / detector_input_root
    detector_input_root = detector_input_root.resolve(strict=False)

    raw_physics_root = physics_payload.get("input_root", ".")
    physics_root = Path(str(raw_physics_root)).expanduser()
    if not physics_root.is_absolute():
        physics_root = detector_root / physics_root
    physics_root = physics_root.resolve(strict=False)
    by_path, by_basename = _physics_indices(physics_payload["images"], physics_root)

    output_records: list[dict[str, Any]] = []
    matched_ids: set[int] = set()
    missing = 0
    aligned = 0
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise DinoIntegrationError(f"DINO result at index {index} must be an object")
        if "physics_evidence" in raw_record or "dino_physics_alignment" in raw_record:
            raise DinoIntegrationError(f"DINO result at index {index} is already enriched")
        image_path = raw_record.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            raise DinoIntegrationError(f"DINO result at index {index} lacks `image_path`")
        physics_image, match_method = _match_physics(
            image_path,
            detector_root=detector_input_root,
            physics_root=physics_root,
            by_path=by_path,
            by_basename=by_basename,
        )
        record = deepcopy(raw_record)
        if physics_image is None or match_method is None:
            missing += 1
            if not allow_missing:
                raise DinoIntegrationError(
                    f"No unambiguous physics result matched DINO image {image_path!r}"
                )
            record["physics_evidence"] = None
            record["dino_physics_alignment"] = None
        else:
            matched_ids.add(id(physics_image))
            raw_schema_version = physics_payload.get("schema_version")
            raw_engine_version = physics_payload.get("engine_version")
            record["physics_evidence"] = _compact_physics(
                physics_image,
                schema_version=(
                    str(raw_schema_version) if raw_schema_version is not None else None
                ),
                engine_version=(
                    str(raw_engine_version) if raw_engine_version is not None else None
                ),
                match_method=match_method,
            )
            record["dino_physics_alignment"] = compute_dino_physics_alignment(
                record, physics_image
            )
            if record["dino_physics_alignment"].get("applicable"):
                aligned += 1
        output_records.append(record)

    if wrapper is None:
        output: Any = output_records
    else:
        wrapper["images"] = output_records
        wrapper["physics_integration"] = {
            "schema_version": DINO_ALIGNMENT_SCHEMA_VERSION,
            "primary_detector_fields_preserved": True,
            "physics_affects_detector_score": False,
        }
        output = wrapper
    return output, {
        "dino_records": len(records),
        "matched_records": len(records) - missing,
        "missing_records": missing,
        "records_with_spatial_alignment": aligned,
        "unmatched_physics_results": len(physics_payload["images"]) - len(matched_ids),
    }


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physics-dino-merge",
        description=(
            "Preserve DINO predictions, attach physics evidence, and optionally "
            "measure non-causal spatial association with PatchHead's patch map."
        ),
    )
    parser.add_argument("--dino-results", required=True)
    parser.add_argument("--physics-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--path-root", default=".")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dino = _load_json(args.dino_results)
        physics = _load_json(args.physics_results)
        enriched, summary = merge_dino_and_physics(
            dino,
            physics,
            path_root=args.path_root,
            allow_missing=args.allow_missing,
        )
        output = Path(args.output).expanduser().resolve(strict=False)
        _write_json_atomic(output, enriched, pretty=args.pretty)
    except (OSError, json.JSONDecodeError, DinoIntegrationError) as exc:
        print(f"physics-dino-merge: {exc}", file=sys.stderr)
        return 2
    print(
        f"Merged {summary['matched_records']}/{summary['dino_records']} DINO record(s); "
        f"{summary['records_with_spatial_alignment']} had usable spatial alignment. "
        f"Output: {output}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
