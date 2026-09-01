"""Strict, non-destructive integration with the official detector output.

The official repository currently emits a JSON array of records containing
``image_path``, ``pred``, and ``is_aigc``. This module adds a compact
``physics_evidence`` sidecar to each record. It never changes or recomputes the
primary detector's fields.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from typing import Any


class IntegrationError(ValueError):
    """Raised when detector and physics outputs cannot be joined safely."""


def _canonical_path(value: str, root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return os.path.normcase(str(path.resolve(strict=False)))


def _load_json(path: str | Path) -> Any:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _compact_cue(cue: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "applicable",
        "status",
        "violation_score",
        "confidence",
        "summary",
        "assumptions",
        "measurements",
        "limitations",
        "overlay_path",
    )
    return {field: deepcopy(cue.get(field)) for field in fields if field in cue}


def _compact_physics(
    image: dict[str, Any],
    *,
    schema_version: str | None,
    engine_version: str | None,
    match_method: str,
) -> dict[str, Any]:
    cues = image.get("cues")
    aggregate = image.get("physics")
    if not isinstance(cues, dict) or not isinstance(aggregate, dict):
        raise IntegrationError("Every physics image result must contain `physics` and `cues` objects")
    return {
        "schema_version": schema_version,
        "engine_version": engine_version,
        "match_method": match_method,
        "aggregate": deepcopy(aggregate),
        "cues": {
            str(name): _compact_cue(cue)
            for name, cue in cues.items()
            if isinstance(cue, dict)
        },
        "errors": deepcopy(image.get("errors", [])),
        "details_image_path": image.get("image_path"),
    }


def merge_detector_and_physics(
    detector_records: Any,
    physics_payload: Any,
    *,
    path_root: str | Path = ".",
    allow_missing: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Attach compact physics evidence to the official detector's records.

    Matching prefers canonical paths. A basename fallback is used only when it
    is unique among all physics results, preventing silent cross-image joins.
    """

    if not isinstance(detector_records, list):
        raise IntegrationError("Detector results must be a JSON array")
    if not isinstance(physics_payload, dict):
        raise IntegrationError("Physics results must be a JSON object")
    physics_images = physics_payload.get("images")
    if not isinstance(physics_images, list):
        raise IntegrationError("Physics results must contain an `images` array")

    detector_root = Path(path_root).expanduser().resolve(strict=False)
    raw_physics_root = physics_payload.get("input_root", ".")
    physics_root = Path(str(raw_physics_root)).expanduser()
    if not physics_root.is_absolute():
        physics_root = detector_root / physics_root
    physics_root = physics_root.resolve(strict=False)

    by_path: dict[str, dict[str, Any]] = {}
    by_basename: dict[str, list[dict[str, Any]]] = {}
    for image in physics_images:
        if not isinstance(image, dict) or not isinstance(image.get("image_path"), str):
            raise IntegrationError("Every physics image result must have a string `image_path`")
        canonical = _canonical_path(image["image_path"], physics_root)
        if canonical in by_path:
            raise IntegrationError(f"Duplicate physics image path: {image['image_path']}")
        by_path[canonical] = image
        basename = Path(image["image_path"]).name
        by_basename.setdefault(basename, []).append(image)

    schema_version = physics_payload.get("schema_version")
    engine_version = physics_payload.get("engine_version")
    enriched: list[dict[str, Any]] = []
    matched_ids: set[int] = set()
    missing_count = 0

    for index, raw_record in enumerate(detector_records):
        if not isinstance(raw_record, dict):
            raise IntegrationError(f"Detector result at index {index} must be an object")
        if "physics_evidence" in raw_record:
            raise IntegrationError(
                f"Detector result at index {index} already contains `physics_evidence`"
            )
        image_path = raw_record.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            raise IntegrationError(f"Detector result at index {index} lacks `image_path`")

        candidates = [
            ("canonical_path", _canonical_path(image_path, detector_root))
        ]
        if not Path(image_path).expanduser().is_absolute():
            candidates.append(
                ("physics_input_root", _canonical_path(image_path, physics_root))
            )
        image = None
        match_method = ""
        for candidate_method, candidate_path in candidates:
            if candidate_path in by_path:
                image = by_path[candidate_path]
                match_method = candidate_method
                break
        if image is None:
            basename_matches = by_basename.get(Path(image_path).name, [])
            if len(basename_matches) == 1:
                image = basename_matches[0]
                match_method = "unique_basename"

        record = deepcopy(raw_record)
        if image is None:
            missing_count += 1
            if not allow_missing:
                raise IntegrationError(
                    f"No unambiguous physics result matched detector image {image_path!r}"
                )
            record["physics_evidence"] = None
        else:
            matched_ids.add(id(image))
            record["physics_evidence"] = _compact_physics(
                image,
                schema_version=str(schema_version) if schema_version is not None else None,
                engine_version=str(engine_version) if engine_version is not None else None,
                match_method=match_method,
            )
        enriched.append(record)

    return enriched, {
        "detector_records": len(detector_records),
        "matched_records": len(detector_records) - missing_count,
        "missing_records": missing_count,
        "unmatched_physics_results": len(physics_images) - len(matched_ids),
    }


def _write_json_atomic(path: Path, payload: Any, *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            ensure_ascii=False,
        )
        handle.write("\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physics-merge",
        description=(
            "Attach compact physics evidence to DID detector records without changing "
            "their prediction fields."
        ),
    )
    parser.add_argument("--detector-results", required=True, help="DID preds JSON array")
    parser.add_argument("--physics-results", required=True, help="Physics batch JSON")
    parser.add_argument("--output", required=True, help="Enriched JSON array")
    parser.add_argument(
        "--path-root",
        default=".",
        help="Base directory for relative detector image paths (default: current directory)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write null physics evidence for unmatched images instead of failing",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        detector = _load_json(args.detector_results)
        physics = _load_json(args.physics_results)
        enriched, summary = merge_detector_and_physics(
            detector,
            physics,
            path_root=args.path_root,
            allow_missing=args.allow_missing,
        )
        output = Path(args.output).expanduser().resolve(strict=False)
        _write_json_atomic(output, enriched, pretty=args.pretty)
    except (OSError, json.JSONDecodeError, IntegrationError) as exc:
        print(f"physics-merge: {exc}", file=sys.stderr)
        return 2

    print(
        f"Merged {summary['matched_records']}/{summary['detector_records']} detector "
        f"record(s); {summary['unmatched_physics_results']} physics result(s) unused. "
        f"Output: {output}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
