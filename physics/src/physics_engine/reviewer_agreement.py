"""Agreement reporting for two independent physics-evidence reviewers.

The physics engine deliberately requires reviewed shadow and reflection point
correspondences.  This module checks whether two reviewers agree on cue
applicability, the resulting geometric status, and the locations of matched
point pairs.  It does not turn reviewer agreement into an AIGC score.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

from PIL import Image, ImageOps

from .annotations import AnnotationStore, ImageAnnotations
from .engine import SUPPORTED_EXTENSIONS
from .reflection import analyze_reflections
from .shadow import analyze_cast_shadows


CUES = ("cast_shadow", "reflection")
DECISIONS = ("applicable", "not_applicable", "uncertain", "unreviewed")
STATUS_LABELS = ("consistent", "inconsistent", "indeterminate", "not_applicable")


def _load_payload(path: str | Path) -> dict[str, Any]:
    annotation_path = Path(path).expanduser().resolve()
    files = (
        [annotation_path]
        if annotation_path.is_file()
        else sorted(annotation_path.glob("*.json"))
        if annotation_path.is_dir()
        else []
    )
    if not files:
        raise FileNotFoundError(f"No annotation JSON was found at {annotation_path}")

    merged_images: dict[str, dict[str, Any]] = {}
    reviewer_ids: set[str] = set()
    owners: dict[str, Path] = {}
    for source in files:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Annotation root must be an object: {source}")
        default_space = payload.get("coordinate_space", "normalized")
        if default_space not in {"normalized", "pixels"}:
            raise ValueError(f"Unknown coordinate_space in {source}")
        reviewer = payload.get("reviewer", {})
        reviewer_id = reviewer.get("id") if isinstance(reviewer, dict) else None
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise ValueError(
                f"Annotation export must contain a non-empty reviewer.id: {source}"
            )
        reviewer_ids.add(reviewer_id.strip())
        images = payload.get("images", {})
        if not isinstance(images, dict):
            raise ValueError(f"Annotation `images` must be an object: {source}")
        for image_key, raw_entry in images.items():
            if image_key in merged_images:
                raise ValueError(
                    f"Duplicate image key {image_key!r} in {owners[image_key]} and {source}"
                )
            if not isinstance(raw_entry, dict):
                raise ValueError(f"Annotation for {image_key!r} must be an object")
            entry = deepcopy(raw_entry)
            entry.setdefault("coordinate_space", default_space)
            merged_images[str(image_key)] = entry
            owners[str(image_key)] = source

    if len(reviewer_ids) > 1:
        raise ValueError(
            f"One reviewer input contains multiple reviewer IDs: {sorted(reviewer_ids)}"
        )
    reviewer_id = next(iter(reviewer_ids))
    return {
        "coordinate_space": "normalized",
        "reviewer": {
            "id": reviewer_id,
            "identity_source": "metadata",
        },
        "images": merged_images,
        "source": str(annotation_path),
    }


def _cue_decision(entry: dict[str, Any] | None, cue: str) -> str:
    if entry is None:
        return "unreviewed"
    section = entry.get(cue, {})
    if section is None:
        return "unreviewed"
    if not isinstance(section, dict):
        raise ValueError(f"{cue} must be an object")
    decision = section.get("applicability")
    if decision is None:
        pairs = section.get("pairs", [])
        return "applicable" if isinstance(pairs, list) and pairs else "unreviewed"
    if decision not in DECISIONS:
        raise ValueError(
            f"{cue}.applicability must be one of {', '.join(DECISIONS)}"
        )
    return str(decision)


def _cohen_kappa(
    values_a: Iterable[str], values_b: Iterable[str], labels: Iterable[str]
) -> dict[str, Any]:
    left = list(values_a)
    right = list(values_b)
    if len(left) != len(right):
        raise ValueError("Agreement vectors must have equal length")
    known = tuple(labels)
    pairs = [(a, b) for a, b in zip(left, right) if a in known and b in known]
    matrix = {a: {b: 0 for b in known} for a in known}
    for a, b in pairs:
        matrix[a][b] += 1
    total = len(pairs)
    if total == 0:
        return {
            "comparable_items": 0,
            "observed_agreement": None,
            "expected_agreement": None,
            "cohen_kappa": None,
            "confusion_matrix": matrix,
        }
    observed = sum(matrix[label][label] for label in known) / total
    expected = sum(
        sum(matrix[label].values())
        * sum(matrix[row][label] for row in known)
        for label in known
    ) / (total * total)
    denominator = 1.0 - expected
    kappa = None if abs(denominator) < 1e-12 else (observed - expected) / denominator
    return {
        "comparable_items": total,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "cohen_kappa": kappa,
        "confusion_matrix": matrix,
    }


def _image_index(root: Path) -> tuple[list[Path], dict[str, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {root}")
    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    index: dict[str, Path] = {}
    ambiguous: set[str] = set()
    for path in images:
        relative = path.relative_to(root).as_posix()
        index[relative] = path
        index[path.resolve().as_posix()] = path
        if path.name in index and index[path.name] != path:
            ambiguous.add(path.name)
        else:
            index[path.name] = path
    for basename in ambiguous:
        index.pop(basename, None)
    return images, index


def _result_status(
    cue: str,
    decision: str,
    annotations: ImageAnnotations,
    width: int,
    height: int,
) -> str:
    if decision == "not_applicable":
        return "not_applicable"
    if decision == "uncertain":
        return "indeterminate"
    if decision == "unreviewed":
        raise ValueError("Unreviewed cues do not have a geometric status")
    result = (
        analyze_cast_shadows(annotations.shadow_pairs, width, height)
        if cue == "cast_shadow"
        else analyze_reflections(annotations.reflection_pairs, width, height)
    )
    return result.status


def _normalized_pairs(
    cue: str, annotations: ImageAnnotations, width: int, height: int
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if cue == "cast_shadow":
        raw_pairs = [
            (pair.object_contact, pair.shadow_tip) for pair in annotations.shadow_pairs
        ]
    else:
        raw_pairs = [
            (pair.object_point, pair.reflection_point)
            for pair in annotations.reflection_pairs
        ]
    return [
        ((first[0] / width, first[1] / height), (second[0] / width, second[1] / height))
        for first, second in raw_pairs
    ]


def _pair_distance(
    left: tuple[tuple[float, float], tuple[float, float]],
    right: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    endpoint_distances = [
        math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(left, right)
    ]
    return sum(endpoint_distances) / len(endpoint_distances)


def _match_pairs(
    left: list[tuple[tuple[float, float], tuple[float, float]]],
    right: list[tuple[tuple[float, float], tuple[float, float]]],
    tolerance: float,
) -> list[float]:
    candidates = sorted(
        (_pair_distance(pair_a, pair_b), index_a, index_b)
        for index_a, pair_a in enumerate(left)
        for index_b, pair_b in enumerate(right)
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[float] = []
    for distance, index_a, index_b in candidates:
        if distance > tolerance:
            break
        if index_a in used_a or index_b in used_b:
            continue
        used_a.add(index_a)
        used_b.add(index_b)
        matches.append(distance)
    return matches


def evaluate_reviewer_agreement(
    image_root: str | Path,
    reviewer_a_path: str | Path,
    reviewer_b_path: str | Path,
    *,
    pair_tolerance: float = 0.05,
) -> dict[str, Any]:
    if not 0.0 < pair_tolerance <= 1.0:
        raise ValueError("pair_tolerance must lie within (0, 1]")
    root = Path(image_root).expanduser().resolve()
    _images, index = _image_index(root)
    payload_a = _load_payload(reviewer_a_path)
    payload_b = _load_payload(reviewer_b_path)
    reviewer_a = payload_a["reviewer"]
    reviewer_b = payload_b["reviewer"]
    if reviewer_a["id"] == reviewer_b["id"]:
        raise ValueError("Reviewer inputs must have distinct reviewer IDs")

    store_a = AnnotationStore(payload_a)
    store_b = AnnotationStore(payload_b)
    keys = sorted(set(payload_a["images"]) | set(payload_b["images"]))
    if not keys:
        raise ValueError("Reviewer inputs contain no image annotations")
    unresolved: list[str] = []
    rows: list[dict[str, Any]] = []
    for image_key in keys:
        entry_a = payload_a["images"].get(image_key)
        entry_b = payload_b["images"].get(image_key)
        image_path = index.get(image_key)
        if image_path is None:
            unresolved.append(image_key)
            continue
        with Image.open(image_path) as opened:
            width, height = ImageOps.exif_transpose(opened).size
        annotations_a = store_a.for_image(image_path, root, width, height)
        annotations_b = store_b.for_image(image_path, root, width, height)
        for cue in CUES:
            decision_a = _cue_decision(entry_a, cue)
            decision_b = _cue_decision(entry_b, cue)
            status_a = (
                _result_status(cue, decision_a, annotations_a, width, height)
                if decision_a != "unreviewed"
                else None
            )
            status_b = (
                _result_status(cue, decision_b, annotations_b, width, height)
                if decision_b != "unreviewed"
                else None
            )
            pairs_a = _normalized_pairs(cue, annotations_a, width, height)
            pairs_b = _normalized_pairs(cue, annotations_b, width, height)
            distances = (
                _match_pairs(pairs_a, pairs_b, pair_tolerance)
                if decision_a == decision_b == "applicable"
                else []
            )
            denominator = len(pairs_a) + len(pairs_b)
            rows.append(
                {
                    "image_key": image_key,
                    "cue": cue,
                    "reviewer_a_decision": decision_a,
                    "reviewer_b_decision": decision_b,
                    "reviewer_a_status": status_a,
                    "reviewer_b_status": status_b,
                    "reviewer_a_pairs": len(pairs_a),
                    "reviewer_b_pairs": len(pairs_b),
                    "matched_pairs": len(distances),
                    "pair_count_dice": (
                        2.0 * len(distances) / denominator if denominator else None
                    ),
                    "mean_matched_pair_distance": (
                        statistics.fmean(distances) if distances else None
                    ),
                }
            )

    per_cue: dict[str, Any] = {}
    for cue in CUES:
        cue_rows = [row for row in rows if row["cue"] == cue]
        both_reviewed = [
            row
            for row in cue_rows
            if row["reviewer_a_decision"] != "unreviewed"
            and row["reviewer_b_decision"] != "unreviewed"
        ]
        status_rows = [
            row
            for row in both_reviewed
            if row["reviewer_a_status"] in STATUS_LABELS
            and row["reviewer_b_status"] in STATUS_LABELS
        ]
        pair_comparable_rows = [
            row
            for row in cue_rows
            if row["reviewer_a_decision"] == row["reviewer_b_decision"] == "applicable"
            and row["pair_count_dice"] is not None
        ]
        pair_rows = [row for row in pair_comparable_rows if row["matched_pairs"] > 0]
        all_pair_distances = [
            row["mean_matched_pair_distance"]
            for row in pair_rows
            if row["mean_matched_pair_distance"] is not None
        ]
        per_cue[cue] = {
            "union_images": len(cue_rows),
            "reviewer_a_reviewed": sum(
                row["reviewer_a_decision"] != "unreviewed" for row in cue_rows
            ),
            "reviewer_b_reviewed": sum(
                row["reviewer_b_decision"] != "unreviewed" for row in cue_rows
            ),
            "both_reviewed": len(both_reviewed),
            "applicability_decision_agreement": _cohen_kappa(
                [row["reviewer_a_decision"] for row in both_reviewed],
                [row["reviewer_b_decision"] for row in both_reviewed],
                DECISIONS[:-1],
            ),
            "geometric_status_agreement": _cohen_kappa(
                [row["reviewer_a_status"] for row in status_rows],
                [row["reviewer_b_status"] for row in status_rows],
                STATUS_LABELS,
            ),
            "pair_concordance": {
                "tolerance_normalized_coordinate_distance": pair_tolerance,
                "comparable_images_with_pairs": len(pair_comparable_rows),
                "images_with_matched_pairs": len(pair_rows),
                "matched_pairs": sum(row["matched_pairs"] for row in pair_rows),
                "mean_image_pair_count_dice": (
                    statistics.fmean(
                        row["pair_count_dice"]
                        for row in pair_comparable_rows
                        if row["pair_count_dice"] is not None
                    )
                    if pair_comparable_rows
                    else None
                ),
                "mean_matched_pair_distance": (
                    statistics.fmean(all_pair_distances)
                    if all_pair_distances
                    else None
                ),
            },
        }

    return {
        "report_version": "0.1.0",
        "purpose": "independent_reviewer_agreement_for_explanation_evidence",
        "reviewers": {"a": reviewer_a, "b": reviewer_b},
        "image_root": str(root),
        "union_image_count": len(keys),
        "evaluated_image_count": len(keys) - len(unresolved),
        "unresolved_image_keys": unresolved,
        "per_cue": per_cue,
        "rows": rows,
        "interpretation": {
            "not_classifier_accuracy": True,
            "coordinate_agreement_is_not_scene_truth": True,
            "kappa_excludes_unreviewed_items": True,
        },
    }


def write_report(output_path: str | Path, report: dict[str, Any]) -> tuple[Path, Path]:
    json_path = Path(output_path).expanduser().resolve(strict=False)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = json_path.with_suffix(".md")
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(json_path)

    lines = [
        "# Physics reviewer-agreement report",
        "",
        f"- Reviewer A: **{report['reviewers']['a']['id']}**",
        f"- Reviewer B: **{report['reviewers']['b']['id']}**",
        f"- Evaluated images: **{report['evaluated_image_count']}**",
        f"- Unresolved image keys: **{len(report['unresolved_image_keys'])}**",
        "",
        "| Cue | Both reviewed | Applicability agreement | Applicability kappa | Status agreement | Status kappa | Matched pairs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cue, summary in report["per_cue"].items():
        applicability = summary["applicability_decision_agreement"]
        status = summary["geometric_status_agreement"]
        lines.append(
            f"| {cue} | {summary['both_reviewed']} | "
            f"{_percent(applicability['observed_agreement'])} | "
            f"{_number(applicability['cohen_kappa'])} | "
            f"{_percent(status['observed_agreement'])} | "
            f"{_number(status['cohen_kappa'])} | "
            f"{summary['pair_concordance']['matched_pairs']} |"
        )
    lines.extend(
        [
            "",
            "Agreement is a quality-control measure for reviewed explanation evidence. It is not detector accuracy and does not establish physical ground truth.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physics-review-agreement",
        description="Compare two independent shadow/reflection annotation sets.",
    )
    parser.add_argument("--images", required=True, help="Root containing reviewed images")
    parser.add_argument("--reviewer-a", required=True, help="Reviewer A JSON or directory")
    parser.add_argument("--reviewer-b", required=True, help="Reviewer B JSON or directory")
    parser.add_argument(
        "--output", default="outputs/reviewer_agreement.json", help="JSON report path"
    )
    parser.add_argument(
        "--pair-tolerance",
        type=float,
        default=0.05,
        help="Maximum mean endpoint distance in normalized image coordinates",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_reviewer_agreement(
            args.images,
            args.reviewer_a,
            args.reviewer_b,
            pair_tolerance=args.pair_tolerance,
        )
        json_path, markdown_path = write_report(args.output, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"physics-review-agreement: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {json_path} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
