"""Evaluate checkpoint-backed PatchHead and physics integration safely.

The evaluator keeps the binary primary-detector question separate from SID's
tampered-image diagnostic, verifies that adding physics did not change primary
scores, and treats patch maps as weak localization rather than segmentation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
from typing import Any, Iterable

import cv2
import numpy as np


CHECKPOINT_EVAL_SCHEMA_VERSION = "0.1.0"
LABEL_NAMES = {0: "real", 1: "full_synthetic", 2: "tampered"}
PARENT_LABELS = {
    "real": 0,
    "fake": 1,
    "aigc": 1,
    "full_synthetic": 1,
    "tampered": 2,
}
PHYSICS_CUES = ("perspective", "cast_shadow", "reflection")


class CheckpointEvaluationError(ValueError):
    """Raised when evaluation inputs cannot be joined or trusted."""


def _load_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_key(value: str) -> str:
    normalized = str(value).replace("\\", "/")
    return PurePosixPath(normalized).as_posix().lstrip("./")


def _relative_manifest_key(item: dict[str, Any]) -> str:
    raw_path = item.get("image_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise CheckpointEvaluationError("Every manifest item needs an image_path")
    parts = PurePosixPath(str(raw_path).replace("\\", "/")).parts
    image_indices = [index for index, part in enumerate(parts) if part == "images"]
    if image_indices and image_indices[-1] + 1 < len(parts):
        return PurePosixPath(*parts[image_indices[-1] + 1 :]).as_posix()
    label_name = item.get("label_name")
    if not isinstance(label_name, str) or not label_name:
        raise CheckpointEvaluationError(
            "Manifest paths outside an images/ directory need label_name"
        )
    return f"{label_name}/{PurePosixPath(*parts).name}"


def _prediction_records(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise CheckpointEvaluationError(
            "Predictions must be an object containing an images array"
        )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(payload["images"]):
        if not isinstance(record, dict):
            raise CheckpointEvaluationError(f"Prediction {index} is not an object")
        image_path = record.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            raise CheckpointEvaluationError(f"Prediction {index} lacks image_path")
        key = _path_key(image_path)
        if key in seen:
            raise CheckpointEvaluationError(f"Duplicate prediction path: {image_path}")
        seen.add(key)
        records.append(record)
    return records


def _labelled_records(
    predictions: Any, manifest: Any | None
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    records = _prediction_records(predictions)
    if manifest is None:
        labelled = []
        for record in records:
            parent = PurePosixPath(_path_key(record["image_path"])).parent.name.lower()
            if parent not in PARENT_LABELS:
                raise CheckpointEvaluationError(
                    f"Cannot infer a label from parent directory {parent!r}"
                )
            label = PARENT_LABELS[parent]
            labelled.append(
                (
                    record,
                    {
                        "label": label,
                        "label_name": LABEL_NAMES[label],
                        "image_path": record["image_path"],
                        "mask_path": None,
                    },
                )
            )
        return labelled

    if not isinstance(manifest, dict) or not isinstance(manifest.get("images"), list):
        raise CheckpointEvaluationError("Manifest must contain an images array")
    by_key: dict[str, dict[str, Any]] = {}
    by_basename: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in manifest["images"]:
        if not isinstance(item, dict):
            raise CheckpointEvaluationError("Every manifest image must be an object")
        label = item.get("label")
        if label not in LABEL_NAMES:
            raise CheckpointEvaluationError(f"Unsupported manifest label: {label!r}")
        key = _relative_manifest_key(item)
        if key in by_key:
            raise CheckpointEvaluationError(f"Duplicate manifest path: {key}")
        by_key[key] = item
        by_basename[PurePosixPath(key).name].append(item)

    labelled = []
    matched_ids: set[int] = set()
    for record in records:
        key = _path_key(record["image_path"])
        item = by_key.get(key)
        if item is None:
            matches = by_basename.get(PurePosixPath(key).name, [])
            if len(matches) == 1:
                item = matches[0]
        if item is None:
            raise CheckpointEvaluationError(
                f"No unambiguous manifest item matched {record['image_path']!r}"
            )
        if id(item) in matched_ids:
            raise CheckpointEvaluationError(
                f"Manifest item matched more than once: {item.get('image_path')!r}"
            )
        matched_ids.add(id(item))
        labelled.append((record, item))
    return labelled


def _wilson(successes: int, total: int) -> dict[str, Any] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    if successes == 0:
        lower = 0.0
    if successes == total:
        upper = 1.0
    return {
        "method": "wilson_score",
        "confidence": 0.95,
        "lower": lower,
        "upper": upper,
    }


def _score_stats(values: Iterable[float]) -> dict[str, Any] | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "minimum": min(finite),
        "maximum": max(finite),
    }


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _primary_metrics(
    labelled: list[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    failures = 0
    for record, item in labelled:
        if record.get("aigc_score") is None or record.get("is_aigc") is None:
            failures += 1
            continue
        by_label[int(item["label"])].append(record)

    binary_pairs = [
        (record, item)
        for record, item in labelled
        if item["label"] in (0, 1)
        and record.get("aigc_score") is not None
        and record.get("is_aigc") is not None
    ]
    labels = np.asarray([int(item["label"] == 1) for _, item in binary_pairs])
    scores = np.asarray([float(record["aigc_score"]) for record, _ in binary_pairs])
    predictions = np.asarray([bool(record["is_aigc"]) for record, _ in binary_pairs])
    tp = int(np.logical_and(predictions, labels == 1).sum())
    tn = int(np.logical_and(~predictions, labels == 0).sum())
    fp = int(np.logical_and(predictions, labels == 0).sum())
    fn = int(np.logical_and(~predictions, labels == 1).sum())
    real_count = tn + fp
    synthetic_count = tp + fn
    real_accuracy = tn / real_count if real_count else None
    synthetic_accuracy = tp / synthetic_count if synthetic_count else None
    accuracy = (tp + tn) / len(binary_pairs) if binary_pairs else None
    balanced = (
        (real_accuracy + synthetic_accuracy) / 2.0
        if real_accuracy is not None and synthetic_accuracy is not None
        else None
    )

    per_label: dict[str, Any] = {}
    for label, name in LABEL_NAMES.items():
        records = by_label.get(label, [])
        alerts = sum(bool(record["is_aigc"]) for record in records)
        per_label[name] = {
            "count": len(records),
            "score": _score_stats(record["aigc_score"] for record in records),
            "aigc_alerts": alerts,
            "aigc_alert_rate": alerts / len(records) if records else None,
            "aigc_alert_rate_ci95": _wilson(alerts, len(records)),
        }
    return {
        "scored_records": sum(len(records) for records in by_label.values()),
        "failed_records": failures,
        "binary_scope": "real versus full_synthetic only; tampered is reported separately",
        "binary": {
            "count": len(binary_pairs),
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "roc_auc": _auc(labels, scores) if len(binary_pairs) else None,
            "confusion": {
                "true_real_pred_real": tn,
                "true_real_pred_aigc": fp,
                "true_synthetic_pred_real": fn,
                "true_synthetic_pred_aigc": tp,
            },
        },
        "per_label": per_label,
    }


def _baseline_parity(predictions: Any, baseline: Any | None) -> dict[str, Any]:
    if baseline is None:
        return {"evaluated": False, "reason": "No detector-only baseline supplied"}
    current = {_path_key(record["image_path"]): record for record in _prediction_records(predictions)}
    reference = {_path_key(record["image_path"]): record for record in _prediction_records(baseline)}
    if set(current) != set(reference):
        raise CheckpointEvaluationError(
            "Integrated and detector-only prediction paths do not match exactly"
        )
    deltas: list[float] = []
    verdict_mismatches = 0
    component_mismatches = 0
    for key in sorted(current):
        left = current[key]
        right = reference[key]
        if left.get("aigc_score") is None or right.get("aigc_score") is None:
            if left.get("aigc_score") != right.get("aigc_score"):
                deltas.append(float("inf"))
        else:
            deltas.append(abs(float(left["aigc_score"]) - float(right["aigc_score"])))
        verdict_mismatches += int(left.get("is_aigc") != right.get("is_aigc"))
        component_mismatches += int(
            left.get("component_scores") != right.get("component_scores")
        )
    current_detector = predictions.get("detector", {})
    baseline_detector = baseline.get("detector", {})
    checkpoint_match = current_detector.get("checkpoint_sha256") == baseline_detector.get(
        "checkpoint_sha256"
    )
    maximum_delta = max(deltas, default=0.0)
    return {
        "evaluated": True,
        "record_count": len(current),
        "checkpoint_sha256_match": checkpoint_match,
        "maximum_absolute_score_delta": maximum_delta,
        "nonzero_score_deltas": sum(delta != 0.0 for delta in deltas),
        "verdict_mismatches": verdict_mismatches,
        "component_score_mismatches": component_mismatches,
        "exact_primary_parity": bool(
            checkpoint_match
            and maximum_delta == 0.0
            and verdict_mismatches == 0
            and component_mismatches == 0
        ),
    }


def _physics_metrics(
    labelled: list[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    counters: dict[int, dict[str, Counter[str]]] = defaultdict(
        lambda: {cue: Counter() for cue in PHYSICS_CUES}
    )
    scores: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {cue: [] for cue in PHYSICS_CUES}
    )
    physics_errors = 0
    records_with_physics = 0
    shared_feature_attempts = 0
    spatial_alignment_applicable = 0
    for record, item in labelled:
        physics = record.get("physics_evidence")
        if not isinstance(physics, dict):
            continue
        records_with_physics += 1
        errors = physics.get("errors")
        if isinstance(errors, list):
            physics_errors += len(errors)
        cues = physics.get("cues", {})
        label = int(item["label"])
        for cue in PHYSICS_CUES:
            result = cues.get(cue) if isinstance(cues, dict) else None
            if not isinstance(result, dict):
                counters[label][cue]["missing"] += 1
                continue
            applicable = bool(result.get("applicable"))
            status = str(result.get("status", "missing"))
            counters[label][cue]["applicable" if applicable else "not_applicable"] += 1
            counters[label][cue][status] += 1
            score = result.get("violation_score")
            if score is not None:
                scores[label][cue].append(float(score))
            if cue == "reflection":
                feature = result.get("measurements", {}).get("feature_backend", {})
                if isinstance(feature, dict) and feature.get("shared_primary_forward"):
                    shared_feature_attempts += 1
        alignment = record.get("dino_physics_alignment")
        if isinstance(alignment, dict) and alignment.get("applicable"):
            spatial_alignment_applicable += 1

    per_label: dict[str, Any] = {}
    for label, name in LABEL_NAMES.items():
        cue_payload: dict[str, Any] = {}
        for cue in PHYSICS_CUES:
            count = counters[label][cue]
            applicable = count["applicable"]
            inconsistent = count["inconsistent"]
            cue_payload[cue] = {
                "applicable": applicable,
                "not_applicable": count["not_applicable"],
                "statuses": {
                    status: count[status]
                    for status in ("consistent", "indeterminate", "inconsistent", "not_applicable")
                },
                "mean_violation_score": (
                    statistics.fmean(scores[label][cue]) if scores[label][cue] else None
                ),
                "displayed_inconsistency_rate_when_applicable": (
                    inconsistent / applicable if applicable else None
                ),
                "displayed_inconsistency_ci95_when_applicable": _wilson(
                    inconsistent, applicable
                ),
            }
        per_label[name] = cue_payload
    return {
        "records_with_physics": records_with_physics,
        "physics_errors": physics_errors,
        "same_pass_reflection_feature_attempts": shared_feature_attempts,
        "records_with_spatial_alignment": spatial_alignment_applicable,
        "per_label": per_label,
    }


def _patch_auc(mask: np.ndarray, scores: np.ndarray) -> float | None:
    return _auc(mask.astype(np.int8).ravel(), scores.ravel())


def _tamper_patch_localization(
    labelled: list[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    metrics: dict[str, list[float]] = defaultdict(list)
    evaluable = 0
    skipped = Counter()
    for record, item in labelled:
        if item["label"] != 2:
            continue
        mask_path = item.get("mask_path")
        patch = record.get("patch_evidence")
        if not isinstance(mask_path, str) or not mask_path:
            skipped["missing_mask"] += 1
            continue
        if not isinstance(patch, dict) or "values" not in patch:
            skipped["missing_patch_evidence"] += 1
            continue
        path = Path(mask_path).expanduser()
        if not path.is_file():
            skipped["mask_file_unavailable"] += 1
            continue
        scores = np.asarray(patch["values"], dtype=np.float32)
        if scores.ndim != 2 or min(scores.shape, default=0) <= 0 or not np.isfinite(scores).all():
            skipped["invalid_patch_evidence"] += 1
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            skipped["mask_decode_failure"] += 1
            continue
        occupancy = cv2.resize(
            (mask > 0).astype(np.float32),
            (scores.shape[1], scores.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        target = occupancy > 0.01
        if not target.any() or target.all():
            skipped["degenerate_resized_mask"] += 1
            continue
        auc = _patch_auc(target, scores)
        if auc is None:
            skipped["undefined_auc"] += 1
            continue
        inside = float(scores[target].mean())
        outside = float(scores[~target].mean())
        target_count = int(target.sum())
        top = np.zeros(scores.size, dtype=bool)
        top[np.argsort(scores.ravel(), kind="mergesort")[-target_count:]] = True
        top = top.reshape(scores.shape)
        intersection = int(np.logical_and(top, target).sum())
        union = int(np.logical_or(top, target).sum())
        metrics["patch_roc_auc"].append(auc)
        metrics["inside_minus_outside_score"].append(inside - outside)
        metrics["top_area_iou"].append(intersection / union if union else 1.0)
        metrics["maximum_patch_inside_mask"].append(
            float(target.ravel()[int(np.argmax(scores))])
        )
        evaluable += 1
    return {
        "interpretation": (
            "Exploratory weak localization only. PatchHead patches were supervised by "
            "image labels and are not tamper segmentation masks."
        ),
        "mask_patch_occupancy_threshold": 0.01,
        "evaluable_tampered_images": evaluable,
        "skipped": dict(skipped),
        "metrics": {name: _score_stats(values) for name, values in metrics.items()},
    }


def evaluate_checkpoint_predictions(
    predictions: Any,
    *,
    manifest: Any | None = None,
    baseline: Any | None = None,
    dataset_name: str = "unidentified",
    backbone_revision: str | None = None,
) -> dict[str, Any]:
    labelled = _labelled_records(predictions, manifest)
    detector = predictions.get("detector") if isinstance(predictions, dict) else None
    if not isinstance(detector, dict):
        raise CheckpointEvaluationError("Predictions lack detector metadata")
    if detector.get("checkpoint_dataset") != "pooled":
        raise CheckpointEvaluationError(
            "Final primary validation requires checkpoint_dataset='pooled'"
        )
    physics_contract = predictions.get("physics_integration", {})
    physics_affects_score = (
        physics_contract.get("physics_affects_detector_score")
        if isinstance(physics_contract, dict)
        else None
    )
    return {
        "schema_version": CHECKPOINT_EVAL_SCHEMA_VERSION,
        "dataset": dataset_name,
        "sample_count": len(labelled),
        "detector": {
            "family": detector.get("family"),
            "architecture": detector.get("arch"),
            "backbone": detector.get("backbone"),
            "backbone_revision": backbone_revision,
            "checkpoint_dataset": detector.get("checkpoint_dataset"),
            "checkpoint_sha256": detector.get("checkpoint_sha256"),
            "threshold": detector.get("threshold"),
            "score_formula": detector.get("score_formula"),
        },
        "integration_contract": {
            "physics_affects_detector_score": physics_affects_score,
            "primary_fields_declared_preserved": physics_contract.get(
                "primary_detector_fields_preserved"
            )
            if isinstance(physics_contract, dict)
            else None,
            "detector_only_parity": _baseline_parity(predictions, baseline),
        },
        "primary_detector": _primary_metrics(labelled),
        "physics_sidecar": _physics_metrics(labelled),
        "tamper_patch_localization": _tamper_patch_localization(labelled),
        "limitations": [
            "The evaluated subset is bounded and is not a replacement for the full held-out benchmark.",
            "Tampered-image records, when present, are outside the pooled checkpoint's binary real/full-synthetic training objective and are reported separately.",
            "Physics consistency is explanatory evidence, not an AIGC probability or class label.",
            "Shadow/reflection ground-truth correspondences are absent; applicability is coverage, not proposal accuracy.",
            "Patch maps receive image-level supervision and are weak localization, not segmentation.",
        ],
    }


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def render_markdown(result: dict[str, Any]) -> str:
    detector = result["detector"]
    primary = result["primary_detector"]
    binary = primary["binary"]
    parity = result["integration_contract"]["detector_only_parity"]
    lines = [
        "# Checkpoint-backed detector and physics validation",
        "",
        f"Dataset/sample: **{result['dataset']}**, {result['sample_count']} image(s).",
        "",
        "## Reproducibility",
        "",
        f"- Checkpoint SHA-256: `{detector.get('checkpoint_sha256')}`",
        f"- Checkpoint dataset tag: `{detector.get('checkpoint_dataset')}`",
        f"- Backbone: `{detector.get('backbone')}`",
        f"- Backbone revision: `{detector.get('backbone_revision')}`",
        f"- Threshold: `{detector.get('threshold')}`",
        "",
        "## Primary detector",
        "",
        "Binary metrics include only real and full-synthetic images. Tampered images are a separate diagnostic.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Accuracy | {_percent(binary['accuracy'])} |",
        f"| Balanced accuracy | {_percent(binary['balanced_accuracy'])} |",
        f"| ROC AUC | {binary['roc_auc'] if binary['roc_auc'] is not None else 'n/a'} |",
    ]
    for label_name, values in primary["per_label"].items():
        lines.append(
            f"| {label_name} AIGC alert rate (n={values['count']}) | "
            f"{_percent(values['aigc_alert_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Integration safety",
            "",
            (
                f"Exact detector-only parity: **{parity.get('exact_primary_parity')}** "
                f"(maximum score delta {parity.get('maximum_absolute_score_delta')}, "
                f"verdict mismatches {parity.get('verdict_mismatches')})."
                if parity.get("evaluated")
                else f"Parity not evaluated: {parity.get('reason')}."
            ),
            "",
            "## Physics sidecar",
            "",
            f"Physics records: {result['physics_sidecar']['records_with_physics']}; "
            f"errors: {result['physics_sidecar']['physics_errors']}; same-pass reflection "
            f"feature attempts: {result['physics_sidecar']['same_pass_reflection_feature_attempts']}.",
            "",
            "| Label | Cue | Applicable | Consistent | Indeterminate | Inconsistent |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for label, cues in result["physics_sidecar"]["per_label"].items():
        for cue, values in cues.items():
            statuses = values["statuses"]
            lines.append(
                f"| {label} | {cue} | {values['applicable']} | "
                f"{statuses['consistent']} | {statuses['indeterminate']} | "
                f"{statuses['inconsistent']} |"
            )
    tamper = result["tamper_patch_localization"]
    lines.extend(
        [
            "",
            "## Tamper patch diagnostic",
            "",
            tamper["interpretation"],
            "",
            f"Evaluable tampered images: {tamper['evaluable_tampered_images']}.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["limitations"])
    return "\n".join(lines) + "\n"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physics-checkpoint-eval",
        description=(
            "Evaluate a pooled PatchHead payload, optional detector-only parity, "
            "physics safety, and weak SID tamper localization."
        ),
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--baseline")
    parser.add_argument("--dataset-name", default="unidentified")
    parser.add_argument("--backbone-revision")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        predictions_path = Path(args.predictions).expanduser().resolve()
        manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None
        baseline_path = Path(args.baseline).expanduser().resolve() if args.baseline else None
        result = evaluate_checkpoint_predictions(
            _load_json(predictions_path),
            manifest=_load_json(manifest_path) if manifest_path else None,
            baseline=_load_json(baseline_path) if baseline_path else None,
            dataset_name=args.dataset_name,
            backbone_revision=args.backbone_revision,
        )
        result["inputs"] = {
            "predictions_sha256": _sha256(predictions_path),
            "manifest_sha256": _sha256(manifest_path) if manifest_path else None,
            "baseline_sha256": _sha256(baseline_path) if baseline_path else None,
        }
        output = Path(args.output).expanduser().resolve()
        _write_text_atomic(
            output,
            json.dumps(
                result,
                indent=2 if args.pretty else None,
                separators=None if args.pretty else (",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
        )
        report = Path(args.report).expanduser().resolve() if args.report else None
        if report:
            _write_text_atomic(report, render_markdown(result))
    except (OSError, json.JSONDecodeError, CheckpointEvaluationError, ValueError) as exc:
        print(f"physics-checkpoint-eval: {exc}", file=sys.stderr)
        return 2
    print(
        f"Evaluated {result['sample_count']} image(s); output: {output}"
        + (f"; report: {report}" if report else "")
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
