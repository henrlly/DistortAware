"""CSV and Markdown reports for normalized harness evaluation records."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


CSV_FIELDS = (
    "model", "transform", "image_id", "image_path", "source", "category", "label",
    "score", "score_kind", "confidence", "threshold", "decision", "missing", "errors",
)


def _csv_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {field: record.get(field) for field in CSV_FIELDS}
    row["errors"] = json.dumps(record.get("errors", []), sort_keys=True)
    return row


def _write_csv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(record) for record in records)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


def _metric_row(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {metrics.get('n', 0)} | {_pct(metrics.get('accuracy'))} | "
        f"{_pct(metrics.get('balanced_accuracy'))} | {_pct(metrics.get('precision'))} | "
        f"{_pct(metrics.get('recall'))} | {_pct(metrics.get('f1'))} | "
        f"{metrics.get('roc_auc', 'n/a') if metrics.get('roc_auc') is not None else 'n/a'} |"
    )


def _classifier_markdown(model: str, metrics: dict[str, Any]) -> str:
    lines = [f"# {model}", "", "| Transform | N | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, value in sorted(metrics.get("groups", {}).items()):
        group_model, transform = key.split(":", 1)
        if group_model == model:
            lines.append(_metric_row(transform, value))
    lines += ["", "## By source", "", "| Transform / source | N | Accuracy | Balanced accuracy | ROC-AUC |", "|---|---:|---:|---:|---:|"]
    for key, value in sorted(metrics.get("groups", {}).items()):
        group_model, transform = key.split(":", 1)
        if group_model != model:
            continue
        for source, source_metrics in sorted(value.get("by_source", {}).items()):
            lines.append(
                f"| {transform} / {source or 'unknown'} | {source_metrics.get('n', 0)} | "
                f"{_pct(source_metrics.get('accuracy'))} | "
                f"{_pct(source_metrics.get('balanced_accuracy'))} | "
                f"{source_metrics.get('roc_auc', 'n/a') if source_metrics.get('roc_auc') is not None else 'n/a'} |"
            )
    lines += ["", "## By label", "", "| Transform / label | N | Accuracy | Balanced accuracy | ROC-AUC |", "|---|---:|---:|---:|---:|"]
    for key, value in sorted(metrics.get("groups", {}).items()):
        group_model, transform = key.split(":", 1)
        if group_model != model:
            continue
        for label, label_metrics in sorted(value.get("by_label", {}).items()):
            lines.append(
                f"| {transform} / {label} | {label_metrics.get('n', 0)} | "
                f"{_pct(label_metrics.get('accuracy'))} | "
                f"{_pct(label_metrics.get('balanced_accuracy'))} | "
                f"{label_metrics.get('roc_auc', 'n/a') if label_metrics.get('roc_auc') is not None else 'n/a'} |"
            )
    return "\n".join(lines) + "\n"


def _physics_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# physics", "",
        "Physics reports physical-consistency evidence, not an AIGC classifier verdict.", "",
        "| Transform | N | Applicable | Applicability | Mean violation | Mean confidence | Status counts |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for transform, value in sorted(metrics.get("physics", {}).items()):
        lines.append(
            f"| {transform} | {value['n']} | {value['applicable']} | "
            f"{_pct(value['applicability_rate'])} | {_pct(value['mean_violation_score'])} | "
            f"{_pct(value['mean_confidence'])} | `{json.dumps(value['status_counts'], sort_keys=True)}` |"
        )
    lines += ["", "## By cue", "", "| Transform / cue | N | Applicable | Applicability | Mean violation | Mean confidence |", "|---|---:|---:|---:|---:|---:|"]
    for transform, value in sorted(metrics.get("physics", {}).items()):
        for cue, cue_metrics in sorted(value.get("by_cue", {}).items()):
            lines.append(
                f"| {transform} / {cue} | {cue_metrics.get('n', 0)} | {cue_metrics.get('applicable', 0)} | "
                f"{_pct(cue_metrics.get('applicability_rate'))} | "
                f"{_pct(cue_metrics.get('mean_violation_score'))} | "
                f"{_pct(cue_metrics.get('mean_confidence'))} |"
            )
    return "\n".join(lines) + "\n"


def _combined_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Harness evaluation report", "",
        f"- Data: `{report['data_dir']}`",
        f"- Manifest: `{report['manifest']}`",
        f"- Manifest fingerprint: `{report.get('manifest_fingerprint', 'unknown')}`",
        f"- Records: **{report['records']}**", "",
        "## Coverage", "", "| Model / transform | Expected | Returned | Missing | Duplicates | Errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in sorted(report["coverage"].items()):
        lines.append(
            f"| {key} | {value['expected']} | {value['returned']} | {value['missing']} | "
            f"{value['duplicates']} | {value['errors']} |"
        )
    lines += ["", "## Model reports", ""]
    for model in sorted(report["models"]):
        lines.append(f"- [{model}](models/{model}/report.md)")
    return "\n".join(lines) + "\n"


def write_reports(output_dir: str | Path, records: list[dict[str, Any]], report: dict[str, Any]) -> None:
    """Write combined and per-model CSV/Markdown reports."""
    output = Path(output_dir).expanduser().resolve()
    _write_csv(output / "records.csv", records)
    models_dir = output / "models"
    for model in report["models"]:
        model_records = [record for record in records if record.get("model") == model]
        model_dir = models_dir / model
        _write_csv(model_dir / "records.csv", model_records)
        if model == "physics":
            content = _physics_markdown(report["metrics"])
        else:
            content = _classifier_markdown(model, report["metrics"])
        (model_dir / "report.md").write_text(content, encoding="utf-8")
    (output / "report.md").write_text(_combined_markdown(report), encoding="utf-8")
