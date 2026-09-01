"""Metrics for normalized harness records."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    if not labels:
        return {"n": 0}
    truth = [int(label > 0) for label in labels]
    pred = [int(score >= threshold) for score in scores]
    tp = sum(a == b == 1 for a, b in zip(truth, pred))
    tn = sum(a == b == 0 for a, b in zip(truth, pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(truth, pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(truth, pred))
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return {"n": len(labels), "accuracy": (tp + tn) / len(labels),
            "balanced_accuracy": (tpr + tnr) / 2, "precision": tp / max(tp + fp, 1),
            "recall": tpr, "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "roc_auc": _roc_auc(truth, scores),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "threshold": threshold}


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        rank = (index + 1 + end) / 2
        rank_sum += rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_model_transform: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("label") is not None and record.get("model") != "physics":
            by_model_transform[(record["model"], record["transform"])].append(record)
    output: dict[str, Any] = {"groups": {}, "physics": {}}
    for (model, transform), group in sorted(by_model_transform.items()):
        scores = [r["score"] for r in group if isinstance(r.get("score"), (int, float))]
        labels = [r["label"] for r in group if isinstance(r.get("score"), (int, float))]
        threshold = float(group[0].get("threshold", .5)) if group else .5
        output["groups"][f"{model}:{transform}"] = _binary_metrics(labels, scores, threshold)
        output["groups"][f"{model}:{transform}"]["by_source"] = _by_source(group, threshold)
        output["groups"][f"{model}:{transform}"]["by_label"] = _by_label(group, threshold)
    physics = [r for r in records if r["model"] == "physics"]
    for transform in sorted({r["transform"] for r in physics}):
        group = [r for r in physics if r["transform"] == transform]
        applicable = [r for r in group if r.get("physics", {}).get("violation_score") is not None]
        scores = [r["physics"]["violation_score"] for r in applicable
                  if r["physics"].get("violation_score") is not None]
        output["physics"][transform] = {
            "n": len(group),
            "status_counts": dict(Counter(r.get("physics", {}).get("status") for r in group)),
            "applicable": len(applicable),
            "applicability_rate": len(applicable) / max(len(group), 1),
            "mean_violation_score": sum(scores) / len(scores) if scores else None,
            "mean_confidence": sum(float(r["physics"]["confidence"]) for r in applicable) / len(applicable) if applicable else None,
            "by_label": _physics_by_label(group),
            "by_cue": _physics_by_cue(group),
        }
    return output


def _by_source(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    result = {}
    for source in sorted({r.get("source", "") for r in records}):
        group = [r for r in records if r.get("source", "") == source and isinstance(r.get("score"), (int, float))]
        result[source] = _binary_metrics([r["label"] for r in group], [r["score"] for r in group], threshold)
    return result


def _by_label(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    result = {}
    for label in sorted({r.get("label") for r in records}):
        group = [r for r in records if r.get("label") == label and isinstance(r.get("score"), (int, float))]
        result[str(label)] = _binary_metrics([r["label"] for r in group], [r["score"] for r in group], threshold)
    return result


def _physics_by_label(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for label in sorted({r.get("label") for r in records}):
        group = [r for r in records if r.get("label") == label]
        applicable = [r for r in group if r.get("physics", {}).get("violation_score") is not None]
        scores = [float(r["physics"]["violation_score"]) for r in applicable]
        result[str(label)] = {"n": len(group), "applicable": len(applicable),
                              "mean_violation_score": sum(scores) / len(scores) if scores else None,
                              "mean_confidence": sum(float(r["physics"].get("confidence", 0)) for r in applicable) / len(applicable) if applicable else None}
    return result


def _physics_by_cue(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    cue_names = sorted({cue for record in records for cue in record.get("cues", {})})
    for cue_name in cue_names:
        cues = [record.get("cues", {}).get(cue_name, {}) for record in records]
        applicable = [cue for cue in cues if cue.get("violation_score") is not None]
        scores = [float(cue["violation_score"]) for cue in applicable]
        result[cue_name] = {
            "n": len(cues),
            "applicable": sum(bool(cue.get("applicable")) for cue in cues),
            "applicability_rate": sum(bool(cue.get("applicable")) for cue in cues) / max(len(cues), 1),
            "status_counts": dict(Counter(cue.get("status") for cue in cues)),
            "mean_violation_score": sum(scores) / len(scores) if scores else None,
            "mean_confidence": sum(float(cue.get("confidence", 0)) for cue in applicable) / len(applicable) if applicable else None,
        }
    return result
