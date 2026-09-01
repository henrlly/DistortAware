"""Threshold and binary metrics shared by training and evaluation."""
from __future__ import annotations

import numpy as np


def threshold_metrics(y, scores, threshold: float) -> dict[str, float | int]:
    y = np.asarray(y).reshape(-1).astype(int)
    scores = np.asarray(scores).reshape(-1)
    pred = scores >= threshold
    real = y == 0
    fake = y == 1
    tp = int(np.sum(pred & fake)); tn = int(np.sum(~pred & real))
    fp = int(np.sum(pred & real)); fn = int(np.sum(~pred & fake))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"n": int(len(y)), "accuracy": float((pred == (y == 1)).mean()),
            "precision": float(precision), "recall": float(recall),
            "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
            "real_accuracy": float(tn / max(tn + fp, 1)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
            "fake_accuracy": float(recall), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "threshold": float(threshold)}


def roc_auc(y, scores) -> float:
    y = np.asarray(y).reshape(-1).astype(int)
    scores = np.asarray(scores).reshape(-1)
    pos = y == 1; neg = y == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if not npos or not nneg:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_ranks = np.arange(1, len(scores) + 1)
    return float((sorted_ranks[y[order] == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def multiclass_accuracy(y, predictions) -> dict[str, float | int]:
    y = np.asarray(y).reshape(-1).astype(int)
    predictions = np.asarray(predictions).reshape(-1).astype(int)
    result = {"n": int(len(y)), "accuracy": float((y == predictions).mean())}
    for cls, name in ((0, "real"), (1, "synthetic"), (2, "tampered")):
        mask = y == cls
        result[f"{name}_n"] = int(mask.sum())
        result[f"{name}_accuracy"] = float((predictions[mask] == cls).mean()) if mask.any() else float("nan")
    return result


def balanced_threshold(y, scores) -> tuple[float, float]:
    y = np.asarray(y).reshape(-1).astype(int)
    scores = np.asarray(scores).reshape(-1)
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_value = -1.0
    best = []
    for threshold in candidates:
        metrics = threshold_metrics(y, scores, float(threshold))
        value = .5 * (metrics["recall"] + metrics["real_accuracy"])
        if value > best_value + 1e-12:
            best_value = value; best = [float(threshold)]
        elif abs(value - best_value) <= 1e-12:
            best.append(float(threshold))
    return float(np.mean(best)), float(best_value)


def target_fpr_threshold(y, scores, target: float = .05) -> float:
    y = np.asarray(y).reshape(-1).astype(int)
    scores = np.asarray(scores).reshape(-1)
    negatives = np.sort(scores[y == 0])
    if not len(negatives):
        return .5
    index = max(0, int(np.ceil((1.0 - target) * len(negatives))) - 1)
    return float(negatives[index])
