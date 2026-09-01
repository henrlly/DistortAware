"""Harness utility for measuring PatchHead transform stability."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageOps


CHECKPOINT_ROBUSTNESS_SCHEMA_VERSION = "0.1.0"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class CheckpointRobustnessError(ValueError):
    """Raised when a checkpoint robustness run cannot proceed safely."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _selected_images(root: Path, per_parent_limit: int) -> list[Path]:
    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise CheckpointRobustnessError(f"Unsupported image type: {root}")
        return [root]
    if not root.is_dir():
        raise CheckpointRobustnessError(f"Image input does not exist: {root}")
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            groups[path.parent.relative_to(root).as_posix()].append(path)
    selected: list[Path] = []
    for group in sorted(groups):
        selected.extend(groups[group][:per_parent_limit])
    if not selected:
        raise CheckpointRobustnessError("No supported images were found")
    return selected


def _stable_transform_seed(path: Path, transform_name: str) -> int:
    digest = hashlib.sha256(f"{path.as_posix()}\0{transform_name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _prepare(
    path: Path,
    *,
    transform_name: str,
    transform: Any,
    model_size: int,
) -> np.ndarray:
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as exc:
        raise CheckpointRobustnessError(f"Could not decode {path}: {exc}") from exc
    state = np.random.get_state()
    np.random.seed(_stable_transform_seed(path, transform_name))
    try:
        image = transform(image).convert("RGB")
    finally:
        np.random.set_state(state)
    image = image.resize((200, 200), Image.Resampling.BICUBIC)
    image = image.resize((model_size, model_size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    numerator = np.sum(left * right, axis=-1)
    result = np.ones_like(numerator, dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = numerator[valid] / denominator[valid]
    result[~valid & (np.linalg.norm(left - right, axis=-1) > 1e-12)] = 0.0
    return np.clip(result, -1.0, 1.0)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-12:
        return 1.0 if np.allclose(left, right, atol=1e-12, rtol=0.0) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _batch_outputs(runtime: Any, batch: np.ndarray) -> dict[str, np.ndarray]:
    outputs = runtime.infer_with_features(batch)
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
        raise CheckpointRobustnessError(
            "PatchHead feature runtime must return four output tensors"
        )
    image_logits, cls_logits, patch_logits, features = (
        np.asarray(output) for output in outputs
    )
    if (
        image_logits.shape != (len(batch),)
        or cls_logits.shape != (len(batch),)
        or patch_logits.ndim != 3
        or patch_logits.shape[0] != len(batch)
        or features.ndim != 4
        or features.shape[0] != len(batch)
        or not all(
            np.isfinite(output).all()
            for output in (image_logits, cls_logits, patch_logits, features)
        )
    ):
        raise CheckpointRobustnessError("PatchHead returned malformed or non-finite tensors")
    scores = 0.5 * (_sigmoid(image_logits) + _sigmoid(cls_logits))
    return {
        "scores": scores,
        "patch_scores": _sigmoid(patch_logits),
        "features": features.astype(np.float32, copy=False),
    }


def _compare_outputs(
    baseline: dict[str, np.ndarray],
    transformed: dict[str, np.ndarray],
    *,
    threshold: float,
) -> dict[str, Any]:
    score_drift = np.abs(transformed["scores"] - baseline["scores"])
    verdict_flips = (transformed["scores"] > threshold) != (
        baseline["scores"] > threshold
    )
    patch_mae: list[float] = []
    patch_pearson: list[float] = []
    dense_token_cosine: list[float] = []
    dense_global_cosine: list[float] = []
    for index in range(len(score_drift)):
        base_patch = baseline["patch_scores"][index]
        transformed_patch = transformed["patch_scores"][index]
        patch_mae.append(float(np.abs(base_patch - transformed_patch).mean()))
        patch_pearson.append(_pearson(base_patch, transformed_patch))
        base_features = baseline["features"][index].reshape(-1, baseline["features"].shape[-1])
        transformed_features = transformed["features"][index].reshape(
            -1, transformed["features"].shape[-1]
        )
        if base_features.shape != transformed_features.shape:
            raise CheckpointRobustnessError("Dense DINO grid shape changed under a transform")
        dense_token_cosine.append(
            float(_cosine_rows(base_features, transformed_features).mean())
        )
        dense_global_cosine.append(
            float(
                _cosine_rows(
                    base_features.mean(axis=0, keepdims=True),
                    transformed_features.mean(axis=0, keepdims=True),
                )[0]
            )
        )
    return {
        "image_count": len(score_drift),
        "score_absolute_drift": _stats(score_drift.tolist()),
        "verdict_flips": int(verdict_flips.sum()),
        "patch_score_mae": _stats(patch_mae),
        "patch_score_pearson": _stats(patch_pearson),
        "dense_token_cosine": _stats(dense_token_cosine),
        "dense_global_cosine": _stats(dense_global_cosine),
    }


def run_checkpoint_robustness(
    image_root: str | Path,
    *,
    checkpoint: str | Path,
    device: str | None = None,
    per_parent_limit: int = 2,
    batch_size: int = 8,
    transform_names: list[str] | None = None,
) -> dict[str, Any]:
    if per_parent_limit <= 0 or batch_size <= 0:
        raise CheckpointRobustnessError(
            "per_parent_limit and batch_size must be positive"
        )
    repository = _repository_root()
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from patchhead.inference import TorchPatchHeadRuntime
    from patchhead.transforms import TRANSFORMS

    selected_names = transform_names or list(TRANSFORMS)
    unknown = [name for name in selected_names if name not in TRANSFORMS]
    if unknown:
        raise CheckpointRobustnessError(f"Unknown transform(s): {', '.join(unknown)}")
    if "clean" not in selected_names:
        selected_names = ["clean", *selected_names]
    if not any(name != "clean" for name in selected_names):
        raise CheckpointRobustnessError(
            "At least one non-clean transform must be requested"
        )
    root = Path(image_root).expanduser().resolve()
    images = _selected_images(root, per_parent_limit)
    runtime = TorchPatchHeadRuntime(checkpoint, device)
    model_size = int(runtime.metadata["model_input_size"])
    threshold = float(runtime.metadata["threshold"])
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for transform_name in selected_names:
        batches: list[dict[str, np.ndarray]] = []
        transform = TRANSFORMS[transform_name]
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            batch = np.stack(
                [
                    _prepare(
                        path,
                        transform_name=transform_name,
                        transform=transform,
                        model_size=model_size,
                    )
                    for path in chunk
                ]
            )
            batches.append(_batch_outputs(runtime, batch))
        outputs[transform_name] = {
            key: np.concatenate([batch[key] for batch in batches], axis=0)
            for key in ("scores", "patch_scores", "features")
        }
    baseline = outputs["clean"]
    per_transform = {
        name: _compare_outputs(baseline, outputs[name], threshold=threshold)
        for name in selected_names
        if name != "clean"
    }
    all_transforms = list(per_transform.values())
    return {
        "schema_version": CHECKPOINT_ROBUSTNESS_SCHEMA_VERSION,
        "input_root": str(root),
        "selected_images": [
            path.relative_to(root).as_posix() if root.is_dir() else path.name
            for path in images
        ],
        "detector": dict(runtime.metadata),
        "protocol": {
            "per_parent_limit": per_parent_limit,
            "batch_size": batch_size,
            "transforms": selected_names,
            "noise_seed": "sha256(image_path + NUL + transform_name), first 32 bits",
            "comparison_reference": "clean output from the same loaded runtime",
        },
        "per_transform": per_transform,
        "aggregate": {
            "transform_count": len(per_transform),
            "total_verdict_flips": sum(item["verdict_flips"] for item in all_transforms),
            "maximum_score_drift": max(
                item["score_absolute_drift"]["maximum"] for item in all_transforms
            ),
            "mean_of_transform_patch_pearson": statistics.fmean(
                item["patch_score_pearson"]["mean"] for item in all_transforms
            ),
            "minimum_mean_transform_patch_pearson": min(
                item["patch_score_pearson"]["mean"] for item in all_transforms
            ),
            "mean_of_transform_dense_token_cosine": statistics.fmean(
                item["dense_token_cosine"]["mean"] for item in all_transforms
            ),
            "minimum_mean_transform_dense_token_cosine": min(
                item["dense_token_cosine"]["mean"] for item in all_transforms
            ),
        },
        "limitations": [
            "This is a bounded engineering sample, not a calibrated robustness benchmark.",
            "Patch-map correlation measures stability of weak image-supervised evidence, not localization correctness.",
            "Dense-token cosine compares matching grid coordinates; crop80 changes depicted content at those coordinates.",
            "Stable features or scores do not establish that an image is authentic.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="patchhead-robustness",
        description=(
            "Measure pooled PatchHead score, patch-map, and same-pass DINO-grid "
            "stability under the official transforms."
        ),
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device")
    parser.add_argument("--per-parent-limit", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--transforms", help="Comma-separated transform names; default is all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names = [name.strip() for name in args.transforms.split(",")] if args.transforms else None
    try:
        result = run_checkpoint_robustness(
            args.image_dir,
            checkpoint=args.checkpoint,
            device=args.device,
            per_parent_limit=args.per_parent_limit,
            batch_size=args.batch_size,
            transform_names=names,
        )
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                result,
                indent=2 if args.pretty else None,
                separators=None if args.pretty else (",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except (OSError, CheckpointRobustnessError, ValueError) as exc:
        print(f"patchhead-robustness: {exc}", file=sys.stderr)
        return 2
    print(
        f"Evaluated {len(result['selected_images'])} image(s) across "
        f"{result['aggregate']['transform_count']} transform(s); output: {output}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
