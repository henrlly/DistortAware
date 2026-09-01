"""Reusable PatchHead inference with optional same-pass patch evidence export.

The orchestration and JSON contract depend only on NumPy and Pillow. PyTorch,
timm, DINOv3 weights, and a trained checkpoint are imported only when the real
runtime is constructed. Tests can therefore exercise every path/shape/score
contract with a small deterministic runtime while the pooled checkpoint is
unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


INFERENCE_SCHEMA_VERSION = "0.2.0"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class PatchHeadInferenceError(RuntimeError):
    """Raised when the primary detector cannot satisfy its inference contract."""


class PatchHeadRuntime(Protocol):
    """Small runtime boundary used by production Torch and deterministic tests."""

    metadata: dict[str, Any]

    def infer(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return image logits, CLS logits, and per-patch logits for one batch."""


DenseFeatureSink = Callable[[str, np.ndarray], None]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _require_pooled_checkpoint(payload: dict[str, Any]) -> str:
    dataset = payload.get("ds")
    if dataset != "pooled":
        raise PatchHeadInferenceError(
            "Primary inference requires the pooled PatchHead checkpoint "
            f"(checkpoint ds={dataset!r})"
        )
    return str(dataset)


def _preflight_checkpoint_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PatchHeadInferenceError("PatchHead checkpoint root must be a dictionary")
    dataset = _require_pooled_checkpoint(payload)
    model_state = payload.get("model")
    if not isinstance(model_state, dict) or not model_state:
        raise PatchHeadInferenceError("PatchHead checkpoint has no trainable model state")
    try:
        threshold = float(payload["threshold"])
        model_size = int(payload["size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PatchHeadInferenceError(
            "PatchHead checkpoint must contain numeric threshold and size metadata"
        ) from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0 or model_size <= 0:
        raise PatchHeadInferenceError("PatchHead checkpoint threshold/size is invalid")
    output_classes: int | None = None
    output_weight = model_state.get("head.patch_logit.weight")
    output_shape = getattr(output_weight, "shape", None)
    if output_shape is not None and len(output_shape) >= 1:
        try:
            output_classes = int(output_shape[0])
        except (TypeError, ValueError):
            output_classes = None
    return {
        "dataset": dataset,
        "threshold": threshold,
        "model_size": model_size,
        "arch": payload.get("arch", "patchhead-dinov3-vitl16"),
        "output_classes": output_classes,
    }


def _load_binary_compat_detector(
    checkpoint_payload: dict[str, Any],
    *,
    detector_class: Any,
    torch: Any,
    device: str,
) -> Any:
    """Load the released one-logit pooled head after the training model moved to 3 classes.

    PR #5 changed the train-time PatchHead heads to authentic/synthetic/tampered.
    The supplied pooled production checkpoint predates that change and retains
    the released one-logit score contract. This adapter reconstructs only those
    two output layers while leaving the backbone, LoRA weights, and score formula
    untouched.
    """

    base = detector_class(
        lora_r=int(checkpoint_payload.get("lora_r", 8)),
        pretrained=True,
    )
    patch_input = int(base.head.patch_logit.in_channels)
    feature_dim = int(base.backbone.embed_dim)
    base.head.patch_logit = torch.nn.Conv2d(patch_input, 1, 1)
    base.cls_head[-1] = torch.nn.Linear(feature_dim, 1)
    missing, unexpected = base.load_state_dict(
        checkpoint_payload["model"], strict=False
    )
    frozen = {name for name, parameter in base.named_parameters() if not parameter.requires_grad}
    optional_prefixes = ("distortion_head.", "threshold_adapter.")
    leaked = [
        name
        for name in missing
        if name not in frozen
        and name not in ("mean", "std")
        and not name.startswith(optional_prefixes)
    ]
    if leaked:
        raise PatchHeadInferenceError(
            f"Binary pooled checkpoint is missing trainable tensors: {leaked[:5]}"
        )
    if unexpected:
        raise PatchHeadInferenceError(
            f"Binary pooled checkpoint has unexpected tensors: {unexpected[:5]}"
        )

    class BinaryOutputAdapter(torch.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward_with_features(self, inputs: Any) -> tuple[Any, Any, Any, Any]:
            grid, cls = self.wrapped._encode_grid(inputs)
            image_logits, patch_logits = self.wrapped.head(grid)
            cls_logits = self.wrapped.cls_head(cls)
            return (
                image_logits.squeeze(1),
                cls_logits.squeeze(1),
                patch_logits.squeeze(1),
                grid,
            )

        def forward(self, inputs: Any) -> tuple[Any, Any, Any]:
            image_logits, cls_logits, patch_logits, _grid = self.forward_with_features(
                inputs
            )
            return image_logits, cls_logits, patch_logits

    return BinaryOutputAdapter(base).to(device).eval()


def _iter_images(path: Path, recursive: bool) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise PatchHeadInferenceError(f"Unsupported image extension: {path.suffix}")
        yield path
        return
    if not path.is_dir():
        raise PatchHeadInferenceError(f"Input does not exist: {path}")
    iterator = path.rglob("*") if recursive else path.glob("*")
    for candidate in sorted(iterator):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield candidate


def _relative_image_path(path: Path, input_path: Path) -> str:
    root = input_path if input_path.is_dir() else input_path.parent
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _prepare_image(
    path: Path, *, canonical_size: int, model_size: int
) -> tuple[np.ndarray, int, int]:
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise PatchHeadInferenceError(f"Could not decode image: {exc}") from exc
    width, height = image.size
    prepared = image.resize(
        (canonical_size, canonical_size), Image.Resampling.BICUBIC
    ).resize((model_size, model_size), Image.Resampling.BICUBIC)
    array = np.asarray(prepared, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1)), width, height


def _validate_outputs(
    image_logits: Any,
    cls_logits: Any,
    patch_logits: Any,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(image_logits, dtype=np.float64)
    cls = np.asarray(cls_logits, dtype=np.float64)
    patch = np.asarray(patch_logits, dtype=np.float64)
    if image.shape != (batch_size,):
        raise PatchHeadInferenceError(
            f"Image logits must have shape ({batch_size},), received {image.shape}"
        )
    if cls.shape != (batch_size,):
        raise PatchHeadInferenceError(
            f"CLS logits must have shape ({batch_size},), received {cls.shape}"
        )
    if patch.ndim != 3 or patch.shape[0] != batch_size or min(patch.shape[1:]) <= 0:
        raise PatchHeadInferenceError(
            "Patch logits must have shape (batch, rows, columns) with non-empty dimensions"
        )
    if not np.isfinite(image).all() or not np.isfinite(cls).all() or not np.isfinite(patch).all():
        raise PatchHeadInferenceError("PatchHead outputs must all be finite")
    return image, cls, patch


@dataclass(slots=True)
class TorchPatchHeadRuntime:
    """Official model runtime, loaded lazily so core contract tests stay light."""

    checkpoint: str | Path
    device: str | None = None
    metadata: dict[str, Any] = field(init=False)
    _torch: Any = field(init=False, repr=False)
    _model: Any = field(init=False, repr=False)
    _device: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise PatchHeadInferenceError(
                "PyTorch is required for real PatchHead inference. Install the detector dependencies."
            ) from exc
        try:
            from .model import DINOV3, PatchHeadDetector, get_device
        except ImportError:  # direct execution from patchhead/
            from model import (  # type: ignore
                DINOV3,
                PatchHeadDetector,
                get_device,
            )

        checkpoint_path = Path(self.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise PatchHeadInferenceError(
                f"PatchHead checkpoint does not exist: {checkpoint_path}. "
                "Supply the pooled checkpoint outside Git."
            )
        try:
            raw_payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            preflight = _preflight_checkpoint_payload(raw_payload)
        except PatchHeadInferenceError:
            raise
        except Exception as exc:  # pragma: no cover - requires malformed artifact
            raise PatchHeadInferenceError(
                f"Could not inspect PatchHead checkpoint metadata: {exc}"
            ) from exc
        selected_device = self.device or get_device()
        try:
            if preflight["output_classes"] == 1:
                model = _load_binary_compat_detector(
                    raw_payload,
                    detector_class=PatchHeadDetector,
                    torch=torch,
                    device=selected_device,
                )
                checkpoint_compatibility = "released_binary_head_adapter"
            else:
                raise PatchHeadInferenceError(
                    "The stable inference contract currently requires a pooled "
                    "one-logit PatchHead checkpoint. A new three-class/distortion-aware "
                    "pooled checkpoint also requires an explicitly versioned score contract."
                )
        except PatchHeadInferenceError:
            raise
        except Exception as exc:  # pragma: no cover - requires real model artifacts
            raise PatchHeadInferenceError(f"Could not load PatchHead: {exc}") from exc
        self._torch = torch
        self._model = model
        self._device = str(selected_device)
        self.metadata = {
            "family": "dino_patchhead",
            "arch": preflight["arch"],
            "backbone": DINOV3,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "checkpoint_dataset": preflight["dataset"],
            "threshold": preflight["threshold"],
            "model_input_size": preflight["model_size"],
            "score_kind": "uncalibrated_aigc_classifier_score",
            "score_formula": "0.5 * (sigmoid(mean(patch_logits)) + sigmoid(cls_logit))",
            "decision_rule": "aigc_score > threshold",
            "checkpoint_compatibility": checkpoint_compatibility,
        }

    def infer(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = self._torch
        tensor = torch.from_numpy(np.asarray(batch, dtype=np.float32)).to(self._device)
        device_type = self._device.split(":", 1)[0]
        with torch.inference_mode(), torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type != "cpu",
        ):
            image_logits, cls_logits, patch_logits = self._model(tensor)
        return (
            image_logits.float().cpu().numpy(),
            cls_logits.float().cpu().numpy(),
            patch_logits.float().cpu().numpy(),
        )

    def infer_with_features(
        self, batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return official outputs plus an in-memory ``(B,H,W,C)`` DINO grid."""
        torch = self._torch
        tensor = torch.from_numpy(np.asarray(batch, dtype=np.float32)).to(self._device)
        device_type = self._device.split(":", 1)[0]
        with torch.inference_mode(), torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type != "cpu",
        ):
            image_logits, cls_logits, patch_logits, feature_grid = (
                self._model.forward_with_features(tensor)
            )
        return (
            image_logits.float().cpu().numpy(),
            cls_logits.float().cpu().numpy(),
            patch_logits.float().cpu().numpy(),
            feature_grid.permute(0, 2, 3, 1).float().cpu().numpy(),
        )


def run_patchhead_inference(
    input_path: str | Path,
    *,
    checkpoint: str | Path | None = None,
    runtime: PatchHeadRuntime | None = None,
    device: str | None = None,
    batch_size: int = 16,
    recursive: bool = True,
    max_images: int | None = None,
    export_patch_evidence: bool = False,
    dense_feature_sink: DenseFeatureSink | None = None,
) -> dict[str, Any]:
    """Run primary inference and return a stable, physics-compatible payload."""

    if batch_size <= 0:
        raise PatchHeadInferenceError("batch_size must be positive")
    if max_images is not None and max_images <= 0:
        raise PatchHeadInferenceError("max_images must be positive")
    source = Path(input_path).expanduser().resolve()
    paths = list(_iter_images(source, recursive))
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise PatchHeadInferenceError("No supported images were found")
    if runtime is None:
        if checkpoint is None:
            raise PatchHeadInferenceError("A pooled PatchHead checkpoint is required")
        runtime = TorchPatchHeadRuntime(checkpoint, device)

    metadata = dict(runtime.metadata)
    try:
        threshold = float(metadata["threshold"])
        model_size = int(metadata["model_input_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PatchHeadInferenceError(
            "Runtime metadata must contain numeric threshold and model_input_size values"
        ) from exc
    if not 0.0 <= threshold <= 1.0 or model_size <= 0:
        raise PatchHeadInferenceError("Runtime threshold/size metadata is invalid")

    canonical_size = 200
    records: list[dict[str, Any] | None] = [None] * len(paths)
    pending: list[tuple[int, Path, np.ndarray, int, int]] = []

    def score_pending() -> None:
        if not pending:
            return
        batch = np.stack([item[2] for item in pending], axis=0)
        try:
            if dense_feature_sink is not None:
                infer_with_features = getattr(runtime, "infer_with_features", None)
                if not callable(infer_with_features):
                    raise PatchHeadInferenceError(
                        "The selected PatchHead runtime cannot expose same-pass dense features"
                    )
                runtime_outputs = infer_with_features(batch)
            else:
                runtime_outputs = runtime.infer(batch)
        except PatchHeadInferenceError:
            raise
        except Exception as exc:
            raise PatchHeadInferenceError(
                f"PatchHead runtime failed for a batch of {len(pending)} image(s): {exc}"
            ) from exc
        expected_outputs = 4 if dense_feature_sink is not None else 3
        if (
            not isinstance(runtime_outputs, (tuple, list))
            or len(runtime_outputs) != expected_outputs
        ):
            if dense_feature_sink is None:
                raise PatchHeadInferenceError(
                    "PatchHead runtime must return exactly image, CLS, and patch logits"
                )
            raise PatchHeadInferenceError(
                "PatchHead feature runtime must return image, CLS, patch logits, and dense features"
            )
        dense_features: np.ndarray | None = None
        if dense_feature_sink is not None:
            dense_features = np.asarray(runtime_outputs[3], dtype=np.float32)
            if (
                dense_features.ndim != 4
                or dense_features.shape[0] != len(pending)
                or min(dense_features.shape[1:], default=0) <= 0
                or not np.isfinite(dense_features).all()
            ):
                raise PatchHeadInferenceError(
                    "Dense DINO features must have finite shape (batch, rows, columns, channels)"
                )
        image_logits, cls_logits, patch_logits = _validate_outputs(
            *runtime_outputs[:3], batch_size=len(pending)
        )
        image_scores = _sigmoid(image_logits)
        cls_scores = _sigmoid(cls_logits)
        patch_scores = _sigmoid(patch_logits)
        combined = 0.5 * (image_scores + cls_scores)
        for offset, (record_index, path, _prepared, width, height) in enumerate(
            pending
        ):
            patch = patch_scores[offset]
            record: dict[str, Any] = {
                "image_path": _relative_image_path(path, source),
                "width": width,
                "height": height,
                "aigc_score": float(combined[offset]),
                "is_aigc": bool(combined[offset] > threshold),
                "component_scores": {
                    "patch_head": float(image_scores[offset]),
                    "cls_head": float(cls_scores[offset]),
                },
            }
            if export_patch_evidence:
                record["patch_evidence"] = {
                    "grid_shape": [int(patch.shape[0]), int(patch.shape[1])],
                    "coordinate_space": "normalized_full_frame",
                    "value_kind": "sigmoid_of_per_patch_aigc_logit_uncalibrated",
                    "values": np.round(patch, 8).tolist(),
                    "training_supervision": "image_label_repeated_across_all_patches",
                    "explains_score_component": "patch_head_only",
                }
            if dense_feature_sink is not None and dense_features is not None:
                dense_feature_sink(record["image_path"], dense_features[offset].copy())
            records[record_index] = record
        pending.clear()

    for index, path in enumerate(paths):
        image_path = _relative_image_path(path, source)
        try:
            prepared, width, height = _prepare_image(
                path, canonical_size=canonical_size, model_size=model_size
            )
        except PatchHeadInferenceError as exc:
            failed_record: dict[str, Any] = {
                "image_path": image_path,
                "error": str(exc),
                "aigc_score": None,
                "is_aigc": None,
                "component_scores": None,
            }
            if export_patch_evidence:
                failed_record["patch_evidence"] = None
            records[index] = failed_record
            continue
        pending.append((index, path, prepared, width, height))
        if len(pending) == batch_size:
            score_pending()
    score_pending()

    completed = [record for record in records if record is not None]
    failed = sum(record.get("error") is not None for record in completed)
    metadata.update(
        {
            "threshold": threshold,
            "model_input_size": model_size,
            "canonical_resize": [canonical_size, canonical_size],
            "patch_evidence_exported": export_patch_evidence,
            "dense_features_forwarded_in_memory": dense_feature_sink is not None,
        }
    )
    return {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "input_root": str(source if source.is_dir() else source.parent),
        "detector": metadata,
        "images": completed,
        "summary": {
            "discovered_images": len(paths),
            "processed_images": len(completed),
            "decode_failures": failed,
            "patch_evidence_exported": export_patch_evidence,
            "dense_features_forwarded_in_memory": dense_feature_sink is not None,
        },
        "limitations": [
            "Scores are not calibrated probabilities unless a separate calibration study establishes that property.",
            "Per-patch logits receive image-level supervision and are weak localization, not segmentation masks.",
            "Patch evidence describes only the patch-head component; the CLS head supplies half of the final score.",
        ],
    }


def write_json_atomic(path: str | Path, payload: Any, *, pretty: bool = False) -> Path:
    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            ensure_ascii=False,
        )
        handle.write("\n")
    temporary.replace(destination)
    return destination
