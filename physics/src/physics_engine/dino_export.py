"""Export the official PatchHead model's existing per-patch evidence tensor.

This module is an adapter, not a second DINO implementation.  It loads
``PatchHeadDetector`` from the official repository's ``patchhead/model.py`` and
serializes the ``patch_logits`` already returned by that model.  PyTorch and
timm are optional dependencies so the geometry-only physics engine stays light.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .engine import SUPPORTED_EXTENSIONS
from .integration import _write_json_atomic
from .schema import utc_now_iso


DINO_EXPORT_SCHEMA_VERSION = "0.1.0"


class DinoExportError(RuntimeError):
    """Raised when the official PatchHead adapter cannot export evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_images(path: Path, recursive: bool) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise DinoExportError(f"Unsupported image extension: {path.suffix}")
        yield path
        return
    if not path.is_dir():
        raise DinoExportError(f"Input does not exist: {path}")
    iterator = path.rglob("*") if recursive else path.glob("*")
    for candidate in sorted(iterator):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield candidate


def _load_patchhead_module(patchhead_dir: Path) -> ModuleType:
    model_path = patchhead_dir / "model.py"
    if not model_path.is_file():
        raise DinoExportError(f"Official PatchHead model was not found at {model_path}")
    spec = importlib.util.spec_from_file_location("techjam_official_patchhead_model", model_path)
    if spec is None or spec.loader is None:
        raise DinoExportError(f"Could not import official PatchHead model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise DinoExportError(
            f"PatchHead dependency {exc.name!r} is missing. Install the optional "
            "DINO dependencies with `pip install -e '.[dino]'`."
        ) from exc
    for symbol in ("load_detector", "get_device"):
        if not hasattr(module, symbol):
            raise DinoExportError(f"Official model module does not expose `{symbol}`")
    return module


def _relative_or_absolute(path: Path, root: Path) -> str:
    if root.is_dir():
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    return str(path)


def export_patchhead_evidence(
    *,
    input_path: str | Path,
    patchhead_dir: str | Path,
    checkpoint: str | Path,
    recursive: bool = False,
    max_images: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run the official model and return detector scores plus 2-D patch maps."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise DinoExportError(
            "PyTorch is required only for DINO export. Install `.[dino]` in the "
            "environment used for the official detector."
        ) from exc

    root = Path(input_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise DinoExportError(f"PatchHead checkpoint does not exist: {checkpoint_path}")
    module = _load_patchhead_module(Path(patchhead_dir).expanduser().resolve())
    selected_device = device or module.get_device()
    try:
        model, checkpoint_payload = module.load_detector(checkpoint_path, selected_device)
    except Exception as exc:
        raise DinoExportError(f"Could not load the official PatchHead checkpoint: {exc}") from exc

    image_paths = list(_iter_images(root, recursive))
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise DinoExportError("No supported images were found")
    threshold = float(checkpoint_payload.get("threshold", 0.5))
    canonical_size = 200
    requested_size = int(checkpoint_payload.get("size", 256))
    records: list[dict[str, Any]] = []

    for image_path in image_paths:
        try:
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            records.append(
                {
                    "image_path": _relative_or_absolute(image_path, root),
                    "error": f"Could not decode image: {exc}",
                    "aigc_score": None,
                    "is_aigc": None,
                    "patch_evidence": None,
                }
            )
            continue
        width, height = image.size
        prepared = image.resize(
            (canonical_size, canonical_size), Image.Resampling.BICUBIC
        ).resize((requested_size, requested_size), Image.Resampling.BICUBIC)
        array = np.asarray(prepared, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(selected_device)
        try:
            with torch.inference_mode():
                image_logit, cls_logit, patch_logits = model(tensor)
                patch_head_score = torch.sigmoid(image_logit.float())[0].item()
                cls_head_score = torch.sigmoid(cls_logit.float())[0].item()
                score = 0.5 * (patch_head_score + cls_head_score)
                patch_values = torch.sigmoid(patch_logits.float())[0].cpu().numpy()
        except Exception as exc:
            raise DinoExportError(f"PatchHead inference failed for {image_path}: {exc}") from exc

        records.append(
            {
                "image_path": _relative_or_absolute(image_path, root),
                "width": width,
                "height": height,
                "aigc_score": float(score),
                "is_aigc": bool(score > threshold),
                "component_scores": {
                    "patch_head": float(patch_head_score),
                    "cls_head": float(cls_head_score),
                },
                "patch_evidence": {
                    "grid_shape": [int(patch_values.shape[0]), int(patch_values.shape[1])],
                    "coordinate_space": "normalized_full_frame",
                    "value_kind": "sigmoid_of_per_patch_aigc_logit_uncalibrated",
                    "values": np.round(patch_values, 8).tolist(),
                    "training_supervision": "image_label_repeated_across_all_patches",
                    "explains_score_component": "patch_head_only",
                },
            }
        )

    return {
        "schema_version": DINO_EXPORT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "input_root": str(root if root.is_dir() else root.parent),
        "detector": {
            "family": "dino_patchhead",
            "arch": checkpoint_payload.get("arch", "patchhead-dinov3-vitl16"),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "threshold": threshold,
            "decision_rule": "aigc_score > threshold",
            "score_kind": "uncalibrated_aigc_classifier_score",
            "score_formula": "0.5 * (sigmoid(mean(patch_logits)) + sigmoid(cls_logit))",
            "canonical_resize": [canonical_size, canonical_size],
            "requested_model_input": [requested_size, requested_size],
        },
        "images": records,
        "limitations": [
            "Per-patch logits receive only image-level supervision and are not segmentation masks.",
            "The map covers the patch-head component, while half of the final score comes from the CLS head.",
            "Scores are not calibrated probabilities unless a separate calibration study establishes that property.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physics-dino-export",
        description="Export the official DINO PatchHead model's existing 2-D patch evidence map.",
    )
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("--patchhead-dir", required=True, help="Official repository's patchhead directory")
    parser.add_argument("--checkpoint", required=True, help="Official PatchHead checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--device", help="Optional torch device override, such as cpu, mps, or cuda")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_images is not None and args.max_images <= 0:
        raise SystemExit("--max-images must be positive")
    try:
        payload = export_patchhead_evidence(
            input_path=args.input,
            patchhead_dir=args.patchhead_dir,
            checkpoint=args.checkpoint,
            recursive=args.recursive,
            max_images=args.max_images,
            device=args.device,
        )
        output = Path(args.output).expanduser().resolve(strict=False)
        _write_json_atomic(output, payload, pretty=args.pretty)
    except (OSError, DinoExportError) as exc:
        print(f"physics-dino-export: {exc}", file=sys.stderr)
        return 2
    failed = sum(image.get("error") is not None for image in payload["images"])
    print(
        f"Exported {len(payload['images'])} DINO record(s); {failed} decode failure(s). "
        f"Output: {output}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
