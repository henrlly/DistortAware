"""Independent single-image and batch evaluation entry point for PatchHead."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

import numpy as np
from PIL import Image

from .inference import run_patchhead_inference


def run(
    input_path: str | Path,
    *,
    checkpoint: str | Path,
    device: str | None = None,
    export_patch_evidence: bool = True,
) -> dict[str, Any]:
    """Return the common detector result for one image."""
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"PatchHead entry point expects one image file: {path}")
    payload = run_patchhead_inference(
        path,
        checkpoint=checkpoint,
        device=device,
        recursive=False,
        export_patch_evidence=export_patch_evidence,
    )
    record = payload["images"][0]
    threshold = float(payload["detector"]["threshold"])
    return {
        "method": "patchhead",
        "image_path": str(path),
        "score": record["aigc_score"],
        "score_kind": "aigc_classifier_score",
        "confidence": None,
        "threshold": threshold,
        "decision": record["is_aigc"],
        "details": {
            "component_scores": record.get("component_scores"),
            "patch_evidence": record.get("patch_evidence"),
            "detector": payload["detector"],
        },
    }


def _image_paths(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir).expanduser().resolve()
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    paths = [root] if root.is_file() else sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise ValueError(f"PatchHead entry point found no images: {root}")
    return paths


def run_batch(
    image_dir: str | Path,
    *,
    checkpoint: str | Path,
    distortion_aware: bool = False,
    device: str | None = None,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Evaluate a three-class PatchHead checkpoint while loading it once."""
    import torch

    from .model import get_device, load_detector

    selected_device = device or get_device()
    model, payload = load_detector(checkpoint, selected_device)
    checkpoint_aware = bool(payload.get("distortion_aware", False))
    if distortion_aware and not checkpoint_aware:
        raise ValueError("distortion-aware evaluation requires a distortion-aware checkpoint")
    threshold = float(payload.get("threshold", 0.5))
    size = int(payload.get("size", 256))
    paths = _image_paths(image_dir)
    results: list[dict[str, Any]] = []
    for offset in range(0, len(paths), batch_size):
        chunk = paths[offset:offset + batch_size]
        tensors = []
        for path in chunk:
            with Image.open(path) as opened:
                image = opened.convert("RGB").resize((200, 200), Image.Resampling.BICUBIC)
                image = image.resize((size, size), Image.Resampling.BICUBIC)
                array = np.asarray(image, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(array).permute(2, 0, 1))
        inputs = torch.stack(tensors).to(selected_device)
        with torch.inference_mode(), torch.autocast(
            device_type=selected_device.split(":")[0], dtype=torch.bfloat16,
            enabled=selected_device != "cpu",
        ):
            if distortion_aware:
                image_logits, cls_logits, _patch_logits, aux = model.forward_distortion_aware(inputs)
                scores = aux["corrected_score"].float()
            else:
                image_logits, cls_logits, _patch_logits = model(inputs)
                scores = .5 * (
                    torch.softmax(image_logits.float(), dim=1)[:, 1:].sum(dim=1)
                    + torch.softmax(cls_logits.float(), dim=1)[:, 1:].sum(dim=1)
                )
            probabilities = .5 * (
                torch.softmax(image_logits.float(), dim=1)
                + torch.softmax(cls_logits.float(), dim=1)
            )
        for index, path in enumerate(chunk):
            score = float(scores[index].cpu())
            class_probabilities = probabilities[index].cpu().tolist()
            details: dict[str, Any] = {"class_probabilities": class_probabilities}
            if distortion_aware:
                details.update({
                    "base_score": float(aux["base_score"][index].float().cpu()),
                    "threshold_logit_shift": float(aux["threshold_shift"][index].float().cpu()),
                    "predicted_distortion": aux["predicted_distortion"][index].float().cpu().tolist(),
                })
            results.append({
                "method": "patchhead_distortion_aware" if distortion_aware else "patchhead_baseline",
                "image_path": str(path), "score": score,
                "score_kind": "corrected_aigc_classifier_score" if distortion_aware else "aigc_classifier_score",
                "confidence": float(max(class_probabilities)), "threshold": threshold,
                "decision": score >= threshold, "details": details, "errors": [],
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--distortion-aware", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    records = run_batch(args.image_dir, checkpoint=args.checkpoint,
                        distortion_aware=args.distortion_aware,
                        device=args.device, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"PatchHead evaluation complete: {len(records)} images -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
