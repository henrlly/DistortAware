"""Independent single-image and batch entry point for the DID detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

import torch

from .did import get_device, load_image, make_reconstructor
from .model import DIDClassifier


def run(
    input_path: str | Path,
    *,
    checkpoint: str | Path = "checkpoints/did.pt",
    reconstructor: str | None = None,
    resolution: int = 192,
    steps: int = 6,
    device: str | None = None,
) -> dict[str, Any]:
    """Return the common detector result for one image."""
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"DID entry point expects one image file: {path}")

    selected_device = device or get_device()
    checkpoint_data = torch.load(checkpoint, map_location=selected_device, weights_only=False)
    model = DIDClassifier(
        pretrained=False,
        backbone=checkpoint_data.get("backbone", "resnet18"),
    ).to(selected_device).eval()
    model.load_state_dict(checkpoint_data["model"])
    reconstructor_name = reconstructor or checkpoint_data.get("recon", "sd15")
    reconstructor_model = make_reconstructor(
        reconstructor_name,
        res=resolution,
        steps=steps,
        device=selected_device,
    )
    image = load_image(path, res=resolution).unsqueeze(0)
    d1, d2 = reconstructor_model.did_features(image)
    score = float(model.score(d1.to(selected_device), d2.to(selected_device))[0].cpu())
    threshold = float(checkpoint_data.get("threshold", 0.5))
    return {
        "method": "did",
        "image_path": str(path),
        "score": score,
        "score_kind": "aigc_classifier_score",
        "confidence": None,
        "threshold": threshold,
        "decision": score > threshold,
        "details": {
            "backbone": checkpoint_data.get("backbone", "resnet18"),
            "reconstructor": reconstructor_name,
            "resolution": resolution,
            "steps": steps,
        },
    }


def run_batch(
    image_dir: str | Path,
    *,
    checkpoint: str | Path,
    reconstructor: str | None = None,
    resolution: int = 256,
    steps: int = 10,
    device: str | None = None,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Evaluate a directory while loading the reconstructor and classifier once."""
    root = Path(image_dir).expanduser().resolve()
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    paths = [root] if root.is_file() else sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise ValueError(f"DID entry point found no images: {root}")
    selected_device = device or get_device()
    checkpoint_data = torch.load(checkpoint, map_location=selected_device, weights_only=False)
    model = DIDClassifier(pretrained=False, backbone=checkpoint_data.get("backbone", "resnet18")).to(selected_device).eval()
    model.load_state_dict(checkpoint_data["model"])
    reconstructor_name = reconstructor or checkpoint_data.get("recon", "sd15")
    reconstruction = make_reconstructor(reconstructor_name, res=resolution, steps=steps,
                                        device=selected_device)
    threshold = float(checkpoint_data.get("threshold", 0.5))
    records: list[dict[str, Any]] = []
    for offset in range(0, len(paths), batch_size):
        chunk = paths[offset:offset + batch_size]
        images = torch.stack([load_image(path, res=resolution) for path in chunk])
        d1, d2 = reconstruction.did_features(images)
        with torch.inference_mode(), torch.autocast(
            device_type=selected_device.split(":")[0], dtype=torch.bfloat16,
            enabled=selected_device != "cpu",
        ):
            scores = model.score(d1.to(selected_device), d2.to(selected_device)).float().cpu()
        for path, value in zip(chunk, scores):
            score = float(value)
            records.append({
                "method": "did", "image_path": str(path), "score": score,
                "score_kind": "aigc_classifier_score",
                "confidence": max(score, 1.0 - score), "threshold": threshold,
                "decision": score >= threshold, "errors": [],
                "details": {"backbone": checkpoint_data.get("backbone", "resnet18"),
                            "reconstructor": reconstructor_name,
                            "resolution": resolution, "steps": steps},
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reconstructor")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    records = run_batch(args.image_dir, checkpoint=args.checkpoint,
                        reconstructor=args.reconstructor, resolution=args.resolution,
                        steps=args.steps, device=args.device, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"DID evaluation complete: {len(records)} images -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
