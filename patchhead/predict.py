"""Distortion-aware PatchHead inference for one or more local images."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import get_device, load_detector
from transforms import MAGNITUDE_NAMES, TYPE_NAMES


def prepare(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((200, 200), Image.Resampling.BICUBIC)
        image = image.resize((size, size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def physical_estimates(magnitudes: dict[str, float]) -> dict[str, float]:
    """Convert normalized predictions to approximate human-readable units."""
    return {
        "jpeg_quality": float(np.clip(100 - 75 * magnitudes["jpeg_severity"], 1, 100)),
        "blur_sigma": 2.25 * magnitudes["blur_sigma"],
        "resize_scale": float(np.clip(1 - .8 * magnitudes["resize_loss"], .05, 1)),
        "noise_sigma": .11 * magnitudes["noise_sigma"],
        "brightness_delta": .35 * magnitudes["brightness_delta"],
        "contrast_delta": .35 * magnitudes["contrast_delta"],
        "saturation_delta": .35 * magnitudes["saturation_delta"],
        "crop_retained_fraction": float(np.clip(1 - .55 * magnitudes["crop_loss"], .05, 1)),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--ckpt", default="patchhead/checkpoints/patchhead.pt")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    device = get_device()
    model, checkpoint = load_detector(args.ckpt, device)
    if not checkpoint.get("distortion_aware", False):
        raise SystemExit("checkpoint is not distortion-aware; retrain with the current patchhead/train.py")
    threshold = float(checkpoint.get("threshold", .5))
    size = int(checkpoint.get("size", 256))
    results = []
    for path in args.images:
        x = prepare(path, size).unsqueeze(0).to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=device != "cpu"):
            _, _, _, aux = model.forward_distortion_aware(x)
        type_values = aux["type_probabilities"][0].float().cpu().numpy()
        # Report gated severities: a magnitude is meaningful only when the
        # corresponding distortion type is believed to be present.
        magnitude_values = aux["predicted_distortion"][0, len(TYPE_NAMES):].float().cpu().numpy()
        distortion_types = {name: float(value) for name, value in zip(TYPE_NAMES, type_values)}
        magnitudes = {name: float(value) for name, value in zip(MAGNITUDE_NAMES, magnitude_values)}
        shift = float(aux["threshold_shift"][0].float().cpu())
        threshold_logit = math.log(max(threshold, 1e-6) / max(1 - threshold, 1e-6))
        dynamic_threshold = 1 / (1 + math.exp(-(threshold_logit + shift)))
        base_score = float(aux["base_score"][0].float().cpu())
        corrected_score = float(aux["corrected_score"][0].float().cpu())
        results.append({
            "image_path": str(path),
            "pred": corrected_score,
            "is_aigc": bool(corrected_score >= threshold),
            "base_score": base_score,
            "global_corrected_score_threshold": threshold,
            "equivalent_dynamic_base_threshold": dynamic_threshold,
            "threshold_logit_shift": shift,
            "distortion_type_probabilities": distortion_types,
            "distortion_normalized_magnitudes": magnitudes,
            "distortion_physical_estimates": physical_estimates(magnitudes),
        })
    output = json.dumps(results, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
