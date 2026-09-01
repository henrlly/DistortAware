"""Evaluate mask model on SID validation and held-out WildFake archives."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ai_detection.data import sid_raw_stream
from ai_detection.model import ResidualUNet, image_tensor, target_mask


@torch.inference_mode()
def predict(model, image, device, size):
    mask_logits, class_logits = model(image_tensor(image, size).unsqueeze(0).to(device))
    return torch.sigmoid(mask_logits)[0, 0].cpu().numpy(), torch.softmax(class_logits, 1)[0].cpu().numpy()


def sid_evaluation(model, device, size, per_class):
    confusion = np.zeros((2, 2), dtype=np.int64)
    intersections = unions = 0
    counts = [0, 0, 0]
    for sample in sid_raw_stream("validation", shuffle=False):
        label = int(sample["label"])
        if label not in range(3) or counts[label] >= per_class:
            continue
        mask, probabilities = predict(model, sample["image"], device, size)
        binary_label = int(label > 0)
        binary_prediction = int(probabilities.argmax() != 0)
        confusion[binary_label, binary_prediction] += 1
        if label == 2:
            truth = target_mask(sample["mask"], label, size)[0].numpy() >= 0.5
            predicted = mask >= 0.5
            intersections += int((truth & predicted).sum())
            unions += int((truth | predicted).sum())
        counts[label] += 1
        if min(counts) == per_class:
            break
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    return {
        "samples": int(confusion.sum()),
        "binary_accuracy": float(np.trace(confusion) / confusion.sum()),
        "balanced_accuracy": float(recalls.mean()),
        "authentic_accuracy": float(recalls[0]),
        "ai_accuracy": float(recalls[1]),
        "tamper_mask_iou": intersections / max(unions, 1),
        "confusion": confusion.tolist(),
    }


def archive_evaluation(model, device, size, archive, truth_fake, per_class, offset):
    records = []
    with zipfile.ZipFile(archive) as zipped:
        names = sorted(n for n in zipped.namelist() if Path(n).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        for name in names[offset:offset + per_class]:
            with zipped.open(name) as handle, Image.open(io.BytesIO(handle.read())) as image:
                mask, probabilities = predict(model, image.convert("RGB"), device, size)
            records.append({
                "correct": bool((probabilities.argmax() != 0) == truth_fake),
                "p_aigc": float(probabilities[1] + probabilities[2]),
                "predicted_area": float((mask >= 0.5).mean()),
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/mask_classifier.pt"))
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--offset", type=int, default=1100)
    parser.add_argument("--real", type=Path, default=Path("data/wildfake/Images/Real/celebahq.zip"))
    parser.add_argument("--fake", type=Path, default=Path("data/wildfake/Images/Diffusion_based/DDIM.zip"))
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    size = int(checkpoint["size"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ResidualUNet().to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    sid = sid_evaluation(model, device, size, args.per_class)
    real = archive_evaluation(model, device, size, args.real, False, args.per_class, args.offset)
    fake = archive_evaluation(model, device, size, args.fake, True, args.per_class, args.offset)
    report = {
        "SID_Set": sid,
        "WildFake": {
            "samples": len(real) + len(fake),
            "binary_accuracy": float(np.mean([r["correct"] for r in real + fake])),
            "authentic_accuracy": float(np.mean([r["correct"] for r in real])),
            "synthetic_accuracy": float(np.mean([r["correct"] for r in fake])),
            "mean_predicted_mask_area_authentic": float(np.mean([r["predicted_area"] for r in real])),
            "mean_predicted_mask_area_synthetic": float(np.mean([r["predicted_area"] for r in fake])),
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
