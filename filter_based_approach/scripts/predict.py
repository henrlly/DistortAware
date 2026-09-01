"""Predict whether one image is authentic or AI-generated/tampered."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ai_detection import MaskPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/mask_classifier.pt"))
    parser.add_argument("--mask-output", type=Path)
    args = parser.parse_args()
    result, mask = MaskPredictor(args.checkpoint).predict_path(args.image)
    print(json.dumps(result, indent=2))
    if args.mask_output:
        Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8)).save(args.mask_output)


if __name__ == "__main__":
    main()
