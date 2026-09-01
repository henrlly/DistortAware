"""Independent single-image and batch entry point for the filter baseline."""

from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from typing import Any


_SOURCE = Path(__file__).resolve().parent / "src"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from ai_detection import MaskPredictor  # noqa: E402


def run(
    input_path: str | Path,
    *,
    checkpoint: str | Path = "models/mask_classifier.pt",
) -> dict[str, Any]:
    """Return the common detector result for one image."""
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Filter entry point expects one image file: {path}")
    result, _mask = MaskPredictor(checkpoint).predict_path(path)
    return {
        "method": "filter",
        "image_path": str(path),
        "score": result["ai_probability"],
        "score_kind": "ai_probability",
        "confidence": result["confidence"],
        "threshold": 0.5,
        "decision": result["ai_probability"] >= 0.5,
        "details": result,
    }


def run_batch(
    image_dir: str | Path,
    *,
    checkpoint: str | Path,
) -> list[dict[str, Any]]:
    root = Path(image_dir).expanduser().resolve()
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    paths = [root] if root.is_file() else sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise ValueError(f"Filter entry point found no images: {root}")
    predictor = MaskPredictor(checkpoint)
    records = []
    for path in paths:
        result, _mask = predictor.predict_path(path)
        score = float(result["ai_probability"])
        records.append({
            "method": "filter", "image_path": str(path), "score": score,
            "score_kind": "ai_probability", "confidence": float(result["confidence"]),
            "threshold": 0.5, "decision": score >= 0.5,
            "details": result, "errors": [],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parent / "models/mask_classifier.pt")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = run_batch(args.image_dir, checkpoint=args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Filter evaluation complete: {len(records)} images -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
