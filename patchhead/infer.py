"""Direct CLI for the primary PatchHead detector."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from .inference import PatchHeadInferenceError, run_patchhead_inference, write_json_atomic
except ImportError:  # direct `python patchhead/infer.py` execution
    from inference import PatchHeadInferenceError, run_patchhead_inference, write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pooled PatchHead inference with optional same-pass patch evidence."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out", default="preds.json")
    parser.add_argument(
        "--ckpt",
        default=str(Path(__file__).resolve().parent / "checkpoints" / "patchhead_pooled.pt"),
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--export-patch-evidence", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_patchhead_inference(
            args.image_dir,
            checkpoint=args.ckpt,
            device=args.device,
            batch_size=args.batch,
            recursive=not args.no_recursive,
            max_images=args.max_images,
            export_patch_evidence=args.export_patch_evidence,
        )
        output = write_json_atomic(args.out, payload, pretty=args.pretty)
    except (OSError, PatchHeadInferenceError) as exc:
        print(f"patchhead-infer: {exc}", file=sys.stderr)
        return 2
    print(
        f"Processed {payload['summary']['processed_images']} image(s); "
        f"{payload['summary']['decode_failures']} decode failure(s). Output: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
