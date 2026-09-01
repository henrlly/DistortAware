"""Command-line interface for directory-scale physics analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .automatic_config import (
    AutomaticProposalConfig,
    DEFAULT_CLIPSEG_REVISION,
    DEFAULT_DINO_REVISION,
)
from .engine import ENGINE_VERSION, PhysicsEngine, PhysicsEngineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physics-engine",
        description=(
            "Produce applicability-aware geometric consistency evidence. Output scores "
            "are physics violations, not AIGC probabilities."
        ),
    )
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument(
        "--output",
        default="outputs/physics_results.json",
        help="JSON output path (default: outputs/physics_results.json)",
    )
    parser.add_argument(
        "--annotations",
        help=(
            "Optional reviewed perspective/shadow/reflection JSON file or directory"
        ),
    )
    parser.add_argument("--overlays-dir", help="Optional directory for cue overlays")
    parser.add_argument(
        "--auto-proposals",
        action="store_true",
        help="Automatically propose shadow and reflection correspondences when reviewed pairs are absent",
    )
    parser.add_argument(
        "--proposal-mask-backend",
        choices=("heuristic", "clipseg"),
        default="heuristic",
        help="Region proposal backend (clipseg requires the optional auto dependencies)",
    )
    parser.add_argument(
        "--proposal-feature-backend",
        choices=("appearance", "dinov3", "none"),
        default="appearance",
        help="Dense feature backend for direct/reflected correspondence",
    )
    parser.add_argument(
        "--proposal-object-backend",
        choices=("edges", "torchvision", "none"),
        default="edges",
        help="Object-ground contact aid for shadow proposals",
    )
    parser.add_argument(
        "--proposal-mask-model",
        default="CIDAS/clipseg-rd64-refined",
        help="CLIPSeg model ID or local path",
    )
    parser.add_argument(
        "--proposal-dino-model",
        default="vit_small_patch16_dinov3.lvd1689m",
        help="timm DINO model used for reflection correspondence",
    )
    parser.add_argument(
        "--proposal-mask-revision",
        default=DEFAULT_CLIPSEG_REVISION,
        help="Pinned CLIPSeg Hub revision (use an empty value only for an intentional unpinned run)",
    )
    parser.add_argument(
        "--proposal-dino-revision",
        default=DEFAULT_DINO_REVISION,
        help="Pinned DINO Hub revision (use an empty value only for an intentional unpinned run)",
    )
    parser.add_argument("--proposal-device", help="Optional torch device for proposal models")
    parser.add_argument(
        "--proposal-cache-dir",
        help="External/ignored cache directory for optional model artifacts",
    )
    parser.add_argument(
        "--proposal-offline",
        action="store_true",
        help="Use only already cached/local proposal-model artifacts",
    )
    parser.add_argument(
        "--strict-proposal-models",
        action="store_true",
        help="Fail the automatic cues instead of falling back to deterministic OpenCV proposals",
    )
    parser.add_argument(
        "--proposal-shadow-threshold",
        type=float,
        default=0.40,
        help="Shadow-region probability threshold (default: 0.40)",
    )
    parser.add_argument(
        "--proposal-mirror-threshold",
        type=float,
        default=0.54,
        help="Mirror-region probability threshold (default: 0.54)",
    )
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories")
    parser.add_argument("--max-images", type=int, help="Process at most this many images")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero status if any image reports an error",
    )
    parser.add_argument("--version", action="version", version=ENGINE_VERSION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_images is not None and args.max_images <= 0:
        raise SystemExit("--max-images must be positive")

    try:
        automatic = AutomaticProposalConfig(
            enabled=args.auto_proposals,
            mask_backend=args.proposal_mask_backend,
            feature_backend=args.proposal_feature_backend,
            object_backend=args.proposal_object_backend,
            mask_model=args.proposal_mask_model,
            dino_model=args.proposal_dino_model,
            mask_revision=args.proposal_mask_revision or None,
            dino_revision=args.proposal_dino_revision or None,
            device=args.proposal_device,
            cache_dir=args.proposal_cache_dir,
            local_files_only=args.proposal_offline,
            allow_model_fallback=not args.strict_proposal_models,
            shadow_threshold=args.proposal_shadow_threshold,
            mirror_threshold=args.proposal_mirror_threshold,
        )
        engine = PhysicsEngine(PhysicsEngineConfig(automatic=automatic))
        result = engine.run(
            args.input,
            annotations_path=args.annotations,
            overlays_dir=args.overlays_dir,
            recursive=args.recursive,
            max_images=args.max_images,
        )
    except Exception as exc:
        print(f"physics-engine: {exc}", file=sys.stderr)
        return 2

    if result.summary["processed_images"] == 0:
        print("physics-engine: no supported images were found", file=sys.stderr)
        return 2

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            result.to_dict(),
            handle,
            indent=2 if args.pretty else None,
            separators=None if args.pretty else (",", ":"),
            ensure_ascii=False,
        )
        handle.write("\n")
    temporary_path.replace(output_path)

    summary = result.summary
    print(
        f"Processed {summary['processed_images']} image(s); "
        f"{summary['images_with_errors']} reported errors. Results: {output_path}"
    )
    if args.strict and summary["images_with_errors"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
