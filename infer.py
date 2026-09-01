"""Unified deliverable inference entry point.

PatchHead is the default primary detector. DID remains available as an explicit
ablation. PatchHead can export its existing patch logits and run the physics
explanation sidecar without a second detector forward pass.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out", default="preds.json")
    parser.add_argument("--detector", choices=("patchhead", "did"), default="patchhead")
    parser.add_argument("--ckpt", help="Detector checkpoint; defaults depend on --detector")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--export-patch-evidence", action="store_true")
    parser.add_argument("--with-physics", action="store_true")
    parser.add_argument("--physics-annotations")
    parser.add_argument("--physics-overlays-dir")
    parser.add_argument("--physics-auto-proposals", action="store_true")
    parser.add_argument(
        "--physics-proposal-mask-backend",
        choices=("heuristic", "clipseg"),
        default="heuristic",
    )
    parser.add_argument(
        "--physics-proposal-feature-backend",
        choices=("appearance", "dinov3", "patchhead", "none"),
        default="appearance",
    )
    parser.add_argument(
        "--physics-proposal-object-backend",
        choices=("edges", "torchvision", "none"),
        default="edges",
    )
    parser.add_argument(
        "--physics-proposal-mask-model",
        default="CIDAS/clipseg-rd64-refined",
        help="CLIPSeg model ID or local path for automatic region proposals",
    )
    parser.add_argument(
        "--physics-proposal-dino-model",
        default="vit_small_patch16_dinov3.lvd1689m",
        help="timm DINO model used for automatic reflection correspondence",
    )
    parser.add_argument(
        "--physics-proposal-mask-revision",
        default="999e0328d9e10b484360c477313983f9afdd7050",
        help="Pinned CLIPSeg Hub revision",
    )
    parser.add_argument(
        "--physics-proposal-dino-revision",
        default="3bf4720a82ec2066db88137180ff1f83a675cef0",
        help="Pinned standalone DINO Hub revision",
    )
    parser.add_argument("--physics-proposal-cache-dir")
    parser.add_argument("--physics-proposal-device")
    parser.add_argument("--physics-proposal-offline", action="store_true")
    parser.add_argument("--physics-strict-proposal-models", action="store_true")
    parser.add_argument("--physics-proposal-shadow-threshold", type=float, default=0.40)
    parser.add_argument("--physics-proposal-mirror-threshold", type=float, default=0.54)
    parser.add_argument(
        "--physics-feature-memory-mib",
        type=int,
        default=512,
        help="Maximum in-memory same-pass DINO feature transfer (default: 512 MiB)",
    )
    parser.add_argument("--allow-missing-physics", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--res", type=int, default=192, help="DID-only reconstruction size")
    parser.add_argument("--steps", type=int, default=6, help="DID-only reconstruction steps")
    return parser


def _physics_modules():
    physics_source = REPOSITORY_ROOT / "physics" / "src"
    if str(physics_source) not in sys.path:
        sys.path.insert(0, str(physics_source))
    from physics_engine.dino_integration import merge_dino_and_physics
    from physics_engine.automatic_config import AutomaticProposalConfig
    from physics_engine.engine import PhysicsEngine, PhysicsEngineConfig
    from physics_engine.integration import merge_detector_and_physics

    return (
        AutomaticProposalConfig,
        PhysicsEngine,
        PhysicsEngineConfig,
        merge_dino_and_physics,
        merge_detector_and_physics,
    )


def _run_physics(
    args: argparse.Namespace, *, dense_feature_maps: dict[str, Any] | None = None
) -> dict[str, Any]:
    AutomaticProposalConfig, PhysicsEngine, PhysicsEngineConfig, _, _ = _physics_modules()
    feature_backend = getattr(args, "physics_proposal_feature_backend", "appearance")
    automatic = AutomaticProposalConfig(
        enabled=bool(getattr(args, "physics_auto_proposals", False)),
        mask_backend=getattr(args, "physics_proposal_mask_backend", "heuristic"),
        feature_backend="external" if feature_backend == "patchhead" else feature_backend,
        object_backend=getattr(args, "physics_proposal_object_backend", "edges"),
        mask_model=getattr(
            args,
            "physics_proposal_mask_model",
            "CIDAS/clipseg-rd64-refined",
        ),
        dino_model=getattr(
            args,
            "physics_proposal_dino_model",
            "vit_small_patch16_dinov3.lvd1689m",
        ),
        mask_revision=getattr(
            args,
            "physics_proposal_mask_revision",
            "999e0328d9e10b484360c477313983f9afdd7050",
        )
        or None,
        dino_revision=getattr(
            args,
            "physics_proposal_dino_revision",
            "3bf4720a82ec2066db88137180ff1f83a675cef0",
        )
        or None,
        device=getattr(args, "physics_proposal_device", None) or args.device,
        cache_dir=getattr(args, "physics_proposal_cache_dir", None),
        local_files_only=bool(getattr(args, "physics_proposal_offline", False)),
        allow_model_fallback=not bool(
            getattr(args, "physics_strict_proposal_models", False)
        ),
        shadow_threshold=float(
            getattr(args, "physics_proposal_shadow_threshold", 0.40)
        ),
        mirror_threshold=float(
            getattr(args, "physics_proposal_mirror_threshold", 0.54)
        ),
    )
    result = PhysicsEngine(PhysicsEngineConfig(automatic=automatic)).run(
        args.image_dir,
        annotations_path=args.physics_annotations,
        overlays_dir=args.physics_overlays_dir,
        recursive=True,
        max_images=args.max_images,
        dense_feature_maps=dense_feature_maps,
    )
    return result.to_dict()


def _run_patchhead(
    args: argparse.Namespace, *, runtime: Any | None = None
) -> dict[str, Any]:
    from patchhead.inference import PatchHeadInferenceError, run_patchhead_inference

    checkpoint = None
    if runtime is None:
        checkpoint = args.ckpt or str(
            REPOSITORY_ROOT / "patchhead" / "checkpoints" / "patchhead_pooled.pt"
        )
    dense_feature_maps: dict[str, Any] | None = None
    dense_feature_bytes = 0
    use_same_pass_features = bool(
        args.with_physics
        and getattr(args, "physics_auto_proposals", False)
        and getattr(args, "physics_proposal_feature_backend", "appearance")
        == "patchhead"
    )
    if use_same_pass_features:
        dense_feature_maps = {}

    def collect_dense_features(image_path: str, values: Any) -> None:
        nonlocal dense_feature_bytes
        import numpy as np

        compact = np.asarray(values, dtype=np.float16)
        dense_feature_bytes += int(compact.nbytes)
        limit_mib = int(getattr(args, "physics_feature_memory_mib", 512))
        if dense_feature_bytes > limit_mib * 1024 * 1024:
            raise PatchHeadInferenceError(
                f"Same-pass physics feature transfer exceeded {limit_mib} MiB; "
                "lower --max-images or use the standalone dinov3/appearance backend"
            )
        assert dense_feature_maps is not None
        dense_feature_maps[image_path] = compact

    payload = run_patchhead_inference(
        args.image_dir,
        checkpoint=checkpoint,
        runtime=runtime,
        device=args.device,
        batch_size=args.batch,
        max_images=args.max_images,
        export_patch_evidence=args.export_patch_evidence or args.with_physics,
        dense_feature_sink=collect_dense_features if use_same_pass_features else None,
    )
    if dense_feature_maps is not None:
        detector = payload.get("detector", {})
        for image_path, values in list(dense_feature_maps.items()):
            dense_feature_maps[image_path] = {
                "values": values,
                "backend": "shared_patchhead_dinov3_tokens",
                "model": detector.get("backbone"),
                "metadata": {
                    "source_detector_family": detector.get("family"),
                    "source_checkpoint_sha256": detector.get("checkpoint_sha256"),
                    "source_checkpoint_dataset": detector.get("checkpoint_dataset"),
                    "feature_dtype": str(values.dtype),
                    "score_independent": True,
                },
            }
    if args.with_physics:
        _, _, _, merge_dino_and_physics, _ = _physics_modules()
        physics = _run_physics(args, dense_feature_maps=dense_feature_maps)
        payload, summary = merge_dino_and_physics(
            payload,
            physics,
            path_root=REPOSITORY_ROOT,
            allow_missing=args.allow_missing_physics,
        )
        payload["summary"]["physics_integration"] = summary
    return payload


def _run_did(args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        import glob
        import numpy as np
        import torch
        from PIL import Image, ImageOps
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"DID dependency {exc.name!r} is missing") from exc

    source_path = REPOSITORY_ROOT / "did"
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    from did import get_device, make_reconstructor
    from model import DIDClassifier

    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    paths = sorted(
        path
        for path in glob.glob(os.path.join(args.image_dir, "**", "*"), recursive=True)
        if path.lower().endswith(extensions)
    )
    if args.max_images is not None:
        paths = paths[: args.max_images]
    if not paths:
        raise RuntimeError(f"No supported images found in {args.image_dir}")

    checkpoint = Path(args.ckpt or REPOSITORY_ROOT / "checkpoints" / "did.pt")
    if not checkpoint.is_file():
        raise RuntimeError(f"DID checkpoint does not exist: {checkpoint}")
    device = args.device or get_device()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    reconstructor = make_reconstructor(
        payload.get("recon", "sd15"), res=args.res, steps=args.steps, device=device
    )
    classifier = DIDClassifier(
        pretrained=False, backbone=payload.get("backbone", "resnet18")
    ).to(device).eval()
    classifier.load_state_dict(payload["model"])
    threshold = float(payload.get("threshold", 0.5))

    def preprocess(path: str):
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image = image.resize((200, 200), Image.Resampling.BICUBIC).resize(
            (args.res, args.res), Image.Resampling.BICUBIC
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    records: list[dict[str, Any]] = []
    for start in range(0, len(paths), args.batch):
        chunk = paths[start : start + args.batch]
        images = torch.stack([preprocess(path) for path in chunk])
        first, second = reconstructor.did_features(images)
        first = torch.nn.functional.interpolate(
            first, size=256, mode="bilinear", align_corners=False
        )
        second = torch.nn.functional.interpolate(
            second, size=256, mode="bilinear", align_corners=False
        )
        with torch.no_grad():
            scores = classifier.score(first.to(device), second.to(device)).cpu().numpy()
        for path, score in zip(chunk, scores):
            records.append(
                {
                    "image_path": path,
                    "pred": round(float(score), 6),
                    "is_aigc": bool(score > threshold),
                }
            )
    if args.with_physics:
        _, _, _, _, merge_detector_and_physics = _physics_modules()
        records, _ = merge_detector_and_physics(
            records,
            _run_physics(args),
            path_root=REPOSITORY_ROOT,
            allow_missing=args.allow_missing_physics,
        )
    return records


def _write_json(path: str | Path, payload: Any, *, pretty: bool) -> Path:
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch <= 0:
        raise SystemExit("--batch must be positive")
    if args.max_images is not None and args.max_images <= 0:
        raise SystemExit("--max-images must be positive")
    if args.detector == "did" and args.export_patch_evidence:
        raise SystemExit("--export-patch-evidence is available only for PatchHead")
    if args.physics_feature_memory_mib <= 0 or args.physics_feature_memory_mib > 2048:
        raise SystemExit("--physics-feature-memory-mib must lie within [1, 2048]")
    if (
        args.detector == "did"
        and args.physics_auto_proposals
        and args.physics_proposal_feature_backend == "patchhead"
    ):
        raise SystemExit("The patchhead physics feature backend requires --detector patchhead")
    try:
        payload = _run_patchhead(args) if args.detector == "patchhead" else _run_did(args)
        output = _write_json(args.out, payload, pretty=args.pretty)
    except Exception as exc:
        print(f"infer: {exc}", file=sys.stderr)
        return 2
    count = len(payload["images"]) if isinstance(payload, dict) else len(payload)
    print(f"wrote {output} ({count} image record(s), detector={args.detector})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
