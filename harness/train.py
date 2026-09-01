"""Run reproducible baseline and distortion-aware PatchHead training."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .common import fingerprint, load_manifest


def train(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).expanduser().resolve()
    manifest_fingerprints = {}
    for name in ("train.csv", "validation.csv", "calibration.csv", "test.csv"):
        manifest = data_dir / name
        if name == "test.csv" and not manifest.is_file():
            manifest = data_dir / "matched_test.csv"
        manifest_fingerprints[name] = fingerprint(load_manifest(manifest))
    modes = ("baseline", "distortion_aware") if args.mode == "both" else (args.mode,)
    for mode in modes:
        mode_dir = output / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = mode_dir / "checkpoint.pt"
        command = [sys.executable, "patchhead/train.py", "--ds", "pooled",
                   "--manifest-dir", str(data_dir),
                   "--epochs", str(args.epochs), "--bs", str(args.bs), "--seed", str(args.seed),
                   "--out", str(checkpoint)]
        if args.init_checkpoint:
            command.extend(["--init-checkpoint", str(Path(args.init_checkpoint).expanduser().resolve())])
        command.append("--no-distortion-aware" if mode == "baseline" else "--distortion-aware")
        subprocess.run(command, cwd=Path(__file__).resolve().parent.parent, check=True)
        import torch
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if checkpoint_payload.get("ds") != "pooled":
            raise RuntimeError(f"{mode} checkpoint is not pooled: {checkpoint_payload.get('ds')!r}")
        if bool(checkpoint_payload.get("distortion_aware", False)) != (mode == "distortion_aware"):
            raise RuntimeError(f"{mode} checkpoint distortion_aware metadata is invalid")
        metadata = {"mode": mode, "checkpoint": str(checkpoint), "data_dir": str(data_dir),
                    "seed": args.seed, "epochs": args.epochs, "batch_size": args.bs,
                    "manifest_fingerprints": manifest_fingerprints}
        (mode_dir / "training.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "comparison.json").write_text(json.dumps({"modes": modes, "data_dir": str(data_dir),
                                                          "manifest_fingerprints": manifest_fingerprints}, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "distortion_aware", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/quick_training"))
    parser.add_argument("--init-checkpoint", help="Initialize the selected mode from an existing PatchHead checkpoint")
    return parser
