"""Launch the local extension demo from hash-verified detector artifacts.

The launcher never substitutes the wiring fixture for a detector. PatchHead is
always started from its sealed pooled checkpoint. PrismGuard is started only
when its complete bundle/checkpoint/license/root tuple is supplied; otherwise
the launcher reports that profile as unavailable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
PATCHHEAD_SHA256 = "828fac3ba5c5b814a1ada36477b36848ab4c8366e040e68fff9f4c9fe14b6989"
DINOV3_L_SHA256 = "45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941"


@dataclass(frozen=True, slots=True)
class DemoConfig:
    checkpoint: Path
    checkpoint_sha256: str
    backbone_safetensors: Path
    backbone_sha256: str
    hf_home: Path
    patchhead_port: int = 8765
    prismguard_port: int = 8766
    fixture_port: int = 8767
    wall_port: int = 8088
    patchhead_token: str = ""
    prismguard_token: str = ""
    fixture_token: str = ""
    prismguard_bundle: Path | None = None
    prismguard_checkpoint: Path | None = None
    prismguard_license_ledger: Path | None = None
    prismguard_root: Path | None = None

    @property
    def prismguard_enabled(self) -> bool:
        return self.prismguard_bundle is not None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a file: {resolved}")
    return resolved


def _require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    return resolved


def _require_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return normalized


def validate_config(config: DemoConfig) -> DemoConfig:
    ports = (
        config.patchhead_port,
        config.prismguard_port,
        config.fixture_port,
        config.wall_port,
    )
    if any(not 1 <= port <= 65535 for port in ports):
        raise ValueError("all ports must lie within [1, 65535]")
    if len(set(ports)) != len(ports):
        raise ValueError("PatchHead, PrismGuard, fixture, and wall ports must be distinct")

    checkpoint = _require_file(config.checkpoint, "PatchHead checkpoint")
    backbone = _require_file(config.backbone_safetensors, "DINOv3-L safetensors")
    hf_home = _require_dir(config.hf_home, "HF_HOME")
    expected_checkpoint = _require_sha256(
        config.checkpoint_sha256, "PatchHead checkpoint SHA-256"
    )
    expected_backbone = _require_sha256(
        config.backbone_sha256, "DINOv3-L SHA-256"
    )
    if sha256_file(checkpoint) != expected_checkpoint:
        raise ValueError("PatchHead checkpoint SHA-256 mismatch")
    if sha256_file(backbone) != expected_backbone:
        raise ValueError("DINOv3-L safetensors SHA-256 mismatch")

    prismguard_values = (
        config.prismguard_bundle,
        config.prismguard_checkpoint,
        config.prismguard_license_ledger,
        config.prismguard_root,
    )
    supplied = sum(value is not None for value in prismguard_values)
    if supplied not in {0, len(prismguard_values)}:
        raise ValueError(
            "PrismGuard requires bundle, checkpoint, license ledger, and repository root together"
        )
    if supplied:
        bundle = _require_file(config.prismguard_bundle, "PrismGuard bundle")  # type: ignore[arg-type]
        prismguard_checkpoint = _require_file(
            config.prismguard_checkpoint, "PrismGuard checkpoint"  # type: ignore[arg-type]
        )
        ledger = _require_file(
            config.prismguard_license_ledger, "PrismGuard license ledger"  # type: ignore[arg-type]
        )
        prismguard_root = _require_dir(
            config.prismguard_root, "PrismGuard repository root"  # type: ignore[arg-type]
        )
    else:
        bundle = prismguard_checkpoint = ledger = prismguard_root = None

    return DemoConfig(
        checkpoint=checkpoint,
        checkpoint_sha256=expected_checkpoint,
        backbone_safetensors=backbone,
        backbone_sha256=expected_backbone,
        hf_home=hf_home,
        patchhead_port=config.patchhead_port,
        prismguard_port=config.prismguard_port,
        fixture_port=config.fixture_port,
        wall_port=config.wall_port,
        patchhead_token=config.patchhead_token or secrets.token_urlsafe(24),
        prismguard_token=config.prismguard_token or secrets.token_urlsafe(24),
        fixture_token=config.fixture_token or secrets.token_urlsafe(24),
        prismguard_bundle=bundle,
        prismguard_checkpoint=prismguard_checkpoint,
        prismguard_license_ledger=ledger,
        prismguard_root=prismguard_root,
    )


def service_commands(config: DemoConfig) -> list[tuple[str, list[str]]]:
    python = sys.executable
    common = ["--physics-profile", "off", "--artifact-profile", "off"]
    commands: list[tuple[str, list[str]]] = [
        (
            "PatchHead",
            [
                python,
                "-m",
                "browser_product.service",
                "--checkpoint",
                str(config.checkpoint),
                "--port",
                str(config.patchhead_port),
                "--api-token",
                config.patchhead_token,
                "--device",
                "cpu",
                *common,
            ],
        ),
        (
            "Wiring fixture",
            [
                python,
                "-m",
                "browser_product.service",
                "--demo-fixture",
                "--port",
                str(config.fixture_port),
                "--api-token",
                config.fixture_token,
                *common,
            ],
        ),
    ]
    if config.prismguard_enabled:
        commands.append(
            (
                "PrismGuard",
                [
                    python,
                    "-m",
                    "browser_product.service",
                    "--prismguard-bundle",
                    str(config.prismguard_bundle),
                    "--prismguard-checkpoint",
                    str(config.prismguard_checkpoint),
                    "--prismguard-license-ledger",
                    str(config.prismguard_license_ledger),
                    "--prismguard-root",
                    str(config.prismguard_root),
                    "--port",
                    str(config.prismguard_port),
                    "--api-token",
                    config.prismguard_token,
                    "--device",
                    "cpu",
                    *common,
                ],
            )
        )
    commands.append(
        (
            "Demo wall",
            [
                python,
                "-m",
                "http.server",
                str(config.wall_port),
                "--bind",
                "127.0.0.1",
                "--directory",
                str(ROOT),
            ],
        )
    )
    return commands


def _wait_for_health(port: int, token: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 180.0
    request = Request(
        f"http://127.0.0.1:{port}/v1/health",
        headers={"Authorization": f"Bearer {token}"},
    )
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service on port {port} exited with {process.returncode}")
        try:
            with urlopen(request, timeout=2.0) as response:  # noqa: S310 - loopback only
                payload = json.load(response)
            if response.status == 200 and payload.get("status") == "ready":
                return
        except (OSError, URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(0.5)
    raise TimeoutError(f"service on port {port} did not become ready")


def _terminate(processes: Iterable[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def run(config: DemoConfig) -> int:
    environment = os.environ.copy()
    environment.update({"HF_HOME": str(config.hf_home), "HF_HUB_OFFLINE": "1"})
    processes: list[subprocess.Popen[bytes]] = []
    commands = service_commands(config)
    try:
        for _label, command in commands:
            processes.append(subprocess.Popen(command, cwd=ROOT, env=environment))
        service_specs = [
            (config.patchhead_port, config.patchhead_token, processes[0]),
            (config.fixture_port, config.fixture_token, processes[1]),
        ]
        if config.prismguard_enabled:
            service_specs.append(
                (config.prismguard_port, config.prismguard_token, processes[2])
            )
        for port, token, process in service_specs:
            _wait_for_health(port, token, process)

        print("Local detector demo ready")
        print(f"Demo wall: http://127.0.0.1:{config.wall_port}/browser_extension/demo/")
        print(
            f"PatchHead: http://127.0.0.1:{config.patchhead_port} token={config.patchhead_token}"
        )
        print(
            f"Wiring fixture: http://127.0.0.1:{config.fixture_port} token={config.fixture_token}"
        )
        if config.prismguard_enabled:
            print(
                f"PrismGuard: http://127.0.0.1:{config.prismguard_port} token={config.prismguard_token}"
            )
        else:
            print("PrismGuard: unavailable (no approved complete artifact tuple supplied)")
        print("Press Ctrl-C to stop all local demo processes")
        signal.pause()
    except KeyboardInterrupt:
        return 0
    finally:
        _terminate(processes)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", default=PATCHHEAD_SHA256)
    parser.add_argument("--backbone-safetensors", required=True, type=Path)
    parser.add_argument("--backbone-sha256", default=DINOV3_L_SHA256)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--patchhead-port", type=int, default=8765)
    parser.add_argument("--prismguard-port", type=int, default=8766)
    parser.add_argument("--fixture-port", type=int, default=8767)
    parser.add_argument("--wall-port", type=int, default=8088)
    parser.add_argument("--patchhead-token", default="")
    parser.add_argument("--prismguard-token", default="")
    parser.add_argument("--fixture-token", default="")
    parser.add_argument("--prismguard-bundle", type=Path)
    parser.add_argument("--prismguard-checkpoint", type=Path)
    parser.add_argument("--prismguard-license-ledger", type=Path)
    parser.add_argument("--prismguard-root", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify artifacts and print the process plan without starting services",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = validate_config(
            DemoConfig(
                checkpoint=args.checkpoint,
                checkpoint_sha256=args.checkpoint_sha256,
                backbone_safetensors=args.backbone_safetensors,
                backbone_sha256=args.backbone_sha256,
                hf_home=args.hf_home,
                patchhead_port=args.patchhead_port,
                prismguard_port=args.prismguard_port,
                fixture_port=args.fixture_port,
                wall_port=args.wall_port,
                patchhead_token=args.patchhead_token,
                prismguard_token=args.prismguard_token,
                fixture_token=args.fixture_token,
                prismguard_bundle=args.prismguard_bundle,
                prismguard_checkpoint=args.prismguard_checkpoint,
                prismguard_license_ledger=args.prismguard_license_ledger,
                prismguard_root=args.prismguard_root,
            )
        )
    except (OSError, ValueError) as exc:
        print(f"browser-demo: {exc}", file=sys.stderr)
        return 2
    if args.check_only:
        print("browser-demo: artifacts verified")
        for label, command in service_commands(config):
            redacted = ["<token>" if value in {
                config.patchhead_token,
                config.prismguard_token,
                config.fixture_token,
            } else value for value in command]
            print(f"{label}: {' '.join(redacted)}")
        return 0
    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
