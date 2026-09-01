"""Single entrypoint for the PatchHead workflow.

Examples::
    python patchhead/cli.py fetch --output-dir data/matched_refactored
    python patchhead/cli.py train --manifest-dir data/matched_refactored --out patchhead/checkpoints/model.pt
    python patchhead/cli.py eval --manifest data/matched_refactored/wildfake_benchmark.csv --ckpt patchhead/checkpoints/model.pt
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str) -> Path:
    filename = {"eval": "evaluate.py", "train": "train.py"}[name]
    return Path(__file__).resolve().with_name(filename)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"train", "eval"}:
        command = [sys.executable, str(_script(sys.argv[1])), *sys.argv[2:]]
        raise SystemExit(subprocess.call(command, cwd=ROOT))
    parser = argparse.ArgumentParser(description="PatchHead fetch/train/eval workflow")
    subs = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subs.add_parser("fetch")
    fetch_parser.add_argument("--output-dir", type=Path, default=Path("data/matched_refactored"))
    fetch_parser.add_argument("--base-quota", type=int, default=1250)
    fetch_parser.add_argument("--wildfake-quota", type=int, default=1000)
    fetch_parser.add_argument("--benchmark-count", type=int, default=500)
    fetch_parser.add_argument("--sid-count", type=int, default=200)
    fetch_parser.add_argument("--seed", type=int, default=42)
    fetch_parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.command == "fetch":
        from fetch import fetch
        fetch(args)
        return
if __name__ == "__main__":
    main()
