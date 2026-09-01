"""Fetch large provider subsets through the existing provider clients.

This is deliberately an orchestration layer: it does not alter Physics or
PatchHead.  The provider implementation remains cached and reproducible while
the harness owns large-run and quick-run manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .common import fingerprint, load_manifest, select_per_class, write_manifest


def _run_provider_fetch(args: argparse.Namespace) -> None:
    if args.sid_per_class != args.base_per_class:
        raise ValueError(
            "the delegated provider fetcher uses one base quota for SID, HEMG, and CIFAKE; "
            "--sid-per-class must equal --base-per-class"
        )
    command = [
        sys.executable, "patchhead/cli.py", "fetch",
        "--output-dir", str(args.output_dir),
        "--base-quota", str(args.base_per_class),
        "--wildfake-quota", str(args.wildfake_per_source),
        "--benchmark-count", str(args.benchmark_count),
        "--sid-count", str(args.sid_eval_count),
        "--seed", str(args.seed),
    ]
    if args.refresh:
        command.append("--refresh")
    env = os.environ.copy()
    env.setdefault("TMPDIR", str(Path.home() / "tmp"))
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    subprocess.run(command, check=True, env=env)


def _make_quick(args: argparse.Namespace) -> dict:
    source = args.output_dir
    quick = args.quick_output_dir or source.parent / f"{source.name}_quick"
    quick.mkdir(parents=True, exist_ok=True)
    split_names = ("train", "validation", "calibration", "matched_test")
    report = {"per_class": args.quick_per_class, "seed": args.seed, "splits": {}}
    for name in split_names:
        path = source / f"{name}.csv"
        records = load_manifest(path)
        split_seed = args.seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
        selected = select_per_class(records, args.quick_per_class, split_seed)
        output_name = "test.csv" if name == "matched_test" else f"{name}.csv"
        write_manifest(quick / output_name, selected)
        report["splits"][output_name] = {"count": len(selected), "fingerprint": fingerprint(selected)}
    for name in ("wildfake_benchmark", "sid_eval_200"):
        path = source / f"{name}.csv"
        if path.is_file():
            records = load_manifest(path)
            selected = []
            underfilled = {}
            for label in sorted({record.label for record in records}):
                candidates = [record for record in records if record.label == label]
                candidates.sort(key=lambda record: record.group_id or record.image_path)
                if name == "wildfake_benchmark" and len(candidates) < args.quick_per_class:
                    underfilled[str(label)] = len(candidates)
                selected.extend(candidates[:args.quick_per_class])
            if underfilled and not args.allow_underfilled_benchmark:
                raise ValueError(f"{name} quick subset is underfilled: {underfilled}")
            write_manifest(quick / f"{name}.csv", selected)
            report["splits"][f"{name}.csv"] = {"count": len(selected), "fingerprint": fingerprint(selected), "underfilled": underfilled}
    (quick / "quick_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def fetch(args: argparse.Namespace) -> None:
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_provider_fetch(args)
    quick_report = _make_quick(args)
    print(json.dumps({"large_output": str(args.output_dir), "quick": quick_report}, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/harness_large"))
    parser.add_argument("--quick-output-dir", type=Path)
    parser.add_argument("--base-per-class", type=int, default=3000)
    parser.add_argument("--sid-per-class", type=int, default=3000,
                        help="Reserved for provider-specific fetchers; current provider uses base quota for SID labels.")
    parser.add_argument("--wildfake-per-source", type=int, default=3000)
    parser.add_argument("--benchmark-count", type=int, default=500)
    parser.add_argument("--sid-eval-count", type=int, default=200)
    parser.add_argument("--quick-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--allow-underfilled-benchmark", action="store_true")
    return parser


def fetch_parser(parser: argparse.ArgumentParser) -> None:
    """Attach fetch arguments to the top-level harness CLI."""
    parser.add_argument("--output-dir", type=Path, default=Path("data/harness_large"))
    parser.add_argument("--quick-output-dir", type=Path)
    parser.add_argument("--base-per-class", type=int, default=3000)
    parser.add_argument("--sid-per-class", type=int, default=3000)
    parser.add_argument("--wildfake-per-source", type=int, default=3000)
    parser.add_argument("--benchmark-count", type=int, default=500)
    parser.add_argument("--sid-eval-count", type=int, default=200)
    parser.add_argument("--quick-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--allow-underfilled-benchmark", action="store_true")


if __name__ == "__main__":
    fetch(build_parser().parse_args())
