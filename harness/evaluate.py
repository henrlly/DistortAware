"""Run all configured models and write one comparable result set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import run_did, run_filter, run_patchhead, run_physics
from .common import TRANSFORMS, fingerprint, load_manifest
from .metrics import summarize
from .reports import write_reports


def evaluate(args: argparse.Namespace) -> None:
    root = Path(args.data_dir).expanduser().resolve()
    manifest_name = args.manifest or ("test.csv" if (root / "test.csv").is_file() else "matched_test.csv")
    records = load_manifest(root / manifest_name)
    work = Path(args.output_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    transforms = tuple(args.transforms.split(","))
    all_results = []
    repo = Path(__file__).resolve().parent.parent
    models = set(args.models.split(","))
    supported = {"physics", "patchhead_baseline", "patchhead_distortion_aware", "filter", "did"}
    unknown = models - supported
    if unknown:
        raise SystemExit(f"unknown harness models: {sorted(unknown)}")
    for transform in transforms:
        if "physics" in models:
            all_results.extend(run_physics(repo, records, transform, work, args.auto_proposals))
        if "patchhead_baseline" in models:
            if not args.baseline_checkpoint:
                raise SystemExit("--baseline-checkpoint is required for patchhead_baseline")
            all_results.extend(run_patchhead(repo, records, transform, work, Path(args.baseline_checkpoint), False))
        if "patchhead_distortion_aware" in models:
            if not args.aware_checkpoint:
                raise SystemExit("--aware-checkpoint is required for patchhead_distortion_aware")
            all_results.extend(run_patchhead(repo, records, transform, work, Path(args.aware_checkpoint), True))
        if "filter" in models:
            if not args.filter_checkpoint:
                raise SystemExit("--filter-checkpoint is required for filter")
            all_results.extend(run_filter(repo, records, transform, work,
                                          Path(args.filter_checkpoint)))
        if "did" in models:
            if not args.did_checkpoint:
                raise SystemExit("--did-checkpoint is required for did")
            all_results.extend(run_did(repo, records, transform, work,
                                       Path(args.did_checkpoint), args.did_reconstructor,
                                       args.did_resolution, args.did_steps, args.did_batch_size))
    with (work / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_results:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    report = {"data_dir": str(root), "manifest": manifest_name,
              "manifest_fingerprint": fingerprint(records), "models": sorted(models),
              "transforms": transforms, "records": len(all_results),
              "coverage": _coverage(all_results, len(records), models, transforms),
              "metrics": summarize(all_results)}
    (work / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_reports(work, all_results, report)
    if any(item["missing"] or item["duplicates"] for item in report["coverage"].values()):
        raise RuntimeError(f"incomplete harness evaluation coverage: {report['coverage']}")
    print(json.dumps(report, indent=2, sort_keys=True))


def _coverage(records, expected_per_group, models, transforms):
    result = {}
    for model in sorted(models):
        for transform in transforms:
            group = [r for r in records if r["model"] == model and r["transform"] == transform]
            ids = [r["image_id"] for r in group]
            missing = sum(bool(r.get("missing")) for r in group)
            result[f"{model}:{transform}"] = {
                "expected": expected_per_group,
                "returned": len(group) - missing,
                "missing": missing + max(expected_per_group - len(group), 0),
                "duplicates": len(ids) - len(set(ids)),
                "errors": sum(bool(r.get("errors")) for r in group),
            }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--models", default="physics,patchhead_baseline,patchhead_distortion_aware")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--aware-checkpoint")
    parser.add_argument("--filter-checkpoint")
    parser.add_argument("--did-checkpoint")
    parser.add_argument("--did-reconstructor", default="sd15")
    parser.add_argument("--did-resolution", type=int, default=256)
    parser.add_argument("--did-steps", type=int, default=10)
    parser.add_argument("--did-batch-size", type=int, default=32)
    parser.add_argument("--transforms", default=",".join(TRANSFORMS))
    parser.add_argument("--output-dir", type=Path, default=Path("results/harness/current"))
    parser.add_argument("--auto-proposals", action="store_true")
    return parser
