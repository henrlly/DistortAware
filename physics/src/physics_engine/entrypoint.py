"""Independent single-image and batch entry point for Physics evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

from .automatic_config import AutomaticProposalConfig
from .engine import PhysicsEngine, PhysicsEngineConfig


def run(
    input_path: str | Path,
    *,
    auto_proposals: bool = False,
) -> dict[str, Any]:
    """Return the common detector result for one image.

    Physics is evidence rather than a trained AIGC classifier, so its score is
    a physical violation score and its decision remains ``None``.
    """
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Physics entry point expects one image file: {path}")
    config = PhysicsEngineConfig(
        automatic=AutomaticProposalConfig(enabled=auto_proposals),
    )
    batch = PhysicsEngine(config).run(path)
    result = batch.to_dict()["images"][0]
    physics = result["physics"]
    return {
        "method": "physics",
        "image_path": str(path),
        "score": physics["violation_score"],
        "score_kind": physics["score_kind"],
        "confidence": physics["confidence"],
        "threshold": None,
        "decision": None,
        "details": {
            "status": physics["status"],
            "summary": physics["summary"],
            "applicable_cues": physics["applicable_cues"],
            "cues": result["cues"],
            "errors": result["errors"],
        },
    }


def run_batch(input_path: str | Path, *, auto_proposals: bool = False) -> list[dict[str, Any]]:
    path = Path(input_path).expanduser().resolve()
    config = PhysicsEngineConfig(automatic=AutomaticProposalConfig(enabled=auto_proposals))
    batch = PhysicsEngine(config).run(path, recursive=False)
    records = []
    for result in batch.to_dict()["images"]:
        physics = result["physics"]
        records.append({
            "method": "physics", "image_path": result["image_path"],
            "score": physics["violation_score"], "score_kind": physics["score_kind"],
            "confidence": physics["confidence"], "threshold": None, "decision": None,
            "details": {"status": physics["status"], "summary": physics["summary"],
                        "applicable_cues": physics["applicable_cues"], "cues": result["cues"]},
            "errors": result["errors"],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--auto-proposals", action="store_true")
    args = parser.parse_args()
    records = run_batch(args.image_dir, auto_proposals=args.auto_proposals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Physics evaluation complete: {len(records)} images -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
