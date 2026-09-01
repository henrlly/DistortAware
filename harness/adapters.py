"""Subprocess adapters for the current Physics and PatchHead entrypoints."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import Record, materialize_view


def _run(command: list[str], env: dict[str, str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _record_index(records: list[Record]) -> dict[str, Record]:
    index = {}
    for record in records:
        index[Path(record.image_path).name] = record
    return index


def _missing_result(model: str, transform: str, record: Record, output: Path) -> dict[str, Any]:
    return {"model": model, "transform": transform, "image_id": record.group_id,
            "image_path": record.image_path, "source": record.source,
            "category": record.category, "label": record.label, "score": None,
            "score_kind": None, "confidence": None, "threshold": None,
            "decision": None, "errors": ["model produced no result for manifest image"],
            "missing": True,
            "native_output": str(output)}


def _run_entrypoint(repo: Path, module: str, view: Path, output: Path,
                    arguments: list[str], python_path: Path | None = None) -> list[dict[str, Any]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", module, "--image-dir", str(view),
               "--output", str(output), *arguments]
    env = os.environ.copy()
    if python_path is not None:
        env["PYTHONPATH"] = str(python_path) + os.pathsep + env.get("PYTHONPATH", "")
    _run(command, env, repo)
    return json.loads(output.read_text(encoding="utf-8"))


def _normalize(repo: Path, records: list[Record], transform: str, work_dir: Path,
               model: str, module: str, arguments: list[str],
               python_path: Path | None = None) -> list[dict[str, Any]]:
    view_records = materialize_view(records, work_dir / "views" / model / transform,
                                    transform, 42)
    view = Path(view_records[0].image_path).parent
    output = work_dir / "native" / f"{model}_{transform}.json"
    payload = _run_entrypoint(repo, module, view, output, arguments, python_path)
    index = _record_index(view_records)
    result = []
    matched = set()
    for raw in payload:
        source = index.get(Path(raw.get("image_path", "")).name)
        if source is None:
            continue
        matched.add(source.group_id)
        errors = raw.get("errors", [])
        if isinstance(errors, str):
            errors = [errors]
        details = raw.get("details", {})
        normalized = {"model": model,
                       "transform": transform, "image_id": source.group_id,
                       "image_path": source.image_path, "source": source.source,
                       "category": source.category, "label": source.label,
                       "score": raw.get("score"), "confidence": raw.get("confidence"),
                       "threshold": raw.get("threshold"), "score_kind": raw.get("score_kind"),
                       "decision": raw.get("decision"), "missing": False,
                       "details": details, "errors": errors, "native_output": str(output)}
        if model == "physics":
            normalized["physics"] = {"status": details.get("status"),
                                     "violation_score": raw.get("score"),
                                     "score_kind": raw.get("score_kind"),
                                     "confidence": raw.get("confidence")}
            normalized["cues"] = details.get("cues", {})
        elif model.startswith("patchhead"):
            normalized["patchhead"] = details
        elif model == "filter":
            normalized["filter"] = details
        elif model == "did":
            normalized["did"] = details
        result.append(normalized)
    result.extend(_missing_result(model,
                                  transform, record, output)
                  for record in records if record.group_id not in matched)
    return result


def run_physics(repo: Path, records: list[Record], transform: str, work_dir: Path,
                auto_proposals: bool = False) -> list[dict[str, Any]]:
    arguments = ["--auto-proposals"] if auto_proposals else []
    return _normalize(repo, records, transform, work_dir, "physics",
                      "physics_engine.entrypoint", arguments, repo / "physics" / "src")


def run_patchhead(repo: Path, records: list[Record], transform: str, work_dir: Path,
                  checkpoint: Path, aware: bool) -> list[dict[str, Any]]:
    arguments = ["--checkpoint", str(checkpoint)]
    if aware:
        arguments.append("--distortion-aware")
    model = "patchhead_distortion_aware" if aware else "patchhead_baseline"
    return _normalize(repo, records, transform, work_dir, model,
                      "patchhead.entrypoint", arguments)


def run_filter(repo: Path, records: list[Record], transform: str, work_dir: Path,
               checkpoint: Path) -> list[dict[str, Any]]:
    return _normalize(repo, records, transform, work_dir, "filter",
                      "filter_based_approach.entrypoint",
                      ["--checkpoint", str(checkpoint)])


def run_did(repo: Path, records: list[Record], transform: str, work_dir: Path,
            checkpoint: Path, reconstructor: str, resolution: int,
            steps: int, batch_size: int) -> list[dict[str, Any]]:
    return _normalize(repo, records, transform, work_dir, "did", "did.entrypoint", [
        "--checkpoint", str(checkpoint), "--reconstructor", reconstructor,
        "--resolution", str(resolution), "--steps", str(steps),
        "--batch-size", str(batch_size),
    ])
