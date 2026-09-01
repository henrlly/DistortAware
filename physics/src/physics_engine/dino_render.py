"""Presentation overlays for DINO patch evidence and physics residual geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from .dino_integration import (
    DinoIntegrationError,
    _match_physics,
    _physics_indices,
)
from .integration import _canonical_path
from .spatial import SpatialEvidenceError, segment_from_evidence, validate_score_grid


CUE_COLORS = {
    "perspective": (255, 70, 180),
    "cast_shadow": (255, 90, 50),
    "reflection": (40, 230, 255),
}


class DinoRenderError(RuntimeError):
    """Raised when a DINO/physics presentation overlay cannot be rendered."""


def _heatmap_rgb(grid: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    source = Image.fromarray(np.round(grid * 255).astype(np.uint8), mode="L")
    resized = np.asarray(source.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    # Compact blue -> yellow -> red map with no matplotlib dependency.
    red = np.clip(2.0 * resized, 0.0, 1.0)
    green = np.clip(1.0 - 2.0 * np.abs(resized - 0.5), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - resized), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    alpha = 0.16 + 0.52 * resized
    return np.round(rgb * 255).astype(np.uint8), alpha[..., None]


def _draw_physics_outliers(panel: Image.Image, physics_image: dict[str, Any]) -> int:
    draw = ImageDraw.Draw(panel)
    width, height = panel.size
    source_width = float(physics_image.get("width") or width)
    source_height = float(physics_image.get("height") or height)
    scale_x = width / source_width
    scale_y = height / source_height
    line_width = max(2, int(round(math.hypot(width, height) / 420)))
    drawn = 0
    cues = physics_image.get("cues", {})
    if not isinstance(cues, dict):
        return 0
    for cue_name, cue in cues.items():
        if not isinstance(cue, dict) or cue.get("status") == "consistent":
            continue
        color = CUE_COLORS.get(str(cue_name), (255, 255, 255))
        evidence = cue.get("evidence", [])
        if not isinstance(evidence, list):
            continue
        for item in evidence:
            if not isinstance(item, dict) or item.get("role") != "outlier":
                continue
            segment = segment_from_evidence(item)
            if segment is None:
                continue
            x1, y1, x2, y2 = segment
            points = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            draw.line(points, fill=color, width=line_width)
            radius = line_width + 1
            for x, y in ((points[0], points[1]), (points[2], points[3])):
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=line_width)
            drawn += 1
    return drawn


def _title_panel(panel: Image.Image, title: str) -> Image.Image:
    band = 34
    output = Image.new("RGB", (panel.width, panel.height + band), (20, 23, 30))
    output.paste(panel, (0, band))
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default(size=15)
    draw.text((12, 9), title, fill=(245, 247, 250), font=font)
    return output


def render_dino_physics_panel(
    image_path: str | Path,
    dino_image: dict[str, Any],
    physics_image: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Render original, DINO heatmap, and heatmap-plus-physics panels."""

    patch = dino_image.get("patch_evidence")
    if not isinstance(patch, dict) or "values" not in patch:
        raise DinoRenderError("DINO record has no patch evidence map")
    if patch.get("coordinate_space") != "normalized_full_frame":
        raise DinoRenderError("Only normalized_full_frame patch maps can be rendered safely")
    try:
        grid = validate_score_grid(patch["values"])
    except SpatialEvidenceError as exc:
        raise DinoRenderError(str(exc)) from exc
    path = Path(image_path).expanduser().resolve()
    try:
        with Image.open(path) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise DinoRenderError(f"Could not decode {path}: {exc}") from exc
    heat_rgb, alpha = _heatmap_rgb(grid, original.size)
    original_array = np.asarray(original, dtype=np.float32)
    heat_panel = Image.fromarray(
        np.round(original_array * (1.0 - alpha) + heat_rgb.astype(np.float32) * alpha).astype(np.uint8)
    )
    combined = heat_panel.copy()
    outlier_segments = _draw_physics_outliers(combined, physics_image)

    panels = [
        _title_panel(original, "Original"),
        _title_panel(heat_panel, "DINO patch evidence (blue low, red high)"),
        _title_panel(combined, "DINO + physics residual geometry"),
    ]
    gutter = 8
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels) + gutter * (len(panels) - 1), panels[0].height),
        (20, 23, 30),
    )
    offset = 0
    for panel in panels:
        canvas.paste(panel, (offset, 0))
        offset += panel.width + gutter
    destination = Path(output_path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")
    return {
        "output_path": str(destination),
        "grid_shape": list(grid.shape),
        "physics_outlier_segments_drawn": outlier_segments,
        "warning": "Visualization shows spatial association, not causal DINO attribution.",
    }


def _safe_name(path: str) -> str:
    stem = Path(path).stem
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "image"
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:64]}__{digest}.png"


def render_payload(
    dino_payload: dict[str, Any],
    physics_payload: dict[str, Any],
    *,
    output_dir: str | Path,
    path_root: str | Path = ".",
) -> list[dict[str, Any]]:
    if not isinstance(dino_payload.get("images"), list):
        raise DinoRenderError("DINO export must contain an `images` array")
    if not isinstance(physics_payload.get("images"), list):
        raise DinoRenderError("Physics results must contain an `images` array")
    base = Path(path_root).expanduser().resolve(strict=False)
    raw_dino_root = Path(str(dino_payload.get("input_root", "."))).expanduser()
    if not raw_dino_root.is_absolute():
        raw_dino_root = base / raw_dino_root
    dino_root = raw_dino_root.resolve(strict=False)
    raw_physics_root = Path(str(physics_payload.get("input_root", "."))).expanduser()
    if not raw_physics_root.is_absolute():
        raw_physics_root = base / raw_physics_root
    physics_root = raw_physics_root.resolve(strict=False)
    by_path, by_basename = _physics_indices(physics_payload["images"], physics_root)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    reports: list[dict[str, Any]] = []
    for record in dino_payload["images"]:
        if not isinstance(record, dict) or not isinstance(record.get("image_path"), str):
            raise DinoRenderError("Every DINO record must have a string `image_path`")
        image_path = record["image_path"]
        physics_image, _method = _match_physics(
            image_path,
            detector_root=dino_root,
            physics_root=physics_root,
            by_path=by_path,
            by_basename=by_basename,
        )
        if physics_image is None:
            raise DinoRenderError(f"No unambiguous physics result matched {image_path!r}")
        canonical_image = _canonical_path(image_path, dino_root)
        reports.append(
            render_dino_physics_panel(
                canonical_image,
                record,
                physics_image,
                destination / _safe_name(image_path),
            )
        )
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physics-dino-render",
        description="Render side-by-side DINO patch evidence and physics residual geometry.",
    )
    parser.add_argument("--dino-results", required=True)
    parser.add_argument("--physics-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--path-root", default=".")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with Path(args.dino_results).open("r", encoding="utf-8") as handle:
            dino = json.load(handle)
        with Path(args.physics_results).open("r", encoding="utf-8") as handle:
            physics = json.load(handle)
        reports = render_payload(
            dino,
            physics,
            output_dir=args.output_dir,
            path_root=args.path_root,
        )
    except (OSError, json.JSONDecodeError, DinoIntegrationError, DinoRenderError) as exc:
        print(f"physics-dino-render: {exc}", file=sys.stderr)
        return 2
    print(f"Rendered {len(reports)} DINO/physics panel(s) in {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
