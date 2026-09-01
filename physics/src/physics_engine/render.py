"""Pillow-based forensic overlays for auditable cue results."""

from __future__ import annotations

from math import atan2, cos, pi, sin
from pathlib import Path

from PIL import Image, ImageDraw

from .schema import CueResult


BUNDLE_COLORS = [
    (52, 211, 153),
    (59, 130, 246),
    (250, 204, 21),
    (168, 85, 247),
]
OUTLIER_COLOR = (239, 68, 68)
REGION_COLOR = (244, 114, 182)
SHADOW_REGION_COLOR = (52, 211, 153)
MIRROR_REGION_COLOR = (59, 130, 246)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: tuple[int, int, int],
    width: int = 3,
) -> None:
    draw.line([start, end], fill=color, width=width)
    angle = atan2(end[1] - start[1], end[0] - start[0])
    arrow_length = 11.0
    for offset in (5.0 * pi / 6.0, -5.0 * pi / 6.0):
        tip = (
            end[0] + arrow_length * cos(angle + offset),
            end[1] + arrow_length * sin(angle + offset),
        )
        draw.line([end, tip], fill=color, width=width)


def _draw_header(image: Image.Image, cue: CueResult) -> None:
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    color = {
        "consistent": (16, 120, 78, 220),
        "inconsistent": (185, 28, 28, 220),
        "indeterminate": (161, 98, 7, 220),
        "not_applicable": (55, 65, 81, 220),
        "error": (127, 29, 29, 220),
    }[cue.status]
    text = f"{cue.cue}: {cue.status}"
    if cue.violation_score is not None:
        text += f"  violation={cue.violation_score:.2f}  confidence={cue.confidence:.2f}"
    draw.rectangle([0, 0, image.width, 28], fill=color)
    draw.text((8, 7), text, fill=(255, 255, 255, 255))
    image.alpha_composite(layer)


def _draw_proposal_regions(draw: ImageDraw.ImageDraw, cue: CueResult) -> None:
    for item in cue.evidence:
        kind = item.get("kind")
        if kind not in {"shadow_region", "mirror_region"}:
            continue
        color = SHADOW_REGION_COLOR if kind == "shadow_region" else MIRROR_REGION_COLOR
        contour = item.get("contour")
        if isinstance(contour, list) and len(contour) >= 3:
            try:
                points = [(float(point[0]), float(point[1])) for point in contour]
            except (TypeError, ValueError, IndexError):
                points = []
            if points:
                draw.line(points + [points[0]], fill=color, width=3)
                continue
        bbox = item.get("bbox_xyxy")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            draw.rectangle(tuple(float(value) for value in bbox), outline=color, width=3)


def render_cue_overlay(image: Image.Image, cue: CueResult, output_path: str | Path) -> Path:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    _draw_proposal_regions(draw, cue)

    if cue.cue == "perspective":
        for region in cue.measurements.get("reviewed_regions_xyxy", []):
            if isinstance(region, (list, tuple)) and len(region) == 4:
                draw.rectangle(region, outline=REGION_COLOR, width=3)
        for item in cue.evidence:
            if item.get("kind") != "line_segment":
                continue
            x1, y1, x2, y2 = item["xyxy"]
            bundle_index = int(item.get("bundle_index", -1))
            color = (
                OUTLIER_COLOR
                if bundle_index < 0
                else BUNDLE_COLORS[bundle_index % len(BUNDLE_COLORS)]
            )
            draw.line([(x1, y1), (x2, y2)], fill=color, width=2)

        for payload in cue.measurements.get("vanishing_points", []):
            point = payload.get("vanishing_point", {})
            if point.get("kind") != "finite":
                continue
            x, y = point.get("xy", [None, None])
            if x is None or y is None:
                continue
            if -20 <= x <= canvas.width + 20 and -20 <= y <= canvas.height + 20:
                color = BUNDLE_COLORS[int(payload["bundle_index"]) % len(BUNDLE_COLORS)]
                draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=color, width=3)

    elif cue.cue == "cast_shadow":
        for item in cue.evidence:
            if item.get("kind") != "shadow_vector":
                continue
            start = tuple(item["object_contact"])
            end = tuple(item["shadow_tip"])
            color = OUTLIER_COLOR if item.get("role") == "outlier" else BUNDLE_COLORS[0]
            _draw_arrow(draw, start, end, color=color)
        point = cue.measurements.get("estimated_projected_light")
        if isinstance(point, dict) and point.get("kind") == "finite":
            x, y = point.get("xy", [None, None])
            if x is not None and y is not None and -20 <= x <= canvas.width + 20 and -20 <= y <= canvas.height + 20:
                draw.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(255, 255, 255), width=3)

    elif cue.cue == "reflection":
        for item in cue.evidence:
            if item.get("kind") != "reflection_connector":
                continue
            start = tuple(item["object_point"])
            end = tuple(item["reflection_point"])
            color = OUTLIER_COLOR if item.get("role") == "outlier" else BUNDLE_COLORS[1]
            draw.line([start, end], fill=color, width=3)
            radius = 4
            draw.ellipse(
                [start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius],
                fill=color,
            )
            draw.ellipse(
                [end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius],
                outline=color,
                width=2,
            )

    _draw_header(canvas, cue)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(path, format="PNG")
    return path
