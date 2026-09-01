"""Serializable result objects shared by every physics cue."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


VALID_STATUSES = {
    "consistent",
    "inconsistent",
    "indeterminate",
    "not_applicable",
    "error",
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class CueResult:
    cue: str
    applicable: bool
    status: str
    violation_score: float | None
    confidence: float
    summary: str
    assumptions: list[str] = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    overlay_path: str | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Unknown cue status: {self.status}")
        self.confidence = _clamp01(self.confidence)
        if self.violation_score is not None:
            self.violation_score = _clamp01(self.violation_score)
        if not self.applicable and self.violation_score is not None:
            raise ValueError("An inapplicable cue cannot have a violation score")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AggregatePhysicsResult:
    score_kind: str
    status: str
    violation_score: float | None
    confidence: float
    applicable_cues: list[str]
    summary: str

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Unknown aggregate status: {self.status}")
        self.confidence = _clamp01(self.confidence)
        if self.violation_score is not None:
            self.violation_score = _clamp01(self.violation_score)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ImageResult:
    image_path: str
    width: int
    height: int
    physics: AggregatePhysicsResult
    cues: dict[str, CueResult]
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "physics": self.physics.to_dict(),
            "cues": {key: cue.to_dict() for key, cue in self.cues.items()},
            "errors": list(self.errors),
        }


@dataclass(slots=True)
class BatchResult:
    schema_version: str
    engine_version: str
    generated_at: str
    input_root: str
    images: list[ImageResult]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "generated_at": self.generated_at,
            "input_root": self.input_root,
            "images": [image.to_dict() for image in self.images],
            "summary": self.summary,
        }


def not_applicable(cue: str, summary: str, *, limitations: list[str] | None = None) -> CueResult:
    return CueResult(
        cue=cue,
        applicable=False,
        status="not_applicable",
        violation_score=None,
        confidence=0.0,
        summary=summary,
        limitations=limitations or [],
    )


def error_result(cue: str, message: str) -> CueResult:
    return CueResult(
        cue=cue,
        applicable=False,
        status="error",
        violation_score=None,
        confidence=0.0,
        summary=message,
        limitations=["The cue failed during processing and did not affect the aggregate."],
    )
