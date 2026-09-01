"""Orchestration for the standalone physics-explanation engine."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from PIL import Image, ImageOps, UnidentifiedImageError

from .annotations import AnnotationStore, ImageAnnotations
from .automatic_config import AutomaticProposalConfig
from .perspective import PerspectiveConfig, analyze_perspective
from .reflection import ReflectionConfig, analyze_reflections
from .render import render_cue_overlay
from .schema import (
    AggregatePhysicsResult,
    BatchResult,
    CueResult,
    ImageResult,
    error_result,
    not_applicable,
    utc_now_iso,
)
from .shadow import ShadowConfig, analyze_cast_shadows

if TYPE_CHECKING:
    from .automatic import (
        AutomaticCueProposals,
        AutomaticProposalBundle,
        AutomaticProposalEngine,
    )


ENGINE_VERSION = "0.6.0"
SCHEMA_VERSION = "0.1.0"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(slots=True)
class PhysicsEngineConfig:
    perspective: PerspectiveConfig = field(default_factory=PerspectiveConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    automatic: AutomaticProposalConfig = field(default_factory=AutomaticProposalConfig)


def _aggregate(cues: dict[str, CueResult]) -> AggregatePhysicsResult:
    applicable = [
        cue
        for cue in cues.values()
        if cue.applicable and cue.violation_score is not None and cue.status != "error"
    ]
    if not applicable:
        return AggregatePhysicsResult(
            score_kind="physics_violation_not_aigc_probability",
            status="indeterminate",
            violation_score=None,
            confidence=0.0,
            applicable_cues=[],
            summary="No physics cue met its applicability requirements.",
        )

    denominator = sum(max(cue.confidence, 0.05) for cue in applicable)
    score = sum(
        cue.violation_score * max(cue.confidence, 0.05)  # type: ignore[operator]
        for cue in applicable
    ) / denominator
    confidence = sum(cue.confidence for cue in applicable) / len(applicable)
    if score >= 0.62:
        status = "inconsistent"
    elif score <= 0.34:
        status = "consistent"
    else:
        status = "indeterminate"

    return AggregatePhysicsResult(
        score_kind="physics_violation_not_aigc_probability",
        status=status,
        violation_score=score,
        confidence=confidence,
        applicable_cues=[cue.cue for cue in applicable],
        summary=(
            f"Aggregated {len(applicable)} applicable cue(s). This is physical-consistency "
            "evidence, not the probability that the image is AI-generated."
        ),
    )


def _iter_images(
    input_path: Path,
    recursive: bool,
    *,
    excluded_roots: tuple[Path, ...] = (),
) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        yield input_path
        return
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    for path in sorted(iterator):
        if any(path.is_relative_to(excluded) for excluded in excluded_roots):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _safe_overlay_stem(image_path: Path, input_root: Path) -> str:
    if input_root.is_dir():
        try:
            relative = image_path.relative_to(input_root)
        except ValueError:
            relative = Path(image_path.name)
    else:
        relative = Path(image_path.name)
    components = list(relative.with_suffix("").parts)
    return "__".join(component.replace(" ", "_") for component in components)


class PhysicsEngine:
    def __init__(
        self,
        config: PhysicsEngineConfig | None = None,
        *,
        proposal_engine: AutomaticProposalEngine | None = None,
    ) -> None:
        self.config = config or PhysicsEngineConfig()
        if proposal_engine is not None:
            self.proposal_engine: AutomaticProposalEngine | None = proposal_engine
        elif self.config.automatic.enabled:
            from .automatic import AutomaticProposalEngine

            self.proposal_engine = AutomaticProposalEngine(self.config.automatic)
        else:
            self.proposal_engine = None

    @staticmethod
    def _decorate_reviewed_result(result: CueResult) -> CueResult:
        result.measurements["evidence_origin"] = "reviewed"
        for item in result.evidence:
            item["evidence_origin"] = "reviewed"
        return result

    @staticmethod
    def _proposal_not_applicable(proposal: AutomaticCueProposals) -> CueResult:
        result = not_applicable(
            proposal.cue,
            proposal.reason,
            limitations=list(proposal.limitations),
        )
        result.confidence = proposal.confidence
        result.measurements = dict(proposal.measurements)
        result.measurements.update(
            {
                "evidence_origin": "automatic_proposal",
                "proposal_applicable": False,
                "proposal_confidence": proposal.confidence,
            }
        )
        if proposal.warnings:
            result.measurements["proposal_warnings"] = list(proposal.warnings)
        result.evidence = [
            dict(item)
            for item in proposal.evidence
            if str(item.get("kind", "")).endswith("_region")
        ]
        return result

    def _decorate_automatic_result(
        self, result: CueResult, proposal: AutomaticCueProposals
    ) -> CueResult:
        definitive_gate_passed = not (
            result.status == "inconsistent"
            and len(proposal.pairs)
            < self.config.automatic.min_pairs_for_definitive_inconsistency
        )
        if not definitive_gate_passed:
            result.status = "indeterminate"
            result.violation_score = min(result.violation_score or 1.0, 0.5)
            result.summary = (
                "Automatic correspondences suggest an inconsistency, but only "
                f"{len(proposal.pairs)} pair(s) survived; at least "
                f"{self.config.automatic.min_pairs_for_definitive_inconsistency} "
                "are required for a definitive automatic inconsistency."
            )
        result.confidence *= proposal.confidence
        geometry_pair_count = result.measurements.pop("reviewed_pair_count", None)
        if geometry_pair_count is not None:
            result.measurements["geometry_pair_count"] = geometry_pair_count
        result.measurements.update(proposal.measurements)
        result.measurements.update(
            {
                "evidence_origin": "automatic_proposal",
                "proposal_applicable": True,
                "proposal_confidence": proposal.confidence,
                "proposed_pair_count": len(proposal.pairs),
                "automatic_definitive_inconsistency_gate": {
                    "passed": definitive_gate_passed,
                    "pair_count": len(proposal.pairs),
                    "required_pairs": self.config.automatic.min_pairs_for_definitive_inconsistency,
                },
            }
        )
        if proposal.warnings:
            result.measurements["proposal_warnings"] = list(proposal.warnings)
        for limitation in proposal.limitations:
            if limitation not in result.limitations:
                result.limitations.append(limitation)

        pair_details = [
            item
            for item in proposal.evidence
            if item.get("kind")
            in {"shadow_pair_proposal", "reflection_pair_proposal"}
        ]
        for index, item in enumerate(result.evidence):
            item["evidence_origin"] = "automatic_proposal"
            if index < len(pair_details):
                detail = pair_details[index]
                for key in (
                    "association",
                    "object_label",
                    "feature_similarity",
                    "nearest_neighbour_margin",
                    "mutual_nearest_neighbour",
                    "component_id",
                ):
                    if key in detail:
                        item[key] = detail[key]
        regions = [
            dict(item)
            for item in proposal.evidence
            if str(item.get("kind", "")).endswith("_region")
        ]
        result.evidence = regions + result.evidence
        return result

    def _analyze_annotations(
        self,
        image: Image.Image,
        annotations: ImageAnnotations,
        width: int,
        height: int,
        *,
        external_features: object | None = None,
    ) -> tuple[CueResult, CueResult, list[str]]:
        proposal_errors: list[str] = []
        bundle: AutomaticProposalBundle | None = None
        shadow_needs_proposal = (
            annotations.shadow_applicability is None and not annotations.shadow_pairs
        )
        reflection_needs_proposal = (
            annotations.reflection_applicability is None
            and not annotations.reflection_pairs
        )
        if (
            self.config.automatic.enabled
            and (shadow_needs_proposal or reflection_needs_proposal)
        ):
            try:
                if self.proposal_engine is None:
                    raise RuntimeError("Automatic proposal engine was not initialized")
                bundle = self.proposal_engine.propose(
                    image,
                    external_features=external_features,  # type: ignore[arg-type]
                    include_shadow=shadow_needs_proposal,
                    include_reflection=reflection_needs_proposal,
                )
            except Exception as exc:
                proposal_errors.append(f"Automatic proposal analysis failed: {exc}")

        if annotations.shadow_applicability in {
            "not_applicable",
            "uncertain",
            "unreviewed",
        }:
            decision = annotations.shadow_applicability
            shadow = not_applicable(
                "cast_shadow",
                f"The reviewer marked cast-shadow evidence as {decision.replace('_', ' ')}.",
                limitations=[
                    "Reviewer applicability decisions take precedence over retained point pairs."
                ],
            )
            shadow.measurements = {"review_applicability": decision}
        elif annotations.shadow_pairs:
            shadow = self._decorate_reviewed_result(
                analyze_cast_shadows(
                    annotations.shadow_pairs, width, height, self.config.shadow
                )
            )
        elif bundle is not None:
            if bundle.shadow.applicable:
                shadow = self._decorate_automatic_result(
                    analyze_cast_shadows(
                        bundle.shadow.pairs,  # type: ignore[arg-type]
                        width,
                        height,
                        replace(
                            self.config.shadow,
                            min_pairs=self.config.automatic.min_pairs,
                            inlier_threshold_degrees=max(
                                self.config.shadow.inlier_threshold_degrees,
                                self.config.automatic.shadow_inlier_threshold_degrees,
                            ),
                        ),
                    ),
                    bundle.shadow,
                )
            else:
                shadow = self._proposal_not_applicable(bundle.shadow)
        elif self.config.automatic.enabled and proposal_errors:
            shadow = error_result("cast_shadow", proposal_errors[0])
        else:
            shadow = not_applicable(
                "cast_shadow",
                "No reviewed object-contact/shadow-tip annotations were supplied for this image.",
                limitations=[
                    "Enable automatic proposals or supply reviewed object-shadow pairs."
                ],
            )

        if annotations.reflection_applicability in {
            "not_applicable",
            "uncertain",
            "unreviewed",
        }:
            decision = annotations.reflection_applicability
            reflection = not_applicable(
                "reflection",
                f"The reviewer marked reflection evidence as {decision.replace('_', ' ')}.",
                limitations=[
                    "Reviewer applicability decisions take precedence over retained point pairs."
                ],
            )
            reflection.measurements = {"review_applicability": decision}
        elif annotations.reflection_pairs:
            reflection = self._decorate_reviewed_result(
                analyze_reflections(
                    annotations.reflection_pairs, width, height, self.config.reflection
                )
            )
        elif bundle is not None:
            if bundle.reflection.applicable:
                reflection = self._decorate_automatic_result(
                    analyze_reflections(
                        bundle.reflection.pairs,  # type: ignore[arg-type]
                        width,
                        height,
                        replace(
                            self.config.reflection,
                            min_pairs=self.config.automatic.min_pairs,
                            inlier_threshold_degrees=max(
                                self.config.reflection.inlier_threshold_degrees,
                                self.config.automatic.reflection_inlier_threshold_degrees,
                            ),
                        ),
                    ),
                    bundle.reflection,
                )
            else:
                reflection = self._proposal_not_applicable(bundle.reflection)
        elif self.config.automatic.enabled and proposal_errors:
            reflection = error_result("reflection", proposal_errors[0])
        else:
            reflection = not_applicable(
                "reflection",
                "No reviewed object/reflection correspondences were supplied for this image.",
                limitations=[
                    "Enable automatic proposals or supply reviewed object/reflection matches."
                ],
            )
        return shadow, reflection, proposal_errors

    def analyze_image(
        self,
        image_path: Path,
        *,
        input_root: Path,
        annotation_store: AnnotationStore,
        overlays_dir: Path | None = None,
        external_features: object | None = None,
    ) -> ImageResult:
        errors: list[str] = []
        try:
            with Image.open(image_path) as opened:
                # Match modern browser/image-viewer orientation. Reviewed
                # annotations are expected in the displayed coordinate system.
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            message = f"Could not decode image: {exc}"
            cues = {
                cue: error_result(cue, message)
                for cue in ("perspective", "cast_shadow", "reflection")
            }
            return ImageResult(
                image_path=str(image_path),
                width=0,
                height=0,
                physics=_aggregate(cues),
                cues=cues,
                errors=[message],
            )

        width, height = image.size
        annotation_error: str | None = None
        try:
            annotations = annotation_store.for_image(
                image_path, input_root, width=width, height=height
            )
        except Exception as exc:
            annotation_error = f"Annotation analysis failed: {exc}"
            errors.append(annotation_error)
            annotations = ImageAnnotations()

        try:
            perspective = analyze_perspective(
                image,
                self.config.perspective,
                [
                    region.xyxy
                    for region in annotations.perspective_regions
                    if region.confidence
                    >= self.config.perspective.min_reviewed_region_confidence
                ]
                if annotations.perspective_regions
                else None,
            )
        except Exception as exc:  # isolate cues so one failure cannot hide the rest
            message = f"Perspective analysis failed: {exc}"
            errors.append(message)
            perspective = error_result("perspective", message)

        try:
            if annotation_error is not None:
                raise ValueError(annotation_error.removeprefix("Annotation analysis failed: "))
            shadow, reflection, proposal_errors = self._analyze_annotations(
                image,
                annotations,
                width,
                height,
                external_features=external_features,
            )
            errors.extend(proposal_errors)
        except Exception as exc:
            message = f"Annotation analysis failed: {exc}"
            if message not in errors:
                errors.append(message)
            shadow = error_result("cast_shadow", message)
            reflection = error_result("reflection", message)

        cues = {
            "perspective": perspective,
            "cast_shadow": shadow,
            "reflection": reflection,
        }

        if overlays_dir is not None:
            stem = _safe_overlay_stem(image_path, input_root)
            for cue in cues.values():
                if not cue.evidence:
                    continue
                try:
                    overlay_path = overlays_dir / f"{stem}__{cue.cue}.png"
                    render_cue_overlay(image, cue, overlay_path)
                    cue.overlay_path = str(overlay_path)
                except Exception as exc:
                    errors.append(f"Could not render {cue.cue} overlay: {exc}")

        return ImageResult(
            image_path=str(image_path),
            width=width,
            height=height,
            physics=_aggregate(cues),
            cues=cues,
            errors=errors,
        )

    def run(
        self,
        input_path: str | Path,
        *,
        annotations_path: str | Path | None = None,
        overlays_dir: str | Path | None = None,
        recursive: bool = False,
        max_images: int | None = None,
        dense_feature_maps: dict[str, object] | None = None,
    ) -> BatchResult:
        root = Path(input_path).expanduser().resolve()
        annotation_store = AnnotationStore.from_path(annotations_path)
        overlay_root = (
            Path(overlays_dir).expanduser().resolve() if overlays_dir is not None else None
        )
        excluded_roots: tuple[Path, ...] = ()
        if overlay_root is not None and root.is_dir():
            if overlay_root == root:
                raise ValueError("--overlays-dir cannot be the input directory itself")
            if overlay_root.is_relative_to(root):
                excluded_roots = (overlay_root,)
        images = list(_iter_images(root, recursive, excluded_roots=excluded_roots))
        if max_images is not None:
            images = images[:max_images]
        results: list[ImageResult] = []
        for image_path in images:
            relative_key = image_path.name
            if root.is_dir():
                try:
                    relative_key = image_path.relative_to(root).as_posix()
                except ValueError:
                    pass
            external_features = None
            if dense_feature_maps:
                for feature_key in (relative_key, str(image_path), image_path.name):
                    if feature_key in dense_feature_maps:
                        external_features = dense_feature_maps[feature_key]
                        break
            results.append(
                self.analyze_image(
                    image_path,
                    input_root=root,
                    annotation_store=annotation_store,
                    overlays_dir=overlay_root,
                    external_features=external_features,
                )
            )

        cue_applicability = {
            cue: sum(int(image.cues[cue].applicable) for image in results)
            for cue in ("perspective", "cast_shadow", "reflection")
        }
        summary = {
            "discovered_images": len(images),
            "processed_images": len(results),
            "images_with_errors": sum(int(bool(image.errors)) for image in results),
            "aggregate_status_counts": {
                status: sum(int(image.physics.status == status) for image in results)
                for status in ("consistent", "inconsistent", "indeterminate")
            },
            "cue_applicability_counts": cue_applicability,
            "automatic_proposals": {
                "enabled": self.config.automatic.enabled,
                "mask_backend": self.config.automatic.mask_backend,
                "feature_backend": self.config.automatic.feature_backend,
                "object_backend": self.config.automatic.object_backend,
                "shadow_images_with_automatic_evidence": sum(
                    image.cues["cast_shadow"].measurements.get("evidence_origin")
                    == "automatic_proposal"
                    for image in results
                ),
                "reflection_images_with_automatic_evidence": sum(
                    image.cues["reflection"].measurements.get("evidence_origin")
                    == "automatic_proposal"
                    for image in results
                ),
            },
        }
        return BatchResult(
            schema_version=SCHEMA_VERSION,
            engine_version=ENGINE_VERSION,
            generated_at=utc_now_iso(),
            input_root=str(root),
            images=results,
            summary=summary,
        )
