"""Checkpoint-backed local inference backend for browser image crops."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any
import uuid
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError


SERVICE_SCHEMA_VERSION = "0.2.0"
ALLOWED_MEDIA_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
SOURCE_KINDS = {"viewport_capture", "direct_image", "manual_upload", "test_fixture"}


class BrowserInferenceError(ValueError):
    """A request error with a stable API code and HTTP status."""

    def __init__(self, message: str, *, code: str, http_status: int) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


@dataclass(slots=True)
class BackendConfig:
    max_upload_bytes: int = 12 * 1024 * 1024
    max_pixels: int = 25_000_000
    cache_entries: int = 128
    physics_profile: str = "off"
    physics_cache_dir: str | None = None
    physics_offline: bool = False
    strict_physics_models: bool = False
    artifact_profile: str = "off"
    artifact_checkpoint: str | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if self.max_upload_bytes <= 0 or self.max_pixels <= 0:
            raise ValueError("Upload and decoded-pixel limits must be positive")
        if not 0 <= self.cache_entries <= 4096:
            raise ValueError("cache_entries must lie within [0, 4096]")
        if self.physics_profile not in {"off", "heuristic", "learned"}:
            raise ValueError("physics_profile must be off, heuristic, or learned")
        if self.artifact_profile not in {"off", "residual"}:
            raise ValueError("artifact_profile must be off or residual")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validate_image(
    image_bytes: bytes, *, media_type: str, max_upload_bytes: int, max_pixels: int
) -> dict[str, Any]:
    if media_type not in ALLOWED_MEDIA_TYPES:
        raise BrowserInferenceError(
            "Content-Type must be image/png, image/jpeg, or image/webp",
            code="unsupported_media_type",
            http_status=415,
        )
    if not image_bytes:
        raise BrowserInferenceError(
            "Image body is empty", code="empty_image", http_status=400
        )
    if len(image_bytes) > max_upload_bytes:
        raise BrowserInferenceError(
            f"Image exceeds the {max_upload_bytes} byte upload limit",
            code="image_too_large",
            http_status=413,
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as opened:
                opened.verify()
            with Image.open(BytesIO(image_bytes)) as opened:
                width, height = opened.size
                image_format = str(opened.format or "unknown").upper()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombWarning) as exc:
        raise BrowserInferenceError(
            f"Could not decode image: {exc}",
            code="invalid_image",
            http_status=400,
        ) from exc
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise BrowserInferenceError(
            f"Decoded image dimensions {width}x{height} exceed the {max_pixels} pixel limit",
            code="decoded_image_too_large",
            http_status=413,
        )
    return {"width": width, "height": height, "format": image_format}


def _patch_summary(record: dict[str, Any]) -> dict[str, Any] | None:
    evidence = record.get("patch_evidence")
    if not isinstance(evidence, dict) or "values" not in evidence:
        return None
    values = np.asarray(evidence["values"], dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        return None
    flat = values.ravel()
    top_count = max(1, int(np.ceil(0.10 * len(flat))))
    return {
        "grid_shape": list(values.shape),
        "value_kind": evidence.get("value_kind"),
        "minimum": float(values.min()),
        "mean": float(values.mean()),
        "maximum": float(values.max()),
        "top_decile_mean": float(np.sort(flat)[-top_count:].mean()),
        "training_supervision": evidence.get("training_supervision"),
        "interpretation": "weak_image_supervised_localization_not_segmentation",
    }


def _public_physics_evidence(record: dict[str, Any]) -> dict[str, Any] | None:
    """Strip request-local filesystem paths from compact physics evidence."""

    raw = record.get("physics_evidence")
    if not isinstance(raw, dict):
        return None
    evidence = deepcopy(raw)
    evidence.pop("details_image_path", None)
    cues = evidence.get("cues")
    if isinstance(cues, dict):
        for cue in cues.values():
            if isinstance(cue, dict):
                cue.pop("overlay_path", None)
                raw_items = cue.get("evidence")
                if isinstance(raw_items, list):
                    compact_items: list[Any] = []
                    for raw_item in raw_items[:24]:
                        if not isinstance(raw_item, dict):
                            compact_items.append(raw_item)
                            continue
                        item = deepcopy(raw_item)
                        contour = item.pop("contour", None)
                        if isinstance(contour, list):
                            item["contour_point_count"] = len(contour)
                        compact_items.append(item)
                    cue["evidence_count"] = len(raw_items)
                    cue["evidence"] = compact_items
    return evidence


def _artifact_mask_summary(mask: Any) -> dict[str, Any] | None:
    """Return a bounded, JSON-safe summary of a residual model evidence mask."""

    values = np.asarray(mask, dtype=np.float32)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        return None
    values = np.clip(values, 0.0, 1.0)
    selected = values >= 0.5
    bbox: list[float] | None = None
    if selected.any():
        rows, columns = np.nonzero(selected)
        height, width = values.shape
        bbox = [
            round(float(columns.min() / width), 4),
            round(float(rows.min() / height), 4),
            round(float((columns.max() + 1) / width), 4),
            round(float((rows.max() + 1) / height), 4),
        ]
    coarse = Image.fromarray(values).resize((8, 8), Image.Resampling.BOX)
    coarse_values = np.asarray(coarse, dtype=np.float32)
    return {
        "shape": list(values.shape),
        "mean_signal": round(float(values.mean()), 6),
        "maximum_signal": round(float(values.max()), 6),
        "area_fraction_at_0_5": round(float(selected.mean()), 6),
        "normalized_bbox_at_0_5": bbox,
        "coarse_grid_8x8": [
            [round(float(value), 4) for value in row] for row in coarse_values
        ],
        "interpretation": "weak_multitask_localization_not_a_forensic_segmentation",
    }


class BrowserInferenceBackend:
    """Own one detector runtime, optional persistent physics engine, and LRU cache."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        runtime: Any | None = None,
        artifact_predictor: Any | None = None,
        config: BackendConfig | None = None,
    ) -> None:
        self.config = config or BackendConfig()
        if runtime is None:
            if checkpoint is None:
                raise ValueError("A pooled PatchHead checkpoint is required")
            from patchhead.inference import TorchPatchHeadRuntime

            runtime = TorchPatchHeadRuntime(checkpoint, self.config.device)
        self.runtime = runtime
        self.detector_metadata = dict(getattr(runtime, "metadata", {}))
        self._model_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._physics_engine = self._build_physics_engine()
        self._artifact_predictor = artifact_predictor
        self._artifact_metadata: dict[str, Any] = {
            "profile": self.config.artifact_profile,
            "status": "disabled",
        }
        if self.config.artifact_profile == "residual":
            if self._artifact_predictor is None:
                self._artifact_predictor = self._build_artifact_predictor()
            else:
                self._artifact_metadata = {
                    "profile": "residual",
                    "status": "ready",
                    "provider": "injected_residual_predictor",
                }

    def _build_physics_engine(self) -> Any | None:
        profile = self.config.physics_profile
        if profile == "off":
            return None
        physics_source = _repository_root() / "physics" / "src"
        if str(physics_source) not in sys.path:
            sys.path.insert(0, str(physics_source))
        from physics_engine.automatic_config import AutomaticProposalConfig
        from physics_engine.engine import PhysicsEngine, PhysicsEngineConfig

        automatic = AutomaticProposalConfig(
            enabled=True,
            mask_backend="clipseg" if profile == "learned" else "heuristic",
            feature_backend="appearance",
            object_backend="torchvision" if profile == "learned" else "edges",
            appearance_fallback_on_insufficient_external=True,
            cache_dir=self.config.physics_cache_dir,
            local_files_only=self.config.physics_offline,
            allow_model_fallback=not self.config.strict_physics_models,
            device=self.config.device,
        )
        return PhysicsEngine(PhysicsEngineConfig(automatic=automatic))

    def _build_artifact_predictor(self) -> Any | None:
        root = _repository_root() / "filter_based_approach"
        source = root / "src"
        checkpoint = (
            Path(self.config.artifact_checkpoint).expanduser()
            if self.config.artifact_checkpoint
            else root / "models" / "mask_classifier.pt"
        )
        try:
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))
            from ai_detection import MaskPredictor

            predictor = MaskPredictor(checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            benchmark: dict[str, Any] | None = None
            report_path = root / "reports" / "evaluation.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                benchmark = {
                    "sid_binary_accuracy": report.get("SID_Set", {}).get(
                        "binary_accuracy"
                    ),
                    "sid_balanced_accuracy": report.get("SID_Set", {}).get(
                        "balanced_accuracy"
                    ),
                    "sid_tamper_mask_iou": report.get("SID_Set", {}).get(
                        "tamper_mask_iou"
                    ),
                    "wildfake_binary_accuracy": report.get("WildFake", {}).get(
                        "binary_accuracy"
                    ),
                }
            self._artifact_metadata = {
                "profile": "residual",
                "status": "ready",
                "provider": "residual_rgb_highpass_unet",
                "checkpoint_sha256": digest,
                "input_size": int(getattr(predictor, "size", 0)) or None,
                "class_names": list(getattr(predictor, "class_names", [])),
                "benchmark": benchmark,
            }
            return predictor
        except Exception as exc:
            self._artifact_metadata = {
                "profile": "residual",
                "status": "unavailable",
                "reason": str(exc)[:240],
            }
            return None

    def health(self) -> dict[str, Any]:
        return {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "status": "ready",
            "detector": {
                key: self.detector_metadata.get(key)
                for key in (
                    "family",
                    "arch",
                    "backbone",
                    "checkpoint_sha256",
                    "checkpoint_dataset",
                    "threshold",
                    "model_input_size",
                    "checkpoint_compatibility",
                )
            },
            "physics_profile": self.config.physics_profile,
            "artifact": deepcopy(self._artifact_metadata),
            "limits": {
                "max_upload_bytes": self.config.max_upload_bytes,
                "max_pixels": self.config.max_pixels,
                "cache_entries": self.config.cache_entries,
            },
        }

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        if self.config.cache_entries == 0:
            return None
        with self._cache_lock:
            result = self._cache.get(key)
            if result is None:
                return None
            self._cache.move_to_end(key)
            return deepcopy(result)

    def _cache_put(self, key: str, result: dict[str, Any]) -> None:
        if self.config.cache_entries == 0:
            return
        with self._cache_lock:
            self._cache[key] = deepcopy(result)
            self._cache.move_to_end(key)
            while len(self._cache) > self.config.cache_entries:
                self._cache.popitem(last=False)

    def _run_detector(
        self, image_path: Path, *, include_physics: bool
    ) -> dict[str, Any]:
        from patchhead.inference import run_patchhead_inference

        dense_maps: dict[str, Any] | None = {} if include_physics else None

        def collect_features(relative_path: str, values: Any) -> None:
            compact = np.asarray(values, dtype=np.float16)
            dense_maps[relative_path] = {
                "values": compact,
                "backend": "shared_patchhead_dinov3_tokens",
                "model": self.detector_metadata.get("backbone"),
                "metadata": {
                    "source_detector_family": self.detector_metadata.get("family"),
                    "source_checkpoint_sha256": self.detector_metadata.get(
                        "checkpoint_sha256"
                    ),
                    "source_checkpoint_dataset": self.detector_metadata.get(
                        "checkpoint_dataset"
                    ),
                    "feature_dtype": str(compact.dtype),
                    "score_independent": True,
                },
            }

        payload = run_patchhead_inference(
            image_path,
            runtime=self.runtime,
            batch_size=1,
            export_patch_evidence=True,
            dense_feature_sink=collect_features if include_physics else None,
        )
        if include_physics:
            assert self._physics_engine is not None and dense_maps is not None
            from physics_engine.dino_integration import merge_dino_and_physics

            physics = self._physics_engine.run(
                image_path,
                dense_feature_maps=dense_maps,
            ).to_dict()
            payload, summary = merge_dino_and_physics(
                payload,
                physics,
                path_root=_repository_root(),
                allow_missing=False,
            )
            payload["summary"]["physics_integration"] = summary
        return payload

    def _run_artifact(self, image_path: Path) -> dict[str, Any]:
        """Run the residual sidecar without allowing it into the verdict path."""

        assert self._artifact_predictor is not None
        try:
            raw, mask = self._artifact_predictor.predict_path(image_path)
            class_scores = raw.get("class_probabilities", {})
            return {
                "status": "available",
                "provider": "residual_rgb_highpass_unet",
                "predicted_class": raw.get("prediction"),
                "binary_signal": raw.get("binary_prediction"),
                "winning_class_score": float(raw.get("confidence", 0.0)),
                "ai_signal_score": float(raw.get("ai_probability", 0.0)),
                "class_scores": {
                    str(key): float(value)
                    for key, value in class_scores.items()
                },
                "evidence_mask": _artifact_mask_summary(mask),
                "checkpoint_sha256": self._artifact_metadata.get(
                    "checkpoint_sha256"
                ),
                "benchmark": deepcopy(self._artifact_metadata.get("benchmark")),
                "interpretation": "explanation_only_uncalibrated_artifact_signal",
                "limitations": [
                    "Residual traces are not unique to AI; cameras, JPEG, resizing, sharpening, denoising, and screenshots can create similar evidence.",
                    "The bundled localization head achieved only 15.5% IoU on its recorded SID evaluation, so its mask is a weak attention aid rather than a reliable edit boundary.",
                ],
            }
        except Exception as exc:
            return {
                "status": "error",
                "provider": "residual_rgb_highpass_unet",
                "reason": str(exc)[:240],
                "interpretation": "no_artifact_evidence_available",
            }

    def analyze(
        self,
        image_bytes: bytes,
        *,
        media_type: str,
        include_physics: bool = False,
        include_artifacts: bool = False,
        source_kind: str = "viewport_capture",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if source_kind not in SOURCE_KINDS:
            source_kind = "viewport_capture"
        image_metadata = _validate_image(
            image_bytes,
            media_type=media_type,
            max_upload_bytes=self.config.max_upload_bytes,
            max_pixels=self.config.max_pixels,
        )
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        physics_enabled = bool(include_physics and self._physics_engine is not None)
        artifacts_enabled = bool(
            include_artifacts and self._artifact_predictor is not None
        )
        cache_key = (
            f"{image_sha256}:{source_kind}:{bool(include_physics)}:"
            f"{physics_enabled}:{self.config.physics_profile}:"
            f"{bool(include_artifacts)}:{artifacts_enabled}:"
            f"{self.config.artifact_profile}"
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            cached["request_id"] = str(uuid.uuid4())
            cached["cache_hit"] = True
            cached["timing_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            return cached

        with self._model_lock:
            cached = self._cache_get(cache_key)
            if cached is not None:
                cached["request_id"] = str(uuid.uuid4())
                cached["cache_hit"] = True
                cached["timing_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
                return cached
            suffix = ALLOWED_MEDIA_TYPES[media_type]
            with tempfile.TemporaryDirectory(prefix="aigc-browser-request-") as temporary:
                path = Path(temporary) / f"capture{suffix}"
                path.write_bytes(image_bytes)
                payload = self._run_detector(path, include_physics=physics_enabled)
                artifact_evidence = (
                    self._run_artifact(path) if artifacts_enabled else None
                )

        record = payload["images"][0]
        if record.get("error"):
            raise BrowserInferenceError(
                str(record["error"]), code="detector_input_error", http_status=400
            )
        physics_evidence = _public_physics_evidence(record) if physics_enabled else None
        dino_physics_alignment = (
            deepcopy(record.get("dino_physics_alignment"))
            if physics_enabled
            else None
        )
        limitations = list(payload.get("limitations", []))
        limitations.extend(
            [
                "The score is an uncalibrated classifier signal, not a probability or proof of authenticity.",
                "Viewport captures can include browser scaling, page overlays, and only the visible crop of an image.",
            ]
        )
        if include_physics and not physics_enabled:
            limitations.append(
                "Physics was requested but the local service was started with physics_profile=off."
            )
        if include_artifacts and not artifacts_enabled:
            limitations.append(
                "Artifact evidence was requested but the residual sidecar is disabled or unavailable."
            )
        result = {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "request_id": str(uuid.uuid4()),
            "cache_hit": False,
            "image": {
                **image_metadata,
                "sha256": image_sha256,
                "source_kind": source_kind,
            },
            "verdict": {
                "is_aigc": bool(record["is_aigc"]),
                "aigc_score": float(record["aigc_score"]),
                "threshold": float(payload["detector"]["threshold"]),
                "score_kind": payload["detector"].get(
                    "score_kind", "uncalibrated_aigc_classifier_score"
                ),
                "decision_rule": payload["detector"].get("decision_rule"),
            },
            "detector": {
                key: payload["detector"].get(key)
                for key in (
                    "family",
                    "arch",
                    "backbone",
                    "checkpoint_sha256",
                    "checkpoint_dataset",
                    "model_input_size",
                    "score_formula",
                    "checkpoint_compatibility",
                )
            },
            "explanation": {
                "patch_evidence": _patch_summary(record),
                "physics_requested": bool(include_physics),
                "physics_profile": self.config.physics_profile,
                "physics": physics_evidence,
                "dino_physics_alignment": dino_physics_alignment,
                "physics_affects_detector_score": False,
                "artifact_requested": bool(include_artifacts),
                "artifact_profile": self.config.artifact_profile,
                "artifact": artifact_evidence,
                "artifact_affects_detector_score": False,
                "provenance": {
                    "status": "unavailable_from_rendered_crop",
                    "reason": "A viewport capture contains rendered pixels but not trustworthy source EXIF or Content Credentials.",
                    "absence_is_evidence": False,
                    "affects_detector_score": False,
                },
            },
            "limitations": list(dict.fromkeys(limitations)),
            "timing_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        self._cache_put(cache_key, result)
        return result


class PrismGuardBrowserInferenceBackend:
    """Expose a sealed PrismGuard detector without changing its score path.

    Unlike :class:`BrowserInferenceBackend`, this adapter does not synthesize a
    PatchHead score or feed explanation features into prediction.  The browser
    verdict is the calibrated probability returned by PrismGuard's immutable
    DINO-only score trace.  Explanation requests are acknowledged but remain
    unavailable until an independent, post-prediction diagnostics adapter is
    attached.
    """

    def __init__(
        self,
        bundle: str | Path | None = None,
        *,
        checkpoint: str | Path | None = None,
        license_ledger: str | Path | None = None,
        detector: Any | None = None,
        config: BackendConfig | None = None,
        decision_threshold: float = 0.5,
    ) -> None:
        self.config = config or BackendConfig(
            physics_profile="off", artifact_profile="off"
        )
        if self.config.physics_profile != "off" or self.config.artifact_profile != "off":
            raise ValueError(
                "PrismGuard browser diagnostics are separate from prediction; "
                "start this adapter with physics and artifact profiles off"
            )
        if not 0.0 <= decision_threshold <= 1.0:
            raise ValueError("decision_threshold must lie within [0, 1]")
        if detector is None:
            if bundle is None:
                raise ValueError("a sealed PrismGuard detector bundle is required")
            from prismguard.detector import PrismGuardDetector

            detector = PrismGuardDetector.load(
                bundle,
                checkpoint=checkpoint,
                license_ledger=license_ledger,
                device=self.config.device or "cpu",
            )
        self.detector = detector
        self.decision_threshold = float(decision_threshold)
        payload = getattr(detector, "payload", {})
        extractor = payload.get("extractor", {}) if isinstance(payload, dict) else {}
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if extractor.get("type") != "frozen_vfm":
            raise ValueError(
                "PrismGuard browser mode requires a release-eligible frozen-VFM "
                "bundle; engineering smoke bundles are prohibited"
            )
        prediction_contract = getattr(detector, "prediction_contract", {})
        if prediction_contract.get("score_source") != "calibrated_dino_logit_only":
            raise ValueError(
                "PrismGuard browser bundle does not expose the pure-DINO score contract"
            )
        self.detector_metadata = {
            "family": "prismguard_pure_dino",
            "arch": extractor.get("registry_name", "prismguard"),
            "backbone": extractor.get("registry_name"),
            "checkpoint_sha256": extractor.get("checkpoint_sha256"),
            "checkpoint_dataset": metadata.get("training_manifest_sha256"),
            "threshold": self.decision_threshold,
            "model_input_size": extractor.get("input_size", 384),
            "checkpoint_compatibility": "sealed_prismguard_bundle",
        }

    def health(self) -> dict[str, Any]:
        return {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "status": "ready",
            "detector": deepcopy(self.detector_metadata),
            "prediction_contract": {
                "equation": "pred = calibrated_probability(dino_logit)",
                "score_source": "calibrated_dino_logit_only",
                "physics_alpha": 0.0,
                "diagnostics_can_influence_prediction": False,
            },
            "physics_profile": "off",
            "artifact": {"profile": "off", "status": "disabled"},
            "limits": {
                "max_upload_bytes": self.config.max_upload_bytes,
                "max_pixels": self.config.max_pixels,
                "cache_entries": 0,
            },
        }

    def analyze(
        self,
        image_bytes: bytes,
        *,
        media_type: str,
        include_physics: bool = False,
        include_artifacts: bool = False,
        source_kind: str = "viewport_capture",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if source_kind not in SOURCE_KINDS:
            source_kind = "viewport_capture"
        image_metadata = _validate_image(
            image_bytes,
            media_type=media_type,
            max_upload_bytes=self.config.max_upload_bytes,
            max_pixels=self.config.max_pixels,
        )
        with Image.open(BytesIO(image_bytes)) as opened:
            image = opened.convert("RGB").copy()
        trace = self.detector.score_images_with_trace([image], batch_size=1)
        scores = np.asarray(trace.pred, dtype=np.float64)
        raw_logits = np.asarray(trace.raw_logit, dtype=np.float64)
        calibrated_logits = np.asarray(trace.calibrated_logit, dtype=np.float64)
        if scores.shape != (1,) or not np.isfinite(scores).all():
            raise BrowserInferenceError(
                "PrismGuard returned an invalid score",
                code="detector_output_error",
                http_status=500,
            )
        score = float(scores[0])
        limitations = [
            "This is a calibrated detector probability, not proof that an image is AI-generated.",
            "Viewport captures can include browser scaling, page overlays, and only the visible crop of an image.",
        ]
        if include_physics or include_artifacts:
            limitations.append(
                "Independent forensic diagnostics were requested but are not attached to this score-only adapter; prediction completed unchanged."
            )
        return {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "request_id": str(uuid.uuid4()),
            "cache_hit": False,
            "image": {
                **image_metadata,
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "source_kind": source_kind,
            },
            "verdict": {
                "is_aigc": score >= self.decision_threshold,
                "aigc_score": score,
                "threshold": self.decision_threshold,
                "score_kind": "calibrated_dino_probability",
                "decision_rule": "display_only_threshold_on_calibrated_dino_probability",
            },
            "detector": deepcopy(self.detector_metadata),
            "score_trace": {
                "raw_dino_logit": float(raw_logits[0]),
                "calibrated_dino_logit": float(calibrated_logits[0]),
                "prediction_source": str(trace.score_source),
            },
            "explanation": {
                "patch_evidence": None,
                "physics_requested": bool(include_physics),
                "physics_profile": "off",
                "physics": None,
                "dino_physics_alignment": None,
                "physics_affects_detector_score": False,
                "artifact_requested": bool(include_artifacts),
                "artifact_profile": "off",
                "artifact": None,
                "artifact_affects_detector_score": False,
                "provenance": {
                    "status": "unavailable_from_rendered_crop",
                    "absence_is_evidence": False,
                    "affects_detector_score": False,
                },
            },
            "limitations": limitations,
            "timing_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }


class DemoFixtureBrowserInferenceBackend:
    """Deterministic extension wiring fixture with no detection claim."""

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(
            physics_profile="off", artifact_profile="off", cache_entries=0
        )
        if self.config.physics_profile != "off" or self.config.artifact_profile != "off":
            raise ValueError("the wiring fixture does not provide diagnostics")
        self.detector_metadata = {
            "family": "prismguard_browser_wiring_fixture",
            "arch": "deterministic_pixel_fixture",
            "backbone": None,
            "checkpoint_sha256": None,
            "checkpoint_dataset": None,
            "threshold": 0.5,
            "model_input_size": None,
            "checkpoint_compatibility": "no_model_plumbing_only",
        }

    def health(self) -> dict[str, Any]:
        return {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "status": "ready",
            "scientific_status": "plumbing_only_no_aigc_performance_claim",
            "detector": deepcopy(self.detector_metadata),
            "prediction_contract": {
                "score_source": "deterministic_pixel_fixture_not_a_model",
                "diagnostics_can_influence_prediction": False,
            },
            "physics_profile": "off",
            "artifact": {"profile": "off", "status": "disabled"},
            "limits": {
                "max_upload_bytes": self.config.max_upload_bytes,
                "max_pixels": self.config.max_pixels,
                "cache_entries": 0,
            },
        }

    def analyze(
        self,
        image_bytes: bytes,
        *,
        media_type: str,
        include_physics: bool = False,
        include_artifacts: bool = False,
        source_kind: str = "viewport_capture",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        image_metadata = _validate_image(
            image_bytes,
            media_type=media_type,
            max_upload_bytes=self.config.max_upload_bytes,
            max_pixels=self.config.max_pixels,
        )
        with Image.open(BytesIO(image_bytes)) as opened:
            pixels = np.asarray(
                opened.convert("L").resize((32, 32), Image.Resampling.BOX),
                dtype=np.float64,
            )
        # Deliberately simple and deterministic: this is solely a transport/UI
        # fixture, never an AIGC detector or accuracy baseline.
        score = float(np.clip(0.1 + pixels.std() / 128.0, 0.05, 0.95))
        return {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "scientific_status": "plumbing_only_no_aigc_performance_claim",
            "request_id": str(uuid.uuid4()),
            "cache_hit": False,
            "image": {
                **image_metadata,
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "source_kind": source_kind if source_kind in SOURCE_KINDS else "viewport_capture",
            },
            "verdict": {
                "is_aigc": score >= 0.5,
                "aigc_score": score,
                "threshold": 0.5,
                "score_kind": "demo_fixture_signal_not_a_probability",
                "decision_rule": "wiring_demo_only",
            },
            "detector": deepcopy(self.detector_metadata),
            "explanation": {
                "patch_evidence": None,
                "physics_requested": bool(include_physics),
                "physics_profile": "off",
                "physics": None,
                "dino_physics_alignment": None,
                "physics_affects_detector_score": False,
                "artifact_requested": bool(include_artifacts),
                "artifact_profile": "off",
                "artifact": None,
                "artifact_affects_detector_score": False,
            },
            "limitations": [
                "WIRING DEMO ONLY: this deterministic pixel signal is not an AIGC detector.",
                "Do not report its scores as model predictions or accuracy evidence.",
            ],
            "timing_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }


__all__ = [
    "ALLOWED_MEDIA_TYPES",
    "BackendConfig",
    "BrowserInferenceBackend",
    "PrismGuardBrowserInferenceBackend",
    "BrowserInferenceError",
    "DemoFixtureBrowserInferenceBackend",
    "SERVICE_SCHEMA_VERSION",
]
