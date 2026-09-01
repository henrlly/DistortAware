# Physics evidence schema

Schema version `0.1.0` is intentionally independent of any supervised detector. A future fusion layer can consume it without treating physical inconsistency as an AIGC probability.

## Batch result

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Contract version. |
| `engine_version` | string | Physics-engine implementation version. |
| `generated_at` | ISO-8601 string | UTC completion time. |
| `input_root` | string | Input file or directory. |
| `images` | array | Per-image results. |
| `summary` | object | Counts of processed, failed, and cue-applicable images. |

## Image result

| Field | Type | Meaning |
|---|---|---|
| `image_path` | string | Path supplied to the engine. |
| `width`, `height` | integer | Decoded image dimensions. |
| `physics` | object | Confidence-weighted aggregation of applicable cues. |
| `cues` | object | `perspective`, `cast_shadow`, and `reflection` results. |
| `errors` | array | Non-fatal processing errors. |

## Cue result

| Field | Type | Meaning |
|---|---|---|
| `applicable` | boolean | Whether the cue's assumptions and minimum evidence were met. |
| `status` | enum | `consistent`, `inconsistent`, `indeterminate`, `not_applicable`, or `error`. |
| `violation_score` | number/null | `[0,1]`; higher means greater physical inconsistency. Null when inapplicable. |
| `confidence` | number | `[0,1]`; confidence in the measurement, not image authenticity. |
| `summary` | string | Templated human-readable finding. |
| `assumptions` | array | Physical assumptions required by the test. |
| `measurements` | object | Auditable intermediate quantities. |
| `evidence` | array | Lines, points, bundles, and inlier/outlier labels for overlays. |
| `limitations` | array | Cue-specific cautions. |
| `overlay_path` | string/null | Generated overlay when requested. |

### Automatic-proposal provenance

Schema `0.1.0` remains backward-compatible; engines `0.5.0` through `0.6.1` add optional fields inside `measurements` and `evidence` rather than changing required top-level fields.

When automatic shadow/reflection evidence is used, `measurements` includes:

| Field | Meaning |
|---|---|
| `evidence_origin` | `automatic_proposal`; reviewed results use `reviewed`. |
| `proposal_applicable` | Whether enough proposal evidence survived the gates. |
| `proposal_confidence` | Confidence in correspondence discovery, before geometry confidence is combined with it. |
| `proposed_pair_count` | Number of automatic point pairs passed to geometry. |
| `geometry_pair_count` | Number of non-degenerate automatic pairs actually used by the geometric analyzer. |
| `mask_backend` / `feature_backend` / `object_backend` | Backend, model, pinned revision, and learned/fallback metadata where applicable. |
| `proposal_warnings` | Model fallback, missing artifact, or other non-fatal warnings. |
| `automatic_definitive_inconsistency_gate` | `passed`, actual pair count, and required pair count for an automatic inconsistency. |

Evidence can additionally contain `shadow_region`, `mirror_region`, `shadow_pair_proposal`, or `reflection_pair_proposal` records. Region records carry contours/bounds and proposal confidence. Pair evidence carries association or feature-match diagnostics. Geometry evidence still records inlier/outlier roles and angular error.

An automatic result with exactly three pairs may be `consistent`, but cannot be definitively `inconsistent`; the engine downgrades that case to `indeterminate`. Reviewed evidence is not subject to this automatic-only gate.

For a same-pass PatchHead feature grid, `feature_backend` additionally records
`source_detector_family`, `source_checkpoint_sha256`,
`source_checkpoint_dataset`, `feature_dtype`, `grid_shape`,
`shared_primary_forward: true`, and `score_independent: true`. Dense feature
values remain in memory and are never serialized.

## Aggregate physics result

Only applicable cues with numeric scores participate. Scores are weighted by cue confidence. Missing cues are neutral.

`score_kind` is always `physics_violation_not_aigc_probability`. Consumers must not rename it to `pred` or interpret it as the probability that an image is AI-generated.

## Annotation file

Coordinates can be normalized to `[0,1]` or specified in pixels:

```json
{
  "schema_version": "0.1.0",
  "coordinate_space": "normalized",
  "annotation_protocol": "independent_review_v1",
  "reviewer": {"id": "reviewer-a"},
  "images": {
    "scene.png": {
      "perspective": {
        "regions": [
          {"xyxy": [0.05, 0.10, 0.95, 0.90], "confidence": 0.9}
        ]
      },
      "cast_shadow": {
        "applicability": "applicable",
        "pairs": [
          {
            "object_contact": [0.20, 0.72],
            "shadow_tip": [0.42, 0.80],
            "confidence": 1.0
          }
        ]
      },
      "reflection": {
        "applicability": "applicable",
        "pairs": [
          {
            "object_point": [0.35, 0.40],
            "reflection_point": [0.65, 0.40],
            "confidence": 1.0
          }
        ]
      }
    }
  }
}
```

`perspective.regions` contains reviewed structural `xyxy` rectangles. Only regions at or above the configured reviewer-confidence floor (0.5 by default) are used, and only long lines whose midpoint and at least one endpoint fall inside a selected rectangle enter the vanishing-point fit. Shadow/reflection `applicability` is one of `applicable`, `not_applicable`, `uncertain`, or `unreviewed`; an explicit non-applicable/uncertain/unreviewed decision takes precedence over any retained points. The geometry engine remains backward-compatible with pair-only files.

`--annotations` may instead point to a directory. Every top-level `*.json` file is loaded, and its `images` entries are merged. This matches the local annotator's one-export-per-image workflow. Duplicate image keys are an error. Each file may independently use `normalized` or `pixels` coordinates; the loader preserves that setting per image.

Image keys are matched in order against the input-relative POSIX path, filename, and absolute path. An image entry may override `coordinate_space`.

## DID compatibility record

`physics-merge` consumes the official detector's JSON array and adds one field without changing the existing record:

```json
{
  "image_path": "images/scene.png",
  "pred": 0.82,
  "is_aigc": true,
  "physics_evidence": {
    "schema_version": "0.1.0",
    "engine_version": "0.6.1",
    "match_method": "canonical_path",
    "aggregate": {
      "score_kind": "physics_violation_not_aigc_probability",
      "status": "inconsistent",
      "violation_score": 0.71
    },
    "cues": {
      "perspective": {
        "applicable": true,
        "status": "inconsistent",
        "violation_score": 0.76
      }
    },
    "errors": []
  }
}
```

Raw line/point `evidence` arrays are omitted from the merged record to keep inference output compact; they remain in the full physics batch JSON. Missing matches fail by default. `--allow-missing` represents them explicitly as `"physics_evidence": null`.

## PatchHead evidence export

Root `infer.py` is the preferred path and serializes the model's existing patch tensor from the same forward pass as its verdict. `physics-dino-export` remains a standalone compatibility tool. Neither recomputes an attribution method.

```json
{
  "schema_version": "0.2.0",
  "input_root": "/images",
  "detector": {
    "family": "dino_patchhead",
    "arch": "patchhead-dinov3-vitl16",
    "threshold": 0.5,
    "score_kind": "uncalibrated_aigc_classifier_score",
    "score_formula": "0.5 * (sigmoid(mean(patch_logits)) + sigmoid(cls_logit))"
  },
  "images": [
    {
      "image_path": "scene.png",
      "aigc_score": 0.91,
      "is_aigc": true,
      "component_scores": {
        "patch_head": 0.94,
        "cls_head": 0.88
      },
      "patch_evidence": {
        "grid_shape": [1, 2],
        "coordinate_space": "normalized_full_frame",
        "value_kind": "sigmoid_of_per_patch_aigc_logit_uncalibrated",
        "training_supervision": "image_label_repeated_across_all_patches",
        "explains_score_component": "patch_head_only",
        "values": [[0.1, 0.2]]
      }
    }
  ]
}
```

The abbreviated `values` above only illustrates nesting; a real 16×16 map has 16 rows and 16 columns. Values must be finite and lie in `[0,1]`. `coordinate_space` must be `normalized_full_frame`. A crop, letterbox, or other spatial transform requires a new schema field carrying its inverse mapping; unknown conventions fail closed.

## DINO/physics enrichment

`physics-dino-merge` deep-copies every DINO field and adds:

- `physics_evidence`: the same compact sidecar used for DID;
- `dino_physics_alignment`: spatial-association diagnostics.

```json
{
  "dino_physics_alignment": {
    "schema_version": "0.1.0",
    "alignment_kind": "spatial_association_not_causal_attribution",
    "applicable": true,
    "grid_shape": [16, 16],
    "per_cue": {
      "cast_shadow": {
        "applicable": true,
        "association_label": "positive",
        "selected_patch_count": 19,
        "mean_selected_patch_score": 0.81,
        "mean_background_patch_score": 0.43,
        "selected_minus_background": 0.38,
        "top_patch_enrichment_over_area": 2.1,
        "physics_status": "inconsistent"
      }
    },
    "overall": {
      "applicable": true,
      "selected_minus_background": 0.34
    }
  }
}
```

Only indeterminate or inconsistent cues with supported evidence items marked `role: "outlier"` can participate. A consistent cue is explicitly excluded even if its robust fit contains isolated residual lines. `association_label` is a heuristic display summary, not a calibrated probability or proof of model reasoning.

## SID pilot manifest

The pilot manifest records source revision, shard hashes and sizes, deterministic sampling parameters, byte caps, labels, source row indices, extracted paths, and per-record hashes. SID labels remain three-way:

- `0`: `real`;
- `1`: `full_synthetic`;
- `2`: `tampered`.

They are not collapsed into a binary fake class for this explanation-safety study.

## Independent-review agreement report

`physics-review-agreement` compares two distinctly identified reviewer exports. For each shadow/reflection cue it reports reviewed coverage, applicability-decision confusion matrices and Cohen's kappa, resulting geometric-status agreement, and matched endpoint distance in normalized image coordinates. Unreviewed items are excluded from kappa rather than silently treated as `not_applicable`. Agreement measures workflow repeatability, not scene truth or detector accuracy.
