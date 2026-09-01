# Physics-engine architecture

## Product boundary

The repository has two visual detectors: DINOv3 PatchHead and DID. Measured results make PatchHead the primary decision channel; DID is useful as an ablation. Physics remains an applicability-aware explanation component.

```text
PatchHead AIGC score/verdict ------------------------------> primary result
       |                                                           |
       +-- existing 2-D patch logits --> spatial association ------+--> presentation
                                                                   |
image --> physics cues --> physical violation evidence ------------+
```

Physics output always declares:

```text
score_kind = physics_violation_not_aigc_probability
```

No adapter averages, overwrites, recalibrates, or relabels the detector score.

## Physics flow

```text
image
  +-- projection/orientation gates --> optional reviewed regions
  +-- long-line extraction --> robust vanishing-point bundles --> perspective
  +-- suspicious result --> nearby center-view stability gate

automatic proposal layer (only when reviewed evidence is absent)
  +-- shadow mask --> object/foreground association --> contact/tip proposals
  +-- mirror mask --> dense feature matches --> direct/reflected proposals

reviewed or automatic correspondences
  +-- object contact -> shadow tip --> projected light constraint --> cast shadow
  +-- object point -> reflected point --> normal VP constraint --> reflection

applicable cues --> confidence-weighted violation aggregate --> JSON + overlays
```

Each cue independently returns `consistent`, `inconsistent`, `indeterminate`, `not_applicable`, or `error`. Missing cues are neutral and failures are isolated.

### Perspective

OpenCV LSD supplies long line segments. Homogeneous geometry robustly groups them into at most three vanishing-point families. The score combines unexplained line length and angular residual.

Applicability now considers line count, coverage, orientation concentration/entropy, and panorama-like frame geometry. A suspicious full-frame result is tested on 90% and 80% center views; material score range, applicability loss, or consistent↔inconsistent disagreement downgrades the display to `indeterminate`. Reviewers can draw structural rectangles so only defensible regions enter the fit.

The gate remains conservative rather than semantic. It can still accept object contours or reject valid non-Manhattan scenes, and an aspect-ratio check cannot identify every stitched or distorted projection.

### Cast shadows and reflections

The compact projective tests remain separate from correspondence discovery. Version 0.5.0 adds a confidence-gated automatic proposal layer while preserving reviewed evidence as the higher-priority path.

CLIPSeg or conservative OpenCV priors first propose shadow and mirror regions. Shadow components are reduced to principal endpoints and associated with nearby torchvision object boxes or local foreground/edge support; the supported endpoint is treated as object contact and the opposite endpoint as shadow tip. Reflection candidates use DINOv3, same-pass PatchHead tokens, or appearance descriptors. Mutual-nearest-neighbour, feature margin, saliency, separation, and diversity gates retain direct/reflected point pairs.

Only then do the original geometric tests decide whether vectors fit one projected light source/direction or reflection connectors fit one planar-reflector normal vanishing point. Proposal confidence multiplies geometry confidence. Fewer than three pairs abstain, and an automatic inconsistency requires at least four pairs; an apparent inconsistency from exactly three pairs is downgraded to `indeterminate`.

Reviewed pairs or explicit reviewed applicability decisions override automatic proposals cue by cue. This keeps automatic mode usable without removing a correction/calibration route.

## DINO PatchHead integration

The official `PatchHeadDetector.forward` returns:

```text
image_logit, cls_logit, patch_logits[B, H, W]
```

The official final score is:

```text
0.5 * (sigmoid(image_logit) + sigmoid(cls_logit))
```

`image_logit` is the mean of patch logits before the sigmoid. Root `infer.py` now exports optional `sigmoid(patch_logits)` from this same forward pass while preserving the official score and threshold. `physics-dino-export` remains a standalone compatibility path.

When `--physics-proposal-feature-backend patchhead` is selected, `forward_with_features` also returns the final dense DINO grid from that same pass. The grid is transferred as float16 in memory only and is capped at 512 MiB by default (2 GiB hard CLI maximum). PatchHead's ordinary `forward` signature and score formula are unchanged. Real pooled-checkpoint validation now records the source checkpoint hash/model on every consumed grid; without that external artifact, standalone DINOv3-small supplies proposal descriptors through the same interface.

The exporter declares the patch map in `normalized_full_frame` coordinates because preprocessing uses full-frame resize without crop or letterboxing. Any future crop/letterbox pipeline must supply an inverse transform; unknown mappings fail closed.

### Spatial association

Only outlier segments from indeterminate or inconsistent physics cues become candidate explanation geometry. They are rasterized onto the DINO grid with a small patch-scale buffer. For each cue and their union, the adapter reports:

- selected and background patch counts;
- mean DINO score near geometry and elsewhere;
- selected-minus-background lift;
- top-15% patch precision, selected-patch recall, and area-normalized enrichment.

This is labelled `spatial_association_not_causal_attribution`. It cannot establish that DINO used a shadow or reflection rule. PatchHead gives every patch the image label during training, so its map is weak localization. The CLS head also contributes half of the final score and has no spatial map.

`physics-dino-render` shows original, DINO heatmap, and heatmap with physics residual geometry as three separate panels so viewers can see the boundary.

## SID_Set pilot architecture

SID_Set embeds images and tamper masks inside roughly 140 GB of Parquet shards. `physics-sid-pilot`:

1. preflights actual shard bytes and rejects Git LFS pointers;
2. enforces configurable caps with hard 50 GiB source and 10 GiB extraction ceilings;
3. streams record batches instead of loading the shard at once;
4. keeps a deterministic per-label reservoir in memory;
5. extracts only selected records and records IDs, row indices, hashes, sizes, and source revision;
6. runs physics with automatic shadow/reflection disabled unless explicitly requested;
7. reports cue coverage, status distribution, real-image explanation safety, and sparse line-residual/tamper-mask overlap.

## DID compatibility

`physics-merge` still accepts the official DID array and adds compact `physics_evidence` while preserving `pred` and `is_aigc`. It is a compatibility/ablation interface, not the preferred primary path after the PatchHead results.

## Remaining priorities

1. Complete two independent reviews and use them to measure automatic pair precision, endpoint error, cue coverage, and calibration—not only mask IoU.
2. Build a larger held-out natural-scene benchmark stratified by light count, soft/hard shadows, mirror type, repeated texture, and scene class while respecting dataset licenses and storage caps.
3. Compare the zero-shot providers with a modern instance shadow/contact model and mirror-instance model on available GPU compute.
4. Freeze independent reviewed shadow/contact and reflection-correspondence ground truth, then measure pair precision and endpoint error.
5. Broaden natural-scene and transformation coverage while keeping selective sources under the declared storage caps.
6. Add richer projection/lens-distortion screening and calibrate every explanation display rule separately from the primary detector threshold.
