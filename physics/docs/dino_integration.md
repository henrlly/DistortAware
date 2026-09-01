# DINO PatchHead explanation integration

## Reviewed upstream interface

The merged detector is a frozen DINOv3 ViT-L/16 backbone with rank-8 LoRA and a spatial PatchHead. At the official 256-pixel input, the head emits a 16×16 logit map.

The official forward contract is:

```python
image_logit, cls_logit, patch_logits = model(x)
score = 0.5 * (sigmoid(image_logit) + sigmoid(cls_logit))
```

The historical evaluator saves only `score` and the thresholded prediction. Root `infer.py` now optionally retains `patch_logits`, the most direct spatial evidence available, from that same forward pass.

For automatic reflection proposals, `PatchHeadDetector.forward_with_features` can additionally expose the final dense DINO grid. The ordinary `forward` contract above remains unchanged. Root inference requests the grid only with `--physics-proposal-feature-backend patchhead`, converts it to float16, transfers it in memory only, and enforces a 512 MiB default/2 GiB maximum budget.

The official results motivate prioritizing this path:

| Test set | PatchHead clean / mean-14 | DID clean / mean-14 |
|---|---:|---:|
| WildFake | 98.5% / 99.0% | 88.6% / 93.0% |
| SID_Set native | 100% / 99.3% | 92.7% / 87.4% |
| Pooled | 99.6% / 98.6% | 87.2% / 89.3% |

These are in-distribution results. The official cross-dataset experiments also show PatchHead collapsing near chance (WildFake→SID 47.3%, SID→WildFake 51.8%), so the evidence map must not be described as a universal synthetic-image explanation.

## Implemented paths

### Unified inference (preferred)

`patchhead/inference.py` provides checkpoint-independent orchestration and a lazy real-model runtime. Root `infer.py` uses it as the default detector, keeps compact output by default, and can attach physics without running PatchHead twice. Contract tests cover score composition, preprocessing, EXIF orientation, duplicate basenames, corrupt inputs, non-square future maps, malformed tensors, and optional physics merging.

Automatic reflections can reuse the same dense DINO tokens for correspondence matching, avoiding a second standalone DINO forward. Automatic shadow/mirror masks and optional object boxes remain separate providers because PatchHead is trained only for image/patch AIGC classification and has no supervised shadow or mirror segmentation head.

### `physics-dino-export`

The exporter dynamically imports the official `patchhead/model.py` and calls its `load_detector`. It intentionally does not copy or reimplement DINO. For each image it records:

- final AIGC score and thresholded verdict;
- patch-head and CLS-head component scores;
- `sigmoid(patch_logits)` as a 2-D array;
- grid shape, coordinate convention, checkpoint hash, preprocessing, and score formula.

The exporter is intentionally specific to the official PatchHead class. The downstream merge, alignment, and renderer are grid-shape agnostic, so another DINOv2/DINOv3 backbone or patch size can use them if it emits the same normalized full-frame evidence contract.

The official full-frame resize maps patch cells monotonically to normalized original-image coordinates. If the upstream pipeline later adds crop, letterbox, padding, or region proposals, the contract must add an explicit inverse transform before spatial comparison can continue.

### `physics-dino-merge`

The merge deep-copies detector output. It never mutates `aigc_score`, `is_aigc`, component scores, threshold, or patch values. Image joining prefers canonical paths and allows a basename fallback only when unique.

Physics outlier segments are rasterized onto the DINO grid. A cue participates only when:

1. the cue is applicable;
2. its status is `indeterminate` or `inconsistent`;
3. at least one supported evidence segment has `role: "outlier"`;
4. the coordinate mapping is known.

For a selected patch mask `M` and patch-score grid `S`, the report includes:

- `mean(S[M]) - mean(S[not M])`;
- selected area fraction;
- overlap with the top 15% of DINO patches;
- top-patch precision and selected-patch recall;
- top-patch precision divided by selected area, reported as enrichment.

The qualitative `positive`, `negative`, or `weak_or_mixed` label is only a display heuristic. The result declares `spatial_association_not_causal_attribution` and repeats four limitations in every record.

### `physics-dino-render`

The renderer creates three panels:

1. original image;
2. DINO patch heatmap;
3. heatmap with outlier geometry from non-consistent physics cues.

Cue colors are kept separate: perspective magenta, cast shadow orange-red, and reflection cyan. The mock output in `outputs/dino_overlays` verifies presentation plumbing but must not be shown as real model attribution.

## Why the map is not a segmentation explanation

PatchHead trains every spatial patch with the image label. A synthetic image therefore teaches all 256 patches to be positive, including visually authentic regions; a real image teaches all patches to be negative. This weak supervision can yield useful spatial variation, but it does not provide pixel-level manipulation labels.

Also, the final score is half CLS-head score. No spatial map explains that component. The exported patch map therefore describes one direct component of the detector, not the complete decision.

For doctored images, the strongest follow-up evaluation is to compare the map against SID masks while stratifying by manipulation size and generator. Metrics should include patch-level average precision, IoU only after selecting a threshold on development data, pointing-game accuracy, and score mass inside the mask. Physics overlap should remain a separate diagnostic.

## Real-model validation

The supplied external pooled checkpoint and pinned DINOv3-L backbone have now
been exercised on bounded SID and WildFake samples. Real patch maps and
same-pass 16×16×1024 DINO grids were exported in memory, source checkpoint/model
provenance was retained, and matched detector-only/integrated SID runs preserved
all 150 primary scores, component scores, and verdicts exactly. Five SID mirror
candidates consumed the shared grid before reflection confidence gates safely
abstained.

Against 50 SID tamper masks, the image-supervised patch map had mean patch ROC
AUC 0.582 and mean top-area IoU 0.201. This confirms weak localization but does
not support a segmentation or causal-attribution claim. See
[`checkpoint_validation.md`](checkpoint_validation.md) for hashes, commands,
transformation metrics, and limitations. Checkpoints and backbones remain
external artifacts and must not be committed.

## Completed upstream contract change

The official inference path now optionally serializes `patch_logits` from the same forward pass through `--export-patch-evidence`. Default prediction JSON remains compact, the output follows `docs/schema.md`, and the checkpoint threshold and score formula are preserved exactly.

Checkpoint-independent contract tests now cover:

- one image and a recursive directory;
- corrupt images;
- duplicate basenames in different folders;
- EXIF-rotated images;
- non-16×16 future grids;
- unknown crop/letterbox mappings;
- records with no applicable physics cue;
- globally consistent cues containing isolated fit residuals.
