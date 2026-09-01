# Automatic shadow and reflection proposals

## Status and purpose

Version 0.6.1 removes the requirement for a reviewer to mark every shadow endpoint or reflection correspondence. The automatic layer proposes evidence; the existing projective analyzers remain the independent consistency judges.

This distinction is important:

```text
proposal model -> candidate regions and pairs -> geometric constraint test -> explanation
```

The proposal model does not output an AI probability. Physics does not vote on or overwrite the primary PatchHead verdict. A generated image can satisfy all tested constraints, and a real or edited image can violate them.

## Data flow

```text
RGB image
  |
  +-- shadow probability map -- connected components -- object/foreground association
  |                                                     |
  |                                      object contact -> shadow tip
  |                                                     |
  |                                      robust projected-light fit
  |
  +-- mirror probability map -- planar regions -- dense direct/reflected matches
                                                        |
                                        mutual nearest neighbours
                                                        |
                                        robust normal-VP fit

reviewed cue evidence, when present --------------------> overrides automatic cue
```

All learned dependencies are optional and imported lazily. Importing root `infer.py`, checking a checkpoint contract, or running core geometry does not load OpenCV/Torch model code unless automatic proposals are enabled.

## Region proposal backends

### CLIPSeg

`clipseg` is the recommended POC backend. It uses [`CIDAS/clipseg-rd64-refined`](https://huggingface.co/CIDAS/clipseg-rd64-refined) at pinned revision `999e0328d9e10b484360c477313983f9afdd7050`, with prompt ensembles for cast-shadow and mirror concepts. Prompt probabilities are combined into one shadow map and one mirror map. Version 0.6.1 then retains strong local-physics proposals by taking the maximum with the full photometric shadow prior and 0.8 times the framed-mirror prior. This is proposal fusion only; projective geometry remains the independent verifier.

Default thresholds are:

- shadow: `0.40`, selected from the bounded SBU smoke test;
- mirror: `0.54`, used for the bounded PMD smoke test.

Connected components outside configured area bounds are discarded. Remaining components retain their probability-derived confidence, bounding box, contour, area fraction, model ID, and backend metadata.

### Heuristic fallback

`heuristic` requires no downloaded model. Shadow probability combines local darkness, approximate channel-ratio preservation, and smoothness. Mirror probability searches for sufficiently large framed quadrilaterals with internal texture and strong borders.

This fallback exists for deterministic fixtures, offline development, and constrained scenes. Its lower measured mask quality and explicit provenance prevent it from being presented as equivalent to a learned segmenter.

## Automatic cast-shadow pairing

For every accepted shadow component:

1. principal-component endpoints describe its dominant direction;
2. nearby object boxes, when enabled, are scored for plausible contact with one endpoint;
3. if no learned box is suitable, edge/foreground support immediately above each endpoint supplies a conservative contact prior;
4. the supported endpoint becomes `object_contact` and the opposite endpoint becomes `shadow_tip`;
5. component confidence, association confidence, segment length, and support quality form the pair confidence;
6. low-confidence pairs are removed before geometry.

The learned `torchvision` option uses a pretrained MobileNet Faster R-CNN only to suggest generic object boxes. It does not claim to segment feet or ground contact. `edges` is the zero-download alternative. Object inference is skipped entirely when no shadow region survives.

The shadow analyzer then checks whether the proposed vectors converge to one projected point light or remain parallel under a distant directional light. It assumes comparable object-ground geometry, one dominant light, and valid associations.

## Automatic reflection pairing

For every accepted planar-mirror region:

1. extract a dense grid with standalone DINOv3, same-pass PatchHead DINO tokens, or local appearance descriptors;
2. divide valid tokens into points inside and outside the mirror region;
3. find cosine-similar direct/reflected candidates;
4. retain only mutual nearest neighbours with adequate similarity, nearest-neighbour margin, spatial separation, and saliency;
5. enforce spatial diversity so one small repeated texture cannot supply every pair;
6. map grid coordinates back to display pixels and pass the connectors to projective geometry.

The standalone learned default is timm `vit_small_patch16_dinov3.lvd1689m` at pinned Hub revision `3bf4720a82ec2066db88137180ff1f83a675cef0`, a small DINOv3 encoder. The feature stage is skipped when no mirror region survives.

Reflection geometry expects connectors between corresponding direct and reflected points to agree on the reflector's normal vanishing point. This currently covers planar mirrors. Water, curved reflectors, windows, screens, repeated objects, and severe occlusion can violate the assumptions.

## Primary-transformer reuse

PatchHead's normal forward contract remains:

```text
image_logit, cls_logit, patch_logits
```

`forward_with_features` adds the final dense DINO grid for callers that explicitly request it. Root `infer.py` can choose `--physics-proposal-feature-backend patchhead`; the same primary forward then supplies both the detector outputs and reflection descriptors.

The grid is converted to float16 and held only in memory. It is never added to output JSON. The default transfer cap is 512 MiB and values above 2 GiB are rejected. The real pooled checkpoint now validates this path end to end: five candidate SID mirror regions consumed same-pass features, every source checkpoint/model field was retained in provenance, and detector-only versus integrated scores and verdicts remained exactly equal under matched settings. No SID reflection passed the correspondence applicability gate; feature-path execution is not a claim of reflection coverage or accuracy.

Future trained shadow-instance, mirror-instance, object-contact, or correspondence heads can replace the current providers through `MaskProvider`, `ObjectProvider`, and `FeatureProvider`. They must still pass confidence and provenance to the independent geometry layer.

For the browser profile, same-pass PatchHead DINO is always attempted first. If
a mirror region exists but DINO supplies fewer than three accepted matches, the
engine may compare a cheap local-appearance descriptor pass. It adopts that
fallback only when it yields more accepted correspondences. Measurements retain
the primary DINO backend and pair count, the fallback count and selected
backend, and an explicit repeated-texture limitation. The fallback still cannot
bypass the three-pair applicability rule or four-pair automatic-inconsistency
gate.

## Precedence and safety semantics

- Reviewed pairs or explicit reviewed applicability decisions override automatic proposals for that cue; unnecessary object/DINO inference is skipped for the covered cue.
- Automatic mode fills only missing shadow/reflection evidence.
- Fewer than three usable pairs are `not_applicable`.
- Proposal confidence multiplies geometry confidence.
- Exactly three automatic pairs may support `consistent`, but cannot produce a definitive `inconsistent` result.
- Automatic inconsistency requires at least four pairs; otherwise status becomes `indeterminate` and the displayed violation score is capped at `0.5`.
- Missing models can fall back to deterministic providers unless `--strict-proposal-models` is set.
- Every result declares `evidence_origin`, backend/model/revision metadata, proposal confidence, pair count, warnings, limitations, and the definitive-inconsistency gate state.

These gates reduce unsupported claims; they do not calibrate natural-image accuracy.

## Running the automatic engine

Install from `physics/`:

```bash
python -m pip install -e '.[auto,eval]'
```

Learned standalone run:

```bash
physics-engine examples/demo_images/automatic_consistent.png \
  --auto-proposals \
  --proposal-mask-backend clipseg \
  --proposal-feature-backend dinov3 \
  --proposal-object-backend torchvision \
  --proposal-cache-dir ../cache/physics-auto \
  --output outputs/automatic_demo.json \
  --overlays-dir outputs/automatic_demo_overlays \
  --pretty
```

Use the two generated overlay files for a presentation. Shadow overlays show detected regions and contact-to-tip arrows. Reflection overlays show the mirror contour and feature matches. Green geometry is inlier evidence; red geometry is an outlier.

The companion `automatic_inconsistent.png` is physically inconsistent but does not pass the tested zero-shot proposal gates: no shadow component survives and only two reflection matches survive. Keep it as the presentation's abstention/coverage example rather than claiming the POC detects every visible inconsistency.

Offline repeat after prefetch:

```bash
physics-engine examples/demo_images/automatic_consistent.png \
  --auto-proposals \
  --proposal-mask-backend clipseg \
  --proposal-feature-backend dinov3 \
  --proposal-object-backend torchvision \
  --proposal-cache-dir ../cache/physics-auto \
  --proposal-offline \
  --strict-proposal-models \
  --output outputs/automatic_offline.json
```

## Storage-bounded mask evaluation

The mask evaluator reads only selected records and rejects any configured source cap above 50 GiB. The defaults are 24 images, 12 overlays, a 2 GiB source cap, and seed 2026.

### SBU shadow test

Source: [official SBU Shadow Dataset page](https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html). The locally used ZIP is 286 MB and has SHA-256 `8ac703bc4ec0cf3f57e128a6e7a8ed73680d9f68e4e22d08fc83a3f60c612109`. Its included README restricts use to non-commercial research; do not redistribute it with the project.

```bash
physics-mask-eval ../cache/physics-auto-eval/SBU-shadow.zip \
  --source-format sbu-zip \
  --cue shadow \
  --backend clipseg \
  --threshold 0.40 \
  --cache-dir ../cache/physics-auto \
  --offline \
  --max-images 24 \
  --max-source-gib 2 \
  --dataset-name SBU-shadow-test \
  --dataset-license non-commercial-research \
  --output outputs/automatic_mask_eval/sbu_clipseg_t040_24.json \
  --overlays-dir outputs/automatic_mask_eval/sbu_clipseg_t040_24_overlays \
  --pretty
```

### PMD mirror test

Source: [`garrying/PMD`](https://huggingface.co/datasets/garrying/PMD), pinned locally at revision `73c8ae81846070f410dd16399ff629104173100d`. The test Parquet is 169.6 MB. Its dataset card declares CC BY-NC 4.0; keep it out of release artifacts unless redistribution terms are reviewed.

```bash
physics-mask-eval ../cache/physics-auto-eval/pmd/data/test-00000-of-00001.parquet \
  --source-format hf-parquet \
  --cue mirror \
  --backend clipseg \
  --threshold 0.54 \
  --cache-dir ../cache/physics-auto \
  --offline \
  --max-images 24 \
  --max-source-gib 2 \
  --dataset-name PMD-test \
  --dataset-revision 73c8ae81846070f410dd16399ff629104173100d \
  --dataset-license CC-BY-NC-4.0 \
  --output outputs/automatic_mask_eval/pmd_clipseg_24.json \
  --overlays-dir outputs/automatic_mask_eval/pmd_clipseg_24_overlays \
  --pretty
```

### Bounded results

All rows below use the same 24 deterministic test records per source.

| Cue/source | Backend | Threshold | IoU | Dice | Precision | Recall | Seconds/image |
|---|---|---:|---:|---:|---:|---:|---:|
| Shadow / SBU | CLIPSeg + photometric prior | 0.40 | 0.4866 | 0.6107 | 0.8335 | 0.5589 | not used as a latency claim |
| Shadow / SBU | Heuristic | 0.48 | 0.0404 | 0.0673 | 0.4924 | 0.0422 | not used as a latency claim |
| Mirror / PMD | CLIPSeg + 0.8× framed-mirror prior | 0.54 | 0.2981 | 0.3670 | 0.5344 | 0.3767 | not used as a latency claim |
| Mirror / PMD | Heuristic | 0.54 | 0.1015 | 0.1104 | 0.1031 | 0.1232 | not used as a latency claim |

These are macro image-level mask metrics on a small deterministic subset. They do not evaluate object-shadow association, correspondence correctness, geometry, generated-image detection, or cross-dataset calibration. Overlays encode true positive in green, false positive in red, and false negative in blue.

The same learned browser profile was also run end to end on the local 12-image WildFake demonstration wall (six COCO real and six DALL-E Advanced images, evaluation only), including pooled PatchHead same-pass DINO and appearance fallback. Perspective produced 5 `consistent`, 3 `indeterminate`, and 4 `not_applicable` results. Cast shadows produced one three-pair `indeterminate` result and 11 abstentions. Shared DINO alone had too few matches on the two proposed mirrors; the disclosed appearance fallback produced two `inconsistent` results and 10 abstentions. Both displayed reflection inconsistencies happened to be DALL-E examples, but this tiny selected wall is not an accuracy estimate. Candidate counts, selected feature backend, and abstention reasons are surfaced by the browser product instead of reducing all abstentions to a generic label.

## Transformation smoke test

The retained learned run used `automatic_consistent.png`, CLIPSeg, DINOv3-small, the edge object backend, cached/offline artifacts, and deliberately POC-level acceptance thresholds:

```bash
physics-battle-test examples/demo_images/automatic_consistent.png \
  --auto-proposals \
  --proposal-mask-backend clipseg \
  --proposal-feature-backend dinov3 \
  --proposal-object-backend edges \
  --proposal-cache-dir ../cache/physics-auto \
  --proposal-offline \
  --min-applicability-retention 0.70 \
  --max-hard-flips 0 \
  --max-mean-score-drift 0.25 \
  --max-score-drift 0.55 \
  --output outputs/automatic_battle_test/automatic_consistent_gated.json \
  --strict
```

| Cue | Applicability retained | Hard flips | Mean drift | Maximum drift |
|---|---:|---:|---:|---:|
| Perspective | 13/14 (92.9%) | 0 | 0.000 | 0.000 |
| Cast shadow | 12/14 (85.7%) | 0 | 0.128 | 0.356 |
| Reflection | 13/14 (92.9%) | 0 | 0.007 | 0.090 |

Runtime was 15.55 seconds for clean plus 14 transforms on the development CPU. Crop/noise caused some cue abstention. The four-pair gate converted a colour-jitter shadow inconsistency based on only three automatic pairs into `indeterminate`, eliminating the hard flip. This validates the safety mechanism on one synthetic fixture, not production robustness.

## SID smoke result

The engine was run on six already-extracted, storage-capped SID pilot images: two real, two full-synthetic, and two tampered. Automatic shadow and reflection proposals abstained on all six; perspective remained applicable and consistent. This avoids a false claim on those examples but is not an accuracy result and does not use shadow/reflection ground truth.

## Known failure modes and next training path

- CLIPSeg can merge adjacent shadows, classify dark objects as shadows, miss soft/partial shadows, or overextend mirror regions.
- Generic object boxes do not supply instance shadow ownership or exact ground contact.
- DINO similarity can match repeated textures, screens, windows, or symmetric objects even with mutual-neighbour and diversity gates.
- Too few distinct objects/shadows or reflected features forces abstention.
- Multiple lights and non-planar reflectors violate the current projective models.
- Mask benchmark licenses are unsuitable for unrestricted redistribution and may be unsuitable for a commercial training release.

A stronger future system would train instance shadow/object association on LISA/SSIS/SOBA-style labels and mirror instances on PMD-style labels, then calibrate proposal confidence on a held-out domain. The current laptop-safe implementation deliberately avoids legacy CUDA/Detectron2 pipelines and unbounded training datasets. Provider interfaces allow those learned heads to be inserted later without changing the geometry schema.
