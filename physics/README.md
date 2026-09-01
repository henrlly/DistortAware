# Physics Explanation Sidecar

Version 0.6.1 provides applicability-aware scene-consistency evidence beside the repository's primary **DINOv3 PatchHead** detector. Physics never changes the detector score or verdict, and a physics-violation score is never presented as an AIGC probability.

The component now supports two evidence paths:

- **Automatic proposals:** CLIPSeg or conservative OpenCV masks propose cast-shadow and mirror regions; object/edge evidence proposes shadow endpoints; DINOv3 or appearance features propose direct/reflected correspondences; the existing projective geometry independently checks consistency.
- **Reviewed evidence:** a person can still supply or correct regions and point pairs. Explicit reviewed decisions always override automatic proposals.

The pooled PatchHead checkpoint is not required for standalone physics automation. A supplied external pooled checkpoint has now been validated for real primary scores and same-pass DINO-grid reuse; it remains deliberately untracked.

## Implemented scope

- Automatic straight-line clustering and explicit vanishing-point estimation.
- Perspective applicability gates for line coverage, orientation structure, panorama-like frames, and stability across nearby crops.
- Optional reviewed structural rectangles for suppressing semantic/decorative edges.
- Automatic cast-shadow region discovery and object-contact/shadow-tip proposals.
- Automatic planar-mirror discovery and mutual-nearest-neighbour reflection matching.
- Optional local-appearance fallback when a proposed mirror has too few
  same-pass DINO matches, with both attempts disclosed and all geometry safety
  gates retained.
- Optional zero-shot CLIPSeg masks, DINOv3 dense features, and torchvision object boxes, all loaded lazily; Hub-backed learned defaults are revision-pinned and report their provenance.
- Conservative fusion of zero-shot masks with photometric shadow and framed-mirror priors, measured on bounded 24-image SBU/PMD subsets without relaxing correspondence gates.
- Download-free OpenCV/appearance fallbacks for constrained demonstrations.
- Confidence propagation, model provenance, assumptions, warnings, and human-visible overlays.
- A safety gate that prevents only three automatic pairs from asserting a definitive inconsistency.
- Reviewed annotations that take precedence over all automatic evidence.
- The same 14 post-processing transforms used by detector evaluation, rerunning automatic proposals on each transformed image.
- Storage-capped mask evaluation that streams Hugging Face Parquet or reads selected SBU ZIP members without unpacking the archive.
- Same-pass in-memory DINO feature transfer from PatchHead, capped at 512 MiB by default and 2 GiB maximum.
- Storage-capped SID_Set sampling, reviewer-agreement tools, and legacy DID/DINO adapters.

## Install

Core geometry:

```bash
cd physics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Automatic learned proposals and bounded mask evaluation:

```bash
python -m pip install -e '.[auto,eval]'
```

Add SID Parquet support only when needed:

```bash
python -m pip install -e '.[sid]'
```

The learned proposal stack uses locally cached model files. Keep `cache/`, model weights, datasets, and generated reports outside Git. The tested CLIPSeg, DINOv3-small, and torchvision model cache occupies roughly 0.75 GB; the two bounded evaluation sources occupy roughly 0.45 GB.

## Automatic physics-only demonstration

The deterministic demo images are tracked. From `physics/`, run the learned path:

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

After the artifacts are cached, add `--proposal-offline` for a repeatable no-network run. Use `--strict-proposal-models` if a missing learned artifact should fail instead of falling back.

For a zero-download constrained-scene demonstration:

```bash
physics-engine examples/demo_images/automatic_consistent.png \
  --auto-proposals \
  --proposal-mask-backend heuristic \
  --proposal-feature-backend appearance \
  --proposal-object-backend edges \
  --output outputs/automatic_fallback.json \
  --overlays-dir outputs/automatic_fallback_overlays \
  --pretty
```

Open the generated `cast_shadow` and `reflection` PNG overlays. They show proposed regions, object-to-shadow vectors, direct-to-reflected matches, and inlier/outlier geometry. The JSON records `evidence_origin: automatic_proposal`, proposal confidence, backend/model metadata, pair count, and safety-gate state.

`automatic_inconsistent.png` is also retained as an honest failure example: with the tested zero-shot defaults, CLIPSeg proposes no usable shadow region and DINO retains only two reflection matches, so both cues abstain. It demonstrates why the system must expose coverage and `not_applicable`, not merely successful examples.

## How automatic proposals work

1. **Find candidate regions.** CLIPSeg uses prompt ensembles to estimate shadow and mirror probability maps. The fallback uses conservative photometric and framed-quadrilateral priors.
2. **Create correspondences.** Shadow components are associated with nearby detected objects or foreground support, then reduced to contact/tip endpoints. Reflection points are matched across the mirror boundary with dense features, mutual-nearest-neighbour checks, similarity margin, separation, and spatial-diversity gates.
3. **Verify geometry independently.** Shadow vectors must agree with one projected light source/direction. Reflection connectors must agree with the normal vanishing point of one planar reflector. The proposal model does not decide whether its own pairs are geometrically consistent.
4. **Abstain when evidence is weak.** Fewer than three usable pairs are `not_applicable`. An automatic `inconsistent` result requires at least four pairs; exactly three can establish consistency but an apparent inconsistency is downgraded to `indeterminate`.

See [`docs/automatic_proposals.md`](docs/automatic_proposals.md) for architecture, exact evaluation commands, measured results, licenses, and failure modes.

## Unified PatchHead and physics inference

When the pooled PatchHead checkpoint is available, root `infer.py` can pass its same-forward DINO grid to reflections without a second DINO pass:

```bash
python infer.py \
  --image-dir path/to/images \
  --ckpt path/to/patchhead_pooled.pt \
  --out outputs/predictions.json \
  --with-physics \
  --physics-auto-proposals \
  --physics-proposal-mask-backend clipseg \
  --physics-proposal-feature-backend patchhead \
  --physics-proposal-object-backend torchvision \
  --physics-proposal-cache-dir cache/physics-auto \
  --physics-overlays-dir outputs/physics_overlays \
  --pretty
```

`patchhead` transfers dense features in memory only; they are not serialized into JSON. The default memory budget is 512 MiB. Use standalone `dinov3` or `appearance` when no pooled checkpoint is available.

One PatchHead forward pass still supplies the official image logit, CLS logit, and patch logits. Its final detector score remains:

```text
0.5 × (sigmoid(image_logit) + sigmoid(cls_logit))
```

Automatic physics remains an explanation sidecar and cannot overwrite this score.

## Checkpoint-backed validation

The supplied pooled checkpoint was validated on the existing storage-capped SID
sample (50 real, 50 full-synthetic, 50 tampered) and a 20-image WildFake
range-read pilot. Under matched settings, detector-only and integrated SID runs
had exact score, component-score, and verdict parity across all 150 images.
Real-versus-full-synthetic SID accuracy was 100/100 on this bounded sample;
tampered images were reported separately because they are outside the binary
training objective. The WildFake pilot achieved 20/20 clean and 99.64% mean
accuracy over the 14 transforms.

`physics-checkpoint-eval` reproduces detector, parity, cue-coverage, and weak
tamper-localization metrics. `harness/patchhead_robustness.py` measures PatchHead score,
patch-map, and same-pass DINO-grid stability. See
[`docs/checkpoint_validation.md`](docs/checkpoint_validation.md) for hashes,
commands, exact results, and limitations.

## Reviewed annotations and precedence

The original annotation workflow remains available for calibration, correction, or evidence review:

```bash
physics-engine examples/demo_images \
  --annotations examples/demo_annotations.json \
  --output outputs/reviewed_demo.json \
  --overlays-dir outputs/reviewed_overlays \
  --pretty
```

Serve the browser annotator from `physics/` with `python -m http.server 8765 --bind 127.0.0.1`, then open `http://127.0.0.1:8765/tools/annotator.html`. Reviewed pairs or explicit applicability decisions override automatic proposals cue by cue; automatic mode only fills missing evidence.

## Transformation battle test

```bash
physics-battle-test examples/demo_images/automatic_consistent.png \
  --auto-proposals \
  --proposal-mask-backend clipseg \
  --proposal-feature-backend dinov3 \
  --proposal-object-backend torchvision \
  --proposal-cache-dir ../cache/physics-auto \
  --proposal-offline \
  --output outputs/automatic_battle_test.json
```

The harness evaluates clean plus JPEG 90/70/50/30, blur 0.5/1/2, resize 1/2 and 1/4, noise 0.02/0.05/0.10, colour jitter, and an 80% center crop. Automatic regions and pairs are regenerated after every transform.

In the local deterministic demo run, perspective retained applicability on 13/14 transforms, shadows on 12/14, and reflections on 13/14. There were zero hard consistent↔inconsistent flips after the four-pair safety gate. These are fixture-level stability results under deliberately permissive POC acceptance thresholds, not a natural-image benchmark.

## Bounded mask evaluation

`physics-mask-eval` validates only the region proposal stage. It enforces a configurable source cap with a hard 50 GiB maximum and does not unpack the complete dataset.

The deterministic 24-image smoke tests produced:

| Cue/source | Backend | Threshold | Macro IoU | Dice | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Shadow / SBU test | CLIPSeg | 0.40 | 0.452 | 0.566 | 0.855 | 0.514 |
| Shadow / SBU test | OpenCV fallback | 0.48 | 0.040 | — | — | — |
| Mirror / PMD test | CLIPSeg | 0.54 | 0.296 | 0.366 | 0.534 | 0.374 |
| Mirror / PMD test | OpenCV fallback | 0.54 | 0.102 | — | — | — |

The CLI can write overlays where green is correct mask overlap, red is a false positive, and blue is a missed target. These numbers measure semantic mask quality only—not object-shadow association, reflection geometry, or AI-image detection accuracy.

## Storage-capped SID_Set study

The existing 150-image, three-shard SID sample now has real pooled-PatchHead and
integrated automatic-physics results. Perspective was applicable on 106/150
images with no displayed inconsistency; one automatic shadow was applicable and
consistent; reflection abstained on every image. Five candidate mirror regions
exercised real same-pass DINO feature transfer before correspondence gates
abstained. See [`docs/sid_pilot.md`](docs/sid_pilot.md) and
[`docs/checkpoint_validation.md`](docs/checkpoint_validation.md).

## Verification

From the repository root:

```bash
scripts/checkpoint_independent_tests.sh
```

Or run the suites directly:

```bash
cd physics
../.venv/bin/python -m unittest discover -s tests -v
cd ..
.venv/bin/python -m unittest patchhead.tests.test_inference patchhead.tests.test_unified_infer -v
```

The version 0.6.1 implementation passes 86 physics tests and 18 PatchHead/unified-inference tests.

## Remaining limitations

- Zero-shot CLIPSeg masks are useful but moderate rather than benchmark-leading; merged shadows, dark objects, soft illumination, and partial mirrors remain common failure modes.
- Generic object boxes do not identify exact feet/contact surfaces. The contact/tip association remains a geometric proposal with confidence gates, not object-level ground truth.
- Reflections currently target planar mirrors; water, curved metal/glass, windows, repeated objects, and screen content can invalidate matching assumptions.
- Multiple lights, diffuse illumination, uneven terrain, and occlusion can invalidate the single projected-light shadow test.
- An image can be generated yet physically coherent, or real/composited yet physically inconsistent. Physics is explanatory evidence, not class truth.
- The bounded 24-image mask samples and synthetic transformation fixture are engineering smoke tests, not calibrated held-out validation.
- PatchHead patch maps show only weak SID tamper localization (mean patch AUC 0.582) and must not be presented as segmentation or causal attribution.
- A trained shadow-instance/contact head or mirror-instance head could replace the zero-shot providers through the existing provider interfaces, but training and calibration require suitable labels and compute.

See [`docs/architecture.md`](docs/architecture.md), [`docs/schema.md`](docs/schema.md), [`docs/automatic_proposals.md`](docs/automatic_proposals.md), [`docs/dino_integration.md`](docs/dino_integration.md), [`docs/checkpoint_validation.md`](docs/checkpoint_validation.md), [`docs/reviewer_protocol.md`](docs/reviewer_protocol.md), and [`docs/next_steps.md`](docs/next_steps.md).
