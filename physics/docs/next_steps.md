# Integration readiness and remaining work

Review date: 2026-09-01 (Asia/Singapore)

## Current state

Physics version 0.6.1 is checkpoint-backed and integration-ready as an
explanation sidecar. The release architecture remains:

```text
PatchHead score and verdict  -> primary decision
PatchHead patch/DINO output  -> optional weak spatial/proposal evidence
Physics constraints          -> optional abstention-aware explanation
DID                          -> comparison/ablation
```

Physics never changes the PatchHead score, threshold, or verdict. Checkpoints,
frozen backbones, datasets, and caches remain outside Git. Curated compact
result reports and the documented residual-sidecar checkpoint are the deliberate
exceptions.

## Completed engineering work

### Scene-consistency engine

- Automatic straight-line extraction, vanishing-point estimation, projection
  safety gates, crop stability, and reviewed structural regions.
- Automatic CLIPSeg/OpenCV cast-shadow and planar-mirror proposals.
- Automatic object-contact/shadow-tip association using torchvision boxes or
  local edge support.
- Automatic direct/reflected matching using standalone DINOv3, same-pass
  PatchHead DINO tokens, or appearance descriptors.
- Disclosed appearance fallback when same-pass DINO is insufficient, while
  retaining the independent geometry and four-pair inconsistency gates.
- Independent projected-light and planar-reflector geometry; proposal models do
  not judge their own consistency.
- Reviewed evidence and explicit applicability decisions override automation.
- Confidence/model provenance, assumptions, limitations, regions, pairs,
  inliers/outliers, and human-visible overlays.
- Safety abstention below three pairs and a four-pair minimum for automatic
  definitive inconsistency.

### Primary-model integration

- The supplied pooled PatchHead checkpoint passes preflight and real inference.
- Its SHA-256, pooled tag, threshold, architecture, and pinned DINOv3-L snapshot
  are documented in `checkpoint_validation.md`.
- The existing patch logits and optional final DINO grid are exported from one
  forward pass without altering PatchHead's model or score.
- Same-pass features are float16, in memory only, and bounded by a configurable
  512 MiB default / 2 GiB CLI maximum.
- Physics-side feature provenance now retains the source checkpoint hash,
  backbone, dataset tag, dtype, grid shape, and `score_independent` declaration.
- Detector-only and integrated 150-image SID runs have exact score, component,
  and verdict parity under matched settings.

### Storage-safe validation

- SID_Set: 150 images (50 per native label) from three pinned Parquet shards;
  1.394 GiB source and 83.0 MiB extracted; no bulk LFS pull.
- WildFake: 10 COCO real and 10 ADM fake images acquired by range reads; about
  1.3 MB and no archive download.
- Automatic-mask smoke sources remain bounded to roughly 448 MiB.
- Model caches plus supplied external checkpoints remain below the recommended
  10 GiB active working limit; the 50 GiB hard cap was never approached.

### Measured results

- SID real/full-synthetic bounded sample: 100/100 correct, ROC AUC 1.0.
- SID tampered diagnostic: 2/50 AIGC alerts; label 2 is outside the pooled binary
  objective and is not folded into headline accuracy.
- SID physics: zero errors; 106/150 perspective-applicable, one applicable
  automatic shadow, no applicable reflection, and no displayed inconsistency.
- SID weak tamper patch-map AUC: 0.582 mean across 50 masks; this is not
  segmentation and is documented as a limitation.
- WildFake 20-image pilot: 20/20 clean; 99.64% mean over 14 transforms, with one
  blur-2.0 fake miss.
- Balanced six-image SID patch/DINO transform check: zero verdict flips, maximum
  score drift 0.00637, mean patch correlation 0.989, mean token cosine 0.980.
- Existing 24-image SBU/PMD mask smokes and deterministic automatic-physics
  transform fixture remain available as proposal-stage evidence.
- Conservative CLIPSeg/prior fusion now reaches 0.4866 shadow IoU on the bounded
  SBU sample and 0.2981 mirror IoU on the bounded PMD sample without changing
  the three-pair applicability or four-pair inconsistency safety gates.

See `checkpoint_validation.md`, `sid_pilot.md`, and `automatic_proposals.md` for
protocols, exact caveats, and reproduction commands.

## Remaining work that needs external evidence or compute

### 1. Freeze independent correspondence ground truth

Two reviewers should independently annotate a balanced natural-scene set
without seeing detector scores, class labels, automatic proposals, or each
other's work. Preserve raw reviews and a separate adjudicated copy.

Measure:

- cue-applicability precision/recall;
- shadow ownership and contact/tip endpoint error;
- reflection correspondence precision and endpoint error;
- pair count and scene coverage;
- proposal-confidence calibration;
- final cue-status agreement;
- false displayed inconsistency on real images.

SID origin labels and mask IoU cannot answer these geometry questions.

### 2. Broaden natural-scene stress testing

Add a license-reviewed, laptop-safe set stratified by hard/soft shadows,
single/multiple lights, indoor/outdoor contact geometry, planar mirrors,
windows/screens/water/curved reflectors, repeated textures, and severe
post-processing. Continue selective acquisition below 10 GiB recommended and
50 GiB hard limits.

### 3. Compare trained proposal heads if resources permit

A trained shadow-instance/contact head and mirror-instance head could replace
the zero-shot providers through the existing interfaces. This requires suitable
licenses, labelled correspondences, and GPU training. Any replacement must keep
confidence/provenance and feed the same independent geometric verifier.

## Presentation and release discipline

The browser API and extension now surface model/checkpoint provenance and
explanation applicability while preserving the primary verdict. For the final
presentation and any later release:

- Keep PatchHead as the primary decision and physics as an explanation-only
  sidecar.
- Show one successful automatic example, one correct abstention, and one known
  failure rather than implying universal coverage.
- Label patch maps as weak evidence and physics scores as constraint violations.
- Do not ship external checkpoints or datasets in Git; document acquisition and
  local cache setup.

## Acceptance matrix

| Gate | Requirement | State |
|---|---|---|
| Automatic proposal contracts | Auditable automatic regions/pairs | **Pass** |
| Automatic inconsistency safety | Fewer than four pairs cannot assert inconsistency | **Pass** |
| Reviewed precedence | Reviewed evidence/decisions override proposals | **Pass** |
| Real pooled primary | Checkpoint preflight and real inference | **Pass** |
| Primary noninterference | Matched detector-only/integrated score and verdict parity | **Pass: exact on 150 SID images** |
| Same-pass feature provenance | Real DINO grid, checkpoint/model traceability | **Pass** |
| SID bounded evaluation | Binary, tamper, cue coverage, error reporting | **Complete bounded study** |
| WildFake bounded evaluation | Range-read clean and 14-transform run | **Complete 20-image pilot** |
| Patch/dense stability | Real checkpoint under 14 transforms | **Complete six-image engineering check** |
| Regression suites | Checkpoint-independent physics and unified contracts | **Pass: 83 + 18 tests** |
| Pair/endpoint accuracy | Frozen independent point correspondence ground truth | **Pending external review** |
| Natural-scene coverage | Broad held-out mirrors/shadows | **Pending** |
| Source hygiene | No external PatchHead/physics checkpoints, model caches, datasets, credentials, or bulk outputs tracked; the documented compact residual sidecar checkpoint is the deliberate exception | **Pass** |

## Deliberate limitations

- Zero-shot masks have moderate coverage and can merge shadows or confuse dark
  objects, windows, screens, and framed pictures.
- Generic boxes do not provide exact feet/contact surfaces or shadow ownership.
- DINO similarity is correspondence evidence, not semantic proof.
- Single-light and planar-reflector assumptions exclude many valid real scenes.
- Automatic abstention is frequent on uncontrolled images; this is preferable
  to fabricated certainty.
- Generated images can be physically coherent, and real composites can be
  inconsistent. Physics is not the authenticity classifier.
- Bounded pilots demonstrate implementation behavior, not deployment accuracy.
