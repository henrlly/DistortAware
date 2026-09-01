# Changelog

## 0.6.1 — 2026-09-01

- Fused the pinned zero-shot CLIPSeg masks with conservative photometric shadow and framed-mirror priors instead of discarding strong deterministic proposals.
- Improved the bounded 24-image SBU shadow macro IoU from 0.4524 to 0.4866 and the bounded PMD mirror macro IoU from 0.2957 to 0.2981 under the unchanged thresholds.
- Preserved the minimum-three-correspondence applicability rule and four-pair automatic-inconsistency safety gate; no extra verdict weight or forced cue applicability was introduced.
- Added an auditable local-appearance fallback when same-pass PatchHead DINO supplies too few reflection matches; the DINO attempt, pair counts, selected backend, and fallback limitation remain in measurements.
- Added regression coverage for fusion bounds and documented learned-profile behavior on the 12-image WildFake demonstration wall.

## 0.6.0 — 2026-08-31

- Validated the supplied pooled PatchHead checkpoint end to end on bounded SID_Set and WildFake samples while keeping all model/data artifacts outside Git.
- Preserved PatchHead score, component scores, threshold, and verdict exactly when the physics sidecar is enabled under matched inference settings.
- Added checkpoint/model provenance to in-memory same-pass DINO grids without changing PatchHead core behavior.
- Added `physics-checkpoint-eval` for binary metrics, sidecar coverage/errors, detector-only parity, and explicitly weak SID tamper localization.
- Moved PatchHead score, patch-map, and DINO-grid stability checks to the harness.
- Made `patchhead/evaluate.py` portable on macOS through a configurable DataLoader worker count.
- Recorded a storage-capped 150-image SID validation, 20-image WildFake range-read pilot, and six-image patch/dense transform study in `docs/checkpoint_validation.md`.

## 0.5.0 — 2026-08-31

- Added fully automatic cast-shadow and planar-reflection proposal paths while keeping reviewed evidence as the higher-priority override.
- Added optional CLIPSeg shadow/mirror masks, DINOv3 dense reflection features, and torchvision object proposals with lazy loading, offline caches, and conservative fallbacks.
- Pinned the default CLIPSeg and standalone DINOv3 Hub revisions and included revision provenance in automatic outputs.
- Added automatic shadow contact/tip association and mutual-nearest-neighbour reflection matching with confidence, margin, saliency, separation, and diversity gates.
- Kept proposal generation separate from projected-light and planar-reflector geometry, with backend/model provenance in every automatic result.
- Added a four-pair safety gate: exactly three automatic pairs cannot assert a definitive physical inconsistency.
- Added automatic-region/pair overlays and deterministic consistent/inconsistent demonstration scenes.
- Added same-pass in-memory PatchHead DINO feature transfer without changing the ordinary forward contract or detector score.
- Added storage-capped SBU ZIP and Hugging Face Parquet mask evaluation with human-visible TP/FP/FN overlays.
- Added automatic-proposal transformation battle testing and a bounded six-image SID abstention smoke test.
- Validated the release with the full physics and PatchHead/unified-inference suites, changed-file lint, packaging, learned offline smoke runs, and diff hygiene checks.

## 0.4.0 — 2026-08-30

- Added checkpoint-independent PatchHead inference contracts, optional same-pass patch-map export, and optional physics attachment.
- Added lightweight CI that exercises primary-detector and physics contracts without downloading model weights or datasets.
- Added orientation-diffuse, panorama-frame, and multi-view crop-stability safety gates for perspective explanations.
- Added optional reviewed perspective regions to suppress semantic and decorative edges.
- Expanded the storage-capped SID study to 150 images from three pinned validation shards and added Wilson confidence intervals.
- Added deterministic independent-review targets, reviewer identity/applicability metadata, and Cohen-kappa/point-concordance reporting for shadow and reflection evidence.
- Added safe re-evaluation of existing capped SID workspaces without re-reading Parquet.
- Made explicit reviewer applicability decisions take precedence over retained point pairs.
- Preserved the primary PatchHead score and verdict; physics remains explanation evidence only.

## 0.3.0 — 2026-08-29

- Prioritized the official DINOv3 PatchHead detector while retaining DID compatibility.
- Added direct export of the official model's existing per-patch logits and component scores.
- Added strict DINO/physics joining and non-causal spatial-association metrics.
- Added three-panel DINO heatmap plus physics-residual rendering.
- Added a streaming, deterministic, storage-capped three-label SID_Set pilot.
- Downloaded and verified one 477.7 MB SID validation shard and extracted only 30.3 MiB.
- Ran a 60-image base pilot and nine-image/135-evaluation real transformation pilot.
- Documented the full-synthetic crop-drift failure instead of tuning it away.
- Expanded automated coverage for spatial maps, DINO contracts/rendering, and Parquet sampling.

## 0.2.0 — 2026-08-29

- Added official DID-output enrichment through `physics-merge`.
- Added the official 14-transform robustness harness and JSON/CSV/Markdown reports.
- Added mild line-image denoising, reducing worst fixture perspective drift under heavy noise.
- Applied EXIF orientation before analysis.
- Excluded nested overlay directories from recursive input discovery.
- Added empty-input failure and malformed/tiny-image regression coverage.
- Expanded automated coverage to 30 tests.

## 0.1.0 — 2026-08-29

- Initial applicability-aware perspective, cast-shadow, and planar-reflection MVP.
- Added structured JSON, overlays, deterministic fixtures, and local reviewed annotation UI.
