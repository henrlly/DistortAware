# SID_Set explanation-safety study

## Storage-safe acquisition

The supplied local `SID_Set` clone contains Git LFS pointer files rather than Parquet payloads. Pulling the roughly 140 GB dataset wholesale would violate the requested working limit. Three validation shards were therefore downloaded individually with the Hugging Face CLI at pinned dataset revision `dc03ead57929879319ce30a82bfcfb8d317b10bd`:

| Shard | Bytes | SHA-256 |
|---|---:|---|
| `validation-00000-of-00034.parquet` | 477,663,216 | `56cf2dd5c6a72a158f91aee4c5e06154f5d0a0903eb258a3de11eedded82c2a6` |
| `validation-00001-of-00034.parquet` | 505,844,042 | `1447bbd98adf7eda68fca5615560c6b1de34c8e30157ff6b34ebd1e015a18042` |
| `validation-00002-of-00034.parquet` | 513,296,014 | `5b177a48b456d862fb090ffe88a7396f8ce9473031ba40fea1359d6ff5ac017a` |

The exact sources total 1,496,803,272 bytes (1.394 GiB). No archive was expanded and no bulk LFS pull occurred. Shards and extracted records remain under ignored cache/output paths.

## Sampling protocol

`physics-sid-pilot` streams 32-row Parquet batches and maintains a deterministic reservoir for each native label:

- `0`: real;
- `1`: full synthetic;
- `2`: tampered.

Seed 2026 selected 50 examples per label. The three shards contributed 61, 44, and 45 selected records, showing that the sample is genuinely multi-shard. The manifest retains shard hash, source row, image ID, label, dimensions, extracted-record hash, dataset revision, and storage limits.

| Storage | Actual | Configured cap | Hard code cap |
|---|---:|---:|---:|
| Selected Parquet source | 1.394 GiB | 2 GiB for this run | 50 GiB |
| Extracted images + masks | 83.0 MiB | 1 GiB for this run | 10 GiB |

An existing capped workspace can be re-evaluated after engine changes without reading or expanding Parquet again:

```bash
physics-sid-pilot --workspace outputs/sid_pilot --evaluate-existing
```

## Perspective explanation findings

No reviewed shadow or reflection correspondences were supplied, so both cues correctly remained `not_applicable`. Version 0.4.0 perspective results were:

| Label | n | Applicable | Statuses | Displayed inconsistent / applicable | Wilson 95% CI | Mean violation |
|---|---:|---:|---|---:|---:|---:|
| Real | 50 | 40 (80%) | 35 consistent, 5 indeterminate, 10 not applicable | 0% | 0.0%–8.8% | 0.113 |
| Full synthetic | 50 | 31 (62%) | 29 consistent, 2 indeterminate, 19 not applicable | 0% | 0.0%–11.0% | 0.109 |
| Tampered | 50 | 35 (70%) | 32 consistent, 3 indeterminate, 15 not applicable | 0% | 0.0%–9.9% | 0.065 |

Interpretation:

- Zero displayed inconsistencies is encouraging for explanation safety, but each zero has a nonzero confidence-interval upper bound.
- Similar low mean scores across labels confirm that perspective is not a substitute classifier.
- Local tampering often preserves global scene geometry; a low score on the tampered class is expected.
- Conditional applicability supports the sidecar design: missing or ambiguous geometry should not influence the primary verdict.

## Version 0.5.0 automatic-proposal smoke

To check the new execution path without expanding or re-reading SID shards, six images were selected from the already-capped workspace: two real, two full-synthetic, and two tampered. CLIPSeg/DINO automatic mode made no applicable cast-shadow or reflection claim on any of the six; perspective was applicable and consistent.

This is evidence of conservative abstention on six examples, not a measurement of accuracy, recall, or class separation. SID labels are image-origin labels rather than shadow/mirror masks or correspondence ground truth. Proper automatic evaluation still requires a frozen reviewed pair set or a purpose-built cue dataset.

## Tamper-mask diagnostic

Sparse perspective residuals and non-empty SID masks were jointly evaluable on 33 tampered images:

| Metric | Mean |
|---|---:|
| Fraction of physics-residual patches inside tamper mask | 0.202 |
| Fraction of tamper mask covered by physics residuals | 0.153 |
| Sparse-grid IoU | 0.080 |

These are line-residual/mask association diagnostics, not segmentation metrics for a trained localizer. Globally consistent cues do not become suspicious explanations merely because a robust fit contains isolated residual lines.

## Full transformation study

All 150 images were evaluated under clean plus 14 post-processing conditions, for 2,250 engine runs. The retained acceptance report records:

| Metric | Result | Gate |
|---|---:|---:|
| Applicability retention | 88.3% | ≥90% — **fail** |
| Hard consistent↔inconsistent flips | 0 | 0 — pass |
| Mean absolute score drift | 0.025 | ≤0.20 — pass |
| Maximum absolute score drift | 0.315 | ≤0.20 — **fail** |

Heavy noise is the principal applicability failure: retention is 58.5% at noise σ=0.05 and 25.5% at σ=0.10. This is conservative abstention under degraded evidence, not a false physical-inconsistency claim. JPEG quality 30 retains 88.7%; every other transform retains at least 94.3% except noise σ=0.02, also 88.7%.

The largest drift changes a clean `consistent` score of 0.185 to a transformed `indeterminate` score of 0.5 after the multi-view safety gate finds only one applicable crop view. It is intentionally not a hard flip. The failure remains in the report rather than being threshold-tuned away on this sample.

## Independent review queue

The generated `review_queue.json` contains scene type, structured geometry, visible shadows, planar reflection, screenshot/composite, and CGI/illustration fields. A stable hash selects 10 records per label (30 total) as a balanced 20% double-review starting set.

Two reviewers should annotate independently with distinct IDs and without seeing automatic proposals. If the starting set does not contain enough applicable shadows or reflections, continue reviewing until at least 20% of cases considered applicable by either reviewer have two reviews. Run `physics-review-agreement`, preserve both raw exports, and evaluate automatic pair precision/endpoint error against a separate adjudicated copy before presenting proposal quality.

## Checkpoint-backed continuation completed

The same manifest was reused for image-identical pooled-PatchHead experiments:

1. detector-only and integrated scores/components/verdicts matched exactly;
2. genuine patch maps and same-pass DINO grids were exported;
3. all 50 tamper masks were compared with the weak patch map;
4. a balanced six-image slice measured patch/DINO stability under all 14 transforms;
5. automatic physics coverage and errors were summarized per native label.

Mean tamper patch AUC was 0.582 and top-area IoU 0.201, which is too weak for a
segmentation claim. Physics produced no displayed SID inconsistency and no
applicable reflection. See [`checkpoint_validation.md`](checkpoint_validation.md)
for the complete protocol and results. Independent reviewed correspondence
ground truth remains necessary to evaluate shadow/reflection proposal accuracy.
