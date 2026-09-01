# Browser product validation record

Validation date: 2026-09-01. This record covers the local review branch only;
it is not an accuracy benchmark and does not supersede the bounded SID/WildFake
results in [`physics/docs/checkpoint_validation.md`](../physics/docs/checkpoint_validation.md).

## Artifact identity

| Artifact | Validated value |
|---|---|
| Pooled PatchHead SHA-256 | `828fac3ba5c5b814a1ada36477b36848ab4c8366e040e68fff9f4c9fe14b6989` |
| Checkpoint threshold | `0.785` |
| Backbone | `vit_large_patch16_dinov3.lvd1689m` |
| Cached backbone snapshot | `30c1109559f65dea34316b0d4842d35c5771fe11` |
| Service schema | `0.2.0` |
| Service physics profile | `learned` (CLIPSeg + torchvision + same-pass DINO/appearance fallback) |
| Residual artifact checkpoint SHA-256 | `b9ea83ddcecd181310ed531093a138ac91b43b33ac9a49a38e9fbca80726964b` |
| PatchHead compatibility | Released binary-head adapter; score/weights unchanged |
| Inference device for this smoke | CPU |

The local PyTorch build reported MPS as unavailable on the automated host, so
an explicitly forced `--device mps` invocation failed before opening the port.
Automatic device selection correctly chose CPU. This is an environment/runtime
limitation rather than a checkpoint or API failure.

## Real-checkpoint HTTP smoke

The service was started offline with the external checkpoint, cached backbone,
cached learned physics models, and bundled residual checkpoint. Requests used
an extension-shaped origin. The original deterministic geometry smoke remains
valid; the current schema was additionally exercised on the evaluation-only
WildFake wall.

| Check | Result |
|---|---|
| Missing bearer token | HTTP 401 |
| Authenticated health | `ready`; checkpoint hash/profile reported |
| CORS | Exact requesting `chrome-extension://…` origin echoed |
| First detector request | Verdict `is_aigc=true`; signal `0.788172`; 250.2 ms |
| Same crop repeated | Exact verdict parity; cache hit; 0.1 ms |
| Physics requested | Exact primary verdict parity; 429.6 ms total |
| Physics result | `consistent`; perspective and reflection applicable |
| Patch explanation | 16×16 weak patch-evidence grid summary |
| DINO/physics object | Present; non-causal; not applicable for globally consistent cues |
| Physics influence | Explicitly `false` |
| Artifact influence | Explicitly `false` |
| Internal request paths | Stripped from the public response |

The fixture's thresholded result is only a pipeline observation. The fixture is
synthetic test art, not a representative benchmark item, and the score is not a
probability.

## Current 12-image WildFake wall smoke

The exact demonstration-only subset contains six COCO val2017 reals and six
DALL-E Advanced images. One authenticated request per image enabled both
explainers. The complete run took 20.0 seconds on CPU after service startup
(1.04–3.23 seconds per image, including first-use model warm-up).

| Check | Result |
|---|---|
| PatchHead thresholded result vs wall label | 12/12 on this selected wall; not a benchmark claim |
| Perspective statuses | 5 consistent, 3 indeterminate, 4 not applicable |
| Cast-shadow statuses | 1 indeterminate (three pairs), 11 not applicable |
| Reflection statuses | 2 inconsistent, 10 not applicable |
| Same-pass DINO fallback | 2 mirror cases; DINO had too few matches and disclosed appearance fallback supplied 18/13 matches |
| Reflection/label observation | Both displayed inconsistencies were DALL-E examples; sample too small for accuracy inference |
| Residual sidecar class agreement | 6/12 (4/6 AIGC, 2/6 real), directly illustrating why it has no verdict weight |
| Verdict noninterference | Physics and artifact flags both `false` for every response |

The physics improvement is primarily better proposal visibility and two
testable reflection cases, not universal coverage. Every abstention now retains
its region count, accepted/required pair count, selected feature backend, and
plain-language reason in the extension.

## Checkpoint-independent contracts

- 12/12 browser-service unit/HTTP tests passed with a fake PatchHead runtime.
- 12/12 extension helper/manifest/demo-wall tests passed under Node.
- 86/86 physics and 18/18 PatchHead/unified-inference tests passed.
- All five extension JavaScript entrypoints passed `node --check`.
- Ruff and `git diff --check` passed for the new Python/source changes.

The service tests cover auth, origin controls, health, payload validation,
bounded cache behavior, official score/verdict preservation, physics/artifact
noninterference, and explanation-aware cache boundaries. The extension tests
cover loopback URL restrictions, numeric setting bounds, whole-page scroll
planning, candidate filtering/ranking/deduplication, fail-closed result wording,
least-privilege permissions, entrypoint presence, detector-blind wall labels,
and absence of remote executable scripts.

## Rendered demo-wall check

The tracked demo page was served over loopback and inspected in a real browser.
All image assets loaded and the layout rendered correctly. Browser geometry
reported three 525×320 `<img>` regions, one 527×361 CSS background, and one
32×32 decoration. The default 96-pixel minimum therefore selects the four
intended regions and rejects the decoration.

The unpacked extension was also reloaded and exercised in Brave against the
real local service. With whole-page mode, a 12-image limit, learned physics, and
residual artifacts enabled, one click traversed four overlapping viewports,
found all 12 WildFake-wall images, restored the original scroll position, and
reported 12 scanned / 6 AIGC signals / 6 no-signals / 0 errors. Every popup row
showed the PatchHead signal plus applicability-aware physics and residual
evidence marked “explanation only.” The 6/6 split is an observation on this
selected wall, not an accuracy estimate. A representative real social feed is
still a manual release acceptance step because its account/session state should
not be automated or recorded in the repository.

## Approved extension archive and offline PatchHead replay

The switchable Manifest V3 extension was packaged again on 2026-09-01 and
approved for the local demo only. Its exact archive is
`dist/prismguard-browser-extension.zip` (21,909 bytes), SHA-256
`a24a5b017e880baec44e3f9888997f3ae0efb2228b28f59d818b103edd1f0f06`.
The ZIP passed `unzip -t`; it contains only the manifest, service worker,
content script, shared library, popup, and options files. Demo/test pages are
not installed with it.

The official timm DINOv3-L snapshot was fetched at the pinned revision above
and its `model.safetensors` independently matched the expected SHA-256
`45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941`.
The service then started with `HF_HUB_OFFLINE=1`, the sealed pooled PatchHead,
`--physics-profile off`, and `--artifact-profile off`. Authenticated health
reported the expected backbone, checkpoint digest, pooled dataset, binary-head
compatibility adapter, and threshold.

Two tracked synthetic geometry fixtures were sent through the real HTTP
endpoint. They produced classifier signals `0.7881718415` and `0.9203067846`
with 16x16 weak patch-evidence summaries. These observations prove that the
downloaded backbone, pooled head, authenticated service, response schema, and
extension-facing score contract execute together. The fixtures are neither a
real/fake benchmark nor evidence of generalization.

The current extension suite passed 17/17 Node tests. The documented
checkpoint-independent Python core passed 11/11 tests, including PrismGuard
diagnostic noninterference and rejection of smoke bundles. The optional
heuristic-physics test could not import OpenCV because the host has a
pre-existing NumPy 2.x/OpenCV 1.x ABI mismatch; the approved demo run has
physics disabled and is unaffected. A production PrismGuard profile remains
unavailable and fails closed because no release-eligible DINO OOF bundle has
yet been produced. It must not be replaced by the CIFAKE proxy or smoke bundle.
