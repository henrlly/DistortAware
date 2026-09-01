# Local browser inference service

This package exposes the repository's pooled DINOv3 PatchHead detector to the
companion Chromium extension over an authenticated loopback API. PatchHead is
the only verdict source. Optional physics output and the repository's compact
RGB/high-pass residual U-Net are attached as explanation evidence and cannot
change the score, threshold, or verdict.

The service never fetches URLs. It accepts rendered image bytes from the
extension, writes each request to an isolated temporary directory, runs local
inference, and deletes the temporary file immediately.

It also accepts a sealed PrismGuard bundle through a separate adapter. In that
mode the extension score is exactly PrismGuard's calibrated pure-DINO
probability; physics, residuals, and other forensic diagnostics are disabled in
this first adapter and cannot affect prediction.

## 1. Prepare the runtime

From the repository root, use an environment containing PyTorch, timm, NumPy,
Pillow, and the physics package:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision timm opencv-python './physics[auto]'
```

The small PatchHead checkpoint is not sufficient by itself: timm also needs
the frozen DINOv3 ViT-L/16 backbone (about 1.1 GB). The first online model load
downloads it. To use a repository-local cache and then run offline:

```bash
export HF_HUB_CACHE="$PWD/cache/patchhead-backbone"
# Leave this unset for the first model download.
export HF_HUB_OFFLINE=1
```

The available pooled checkpoint uses the released one-logit binary head. PR #5
moved the training model to three classes for distortion-aware work, but no
matching pooled checkpoint/versioned interactive score contract is available.
The stable inference runtime therefore reconstructs the checkpoint's original
one-output layers and preserves its exact sigmoid score formula. It does not
convert, retrain, or edit checkpoint weights. A future three-class pooled
checkpoint is rejected until its scoring contract is integrated explicitly.

Checkpoint and cache paths are ignored by Git. Do not copy model weights into
the extension.

## 2. Start the service

PrismGuard mode (preferred once the approved DINO checkpoint, license ledger,
and trained bundle exist):

```bash
.venv/bin/python -m browser_product.service \
  --prismguard-bundle /absolute/path/to/detector-bundle.json \
  --prismguard-checkpoint /absolute/path/to/dinov3-vitl16-checkpoint \
  --prismguard-license-ledger /absolute/path/to/checkpoint-licenses.json \
  --prismguard-root /absolute/path/to/prismguard
```

The current PrismGuard repository deliberately has no release-eligible trained
bundle yet, so this command fails closed until the gated DINO training run is
complete. The browser adapter never falls back to the rejected handcrafted
smoke model or the older PatchHead score.

For a checkpoint-free extension wiring demo only:

```bash
python3 -m browser_product.service \
  --demo-fixture \
  --port 8767 \
  --api-token prismguard-demo-only
```

This emits a deterministic pixel-variation signal and marks health and every
result `plumbing_only_no_aigc_performance_claim`. It is not a detector and must
not be used in evaluation. The extension accepts it only when its explicit
**Wiring demo fixture** profile is selected.

Fast detector-only mode, recommended for scanning a feed:

```bash
.venv/bin/python -m browser_product.service \
  --checkpoint /absolute/path/to/patchhead_pooled.pt
```

The service enables the bundled residual artifact sidecar by default. Its
386,932-parameter checkpoint is tracked under
`filter_based_approach/models/mask_classifier.pt`; no extra download is needed.
The extension decides per request whether to run it. Use
`--artifact-profile off` when detector-only latency is required. Artifact class
scores and the weak evidence-mask summary are uncalibrated and explanation-only.

The default auto-selects CUDA, MPS, or CPU. Use `--device` only when you have a
reason to override that choice. Startup prints both the URL and a random bearer
token. Keep the terminal open and paste that token into the extension's
settings page.

For automatic perspective, cast-shadow, and planar-reflection explanations:

```bash
.venv/bin/python -m browser_product.service \
  --checkpoint /absolute/path/to/patchhead_pooled.pt \
  --physics-profile heuristic
```

The `heuristic` profile uses local classical proposals plus the same-pass DINO
feature grid and does not require extra model downloads. `learned` additionally
requests CLIPSeg/torchvision proposal models and may download their weights;
use it only after preparing that cache. Physics is still run only when the
extension's “Request physics explanations” setting is enabled.

For repeatable demos, supply a token through the environment rather than a
command-line argument:

```bash
export AIGC_EXTENSION_TOKEN="replace-with-a-long-random-demo-token"
.venv/bin/python -m browser_product.service \
  --checkpoint /absolute/path/to/patchhead_pooled.pt
```

## API

Both endpoints require `Authorization: Bearer <token>`:

- `GET /v1/health` returns detector/checkpoint provenance, service limits, the
  configured physics profile, and artifact-sidecar readiness.
- `POST /v1/analyze` accepts a raw PNG, JPEG, or WebP body. Optional headers
  are `X-AIGC-Explain: true`, `X-AIGC-Artifact: true`, and
  `X-AIGC-Source-Kind: viewport_capture`.

Example:

```bash
curl --fail-with-body \
  -H "Authorization: Bearer $AIGC_EXTENSION_TOKEN" \
  -H "Content-Type: image/png" \
  --data-binary @physics/examples/demo_images/automatic_consistent.png \
  http://127.0.0.1:8765/v1/analyze
```

The response intentionally calls `aigc_score` an uncalibrated classifier
signal. It is not a probability or authenticity proof. `patch_evidence` is
weak image-supervised localization, not a segmentation mask. Viewport crops
also inherit browser scaling, overlays, and partial-image limitations.
When physics is requested, `dino_physics_alignment` reports non-causal spatial
association between weak PatchHead regions and physics outlier geometry; it
does not claim that physics caused the detector score.
When residual evidence is requested, `explanation.artifact` reports a
three-class sidecar signal and a bounded 8×8 evidence-mask summary. Recorded
cross-dataset WildFake accuracy is 74%, while SID mask IoU is only 15.5%; the
API therefore sets `artifact_affects_detector_score` to `false` and describes
the mask as weak localization. Rendered crops do not retain trustworthy EXIF or
C2PA data, so missing provenance is explicitly `unknown`, never “authentic.”

## Safety boundaries

- Default bind is `127.0.0.1`; non-loopback binds print a warning.
- A bearer token is mandatory, and CORS accepts only extension or loopback
  origins.
- Upload size, decoded pixels, image format, and response JSON are bounded or
  validated.
- Model execution is serialized; a bounded SHA-256 LRU avoids duplicate work.
- No source URL, page cookie, caption, or browsing history enters the API.
- The extension token is stored in Chromium's local extension storage. Treat
  it as a local secret and rotate it by restarting the service when needed.

## Tests

Checkpoint-independent contract tests use a fake PatchHead runtime:

```bash
.venv/bin/python -m unittest discover -s browser_product/tests -v
```

The acceptance test for a release candidate should additionally start this
service with the external pooled checkpoint, call both endpoints, verify a
cache hit on a repeated crop, and manually scan the extension demo wall.
The latest local evidence is recorded in [`VALIDATION.md`](VALIDATION.md).
