# AIGC Signal Inspector extension

For a concise installation, model-switching, and live-demo walkthrough, see
[`docs/BROWSER_EXTENSION_SETUP.md`](../docs/BROWSER_EXTENSION_SETUP.md).

This is a user-initiated Chromium Manifest V3 prototype for scanning rendered
visual regions in the active tab. It discovers `<img>` elements, video
posters/current frames, and CSS background images; filters decorative regions;
captures cropped viewport pixels; and sends them to the local service in
[`browser_product/`](../browser_product/README.md).

It does not run continuously and has no broad website host permission. It does
not fetch cross-origin image URLs: capturing rendered pixels makes CDN, blob,
and canvas-composited page layers less fragile while keeping page cookies and
source URLs out of the inference API.

## Load it locally

1. Start one or both local model services and copy each printed bearer token.
2. Open `chrome://extensions` (or the equivalent Chromium extensions page).
3. Enable Developer mode, choose **Load unpacked**, and select this
   `browser_extension/` directory.
4. Open the extension's settings. Choose **Original PatchHead** or
   **PrismGuard pure DINO**, enter the matching endpoint/token, choose either
   **Whole page (auto-scroll)** or **Current viewport only**, and select
   **Test connection**. Both profiles are saved independently, so switching
   models does not overwrite the other service configuration.
5. Visit a normal HTTP(S) page, open the extension, and choose **Scan page
   images** (or **Scan visible images** in viewport mode).

The extension adds a human-visible badge to each selected region and lists the
underlying classifier signal, viewport crop size, latency, weak patch summary,
physics proposal counts/reasons, and residual artifact evidence in the popup.
PatchHead alone determines the badge verdict. Physics and artifacts never vote.

When the local service is started with a sealed PrismGuard bundle, the same UI
instead displays the calibrated pure-DINO probability. That adapter has a hard
score boundary: diagnostics are unavailable rather than being allowed to alter
the DINO logit, calibration, threshold, or returned score.

The extension validates the selected model against `/v1/health` and validates
every inference response. A PatchHead endpoint cannot silently satisfy a
PrismGuard selection (or vice versa).

## Checkpoint-free wiring demo

To demonstrate capture, cropping, badges, model selection, and failure handling
before either external model artifact is available, start the explicit fixture:

```bash
python3 -m browser_product.service \
  --demo-fixture \
  --port 8767 \
  --api-token prismguard-demo-only
```

Select **Wiring demo fixture — no accuracy claim** in extension settings, test
the connection, then scan the deterministic demo wall below. Every fixture
response is labeled `plumbing_only_no_aigc_performance_claim`; it must never be
shown as model accuracy or an AIGC conclusion.

Build a deterministic installable archive with:

```bash
python3 scripts/package_browser_extension.py
```

For local development, load the unpacked `browser_extension/` directory so
edits are visible after pressing Reload on the extensions page.

Whole-page mode temporarily scrolls by overlapping viewports so lazy images can
render, captures each stable image/source once, then restores the starting
position. It stops at the configurable 1–60 image limit or a hard 40-viewport
safety cap, including on infinite feeds. Viewport mode remains available for a
faster, non-scrolling scan.

Protected browser pages, extension stores, and some DRM surfaces cannot be
scripted or captured. Keep the target tab active throughout capture. A scan
covers rendered crops—not full-resolution originals—and current video frames do
not provide temporal video forensics. Virtualized feeds may recycle one DOM
element for new content; the extension tracks source changes but cannot promise
complete coverage of every transient item.

## Deterministic demo wall

From the repository root:

```bash
python3 -m http.server 8088 --directory .
```

Open `http://127.0.0.1:8088/browser_extension/demo/` and scan it. The page has
large `<img>` fixtures, a CSS background, and one deliberately tiny decoration
that candidate filtering should ignore. These synthetic geometry images test
the end-to-end wiring only; they are not an accuracy benchmark.

## WildFake blind-label evaluation wall

Prepare a 12-image local subset (six per class) from the exact demonstration
strata declared for the hackathon—COCO val2017 (4,998 images) and DALL-E
Advanced/DALL-E 3 (8,843 images):

```bash
python3 scripts/prepare_wildfake_validation_demo.py
```

The sampler reads only selected ZIP members over HTTP ranges; it does not
download either full archive. Generated images and their provenance manifest
land under ignored `data/wildfake_validation_demo/`. They are explicitly for
demonstration/evaluation only and must never be copied into a training split.

With the same local HTTP server running, open
`http://127.0.0.1:8088/browser_extension/wildfake_demo/`. Each ground-truth label
is shown in a sibling caption below its image. The `<img>` itself has a neutral
filename and neutral alt text, and the extension captures only the rendered
image rectangle, so the detector does not receive the answer. Scan and compare
each badge with its caption. Whole-page mode can cover the entire wall in one
bounded run while the detector remains blind to the labels.

## Tests

```bash
node --test browser_extension/tests/*.test.cjs
node --check browser_extension/lib/shared.js
node --check browser_extension/content.js
node --check browser_extension/service_worker.js
node --check browser_extension/popup/popup.js
node --check browser_extension/options/options.js
node --check browser_extension/wildfake_demo/app.js
```

The tests validate candidate selection/deduplication, result wording, malformed
response handling, loopback URL restrictions, least-privilege manifest fields,
and the existence of every declared entrypoint.
See the companion [`validation record`](../browser_product/VALIDATION.md) for
the real-checkpoint HTTP smoke and rendered demo-wall evidence.

## Current limitations

- PatchHead is the only verdict source. DID remains an offline ablation because
  diffusion reconstruction is unsuitable for an interactive feed scan.
- Physics is opt-in, explanation-only, and can abstain when scene constraints
  are not testable. Candidate region/pair counts and abstention reasons are
  shown instead of treating abstention as a negative finding.
- The residual artifact sidecar is explanation-only. It is sensitive to camera
  processing, JPEG, resizing, sharpening, denoising, and screenshots, and its
  current localization mask is weak.
- Scores are uncalibrated classifier signals, not probabilities or proof.
- Whole-page coverage is bounded and based on rendered viewports, not direct
  full-resolution asset downloads. Dynamic or virtualized pages can still
  change during capture and may need another scan.
- Images inside nested iframes or shadow DOM may not be individually discovered,
  even though their pixels are visible in the viewport screenshot.
- Rendered screenshots do not preserve trustworthy source metadata. Provenance
  is shown as unavailable, and credential absence is never negative evidence.
