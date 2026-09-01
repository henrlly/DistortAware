# Browser product architecture and implementation plan

## Product outcome

The deliverable is a Chromium Manifest V3 extension backed by a local Python
inference service. After an explicit click, it discovers rendered image-like
regions on the active page, captures the pixels the user can actually see,
runs the repository's primary pooled PatchHead detector locally, and overlays
an auditable result on each region.

The extension does not upload page images to a third party. The pooled PatchHead
checkpoint and model caches remain external to Git; the small residual-sidecar
checkpoint is the one intentionally bundled exception.

## Decision hierarchy

```text
rendered image crop (one viewport or bounded auto-scroll)
        |
        v
pooled DINOv3 PatchHead --------------------> AIGC score + thresholded verdict
        |
        +-- existing patch logits ----------> weak spatial summary
        |
        +-- optional same-pass DINO grid ---> physics proposal descriptors
                                                  |
                                    optional physics sidecar
                                    perspective / shadow / reflection
        |
        +-- optional RGB + high-pass residual -> compact U-Net class/mask evidence
```

- PatchHead is the sole interactive verdict source.
- Physics and residual artifacts are opt-in explanation evidence and cannot
  alter the score or verdict.
- DID remains a documented offline ablation. Its diffusion reconstruction is
  too slow for a feed-scanning interaction and is not silently fused into the
  browser result.
- Source provenance is unavailable from rendered screenshots and is reported as
  unknown rather than being inferred from missing metadata.

## Components

### Extension

- Uses only `activeTab`, `scripting`, and `storage` permissions plus loopback
  host access.
- Injects discovery code only after the user clicks Scan; it does not observe
  browsing continuously.
- Finds visible `<img>` elements, video posters/current frames, and CSS
  background images at each captured viewport.
- Filters tiny/decorative regions, ranks by visible area, deduplicates heavily
  overlapping candidates, and enforces a user-configurable cap.
- Supports a current-viewport mode and a bounded whole-page mode. Whole-page
  mode auto-scrolls through overlapping viewports, waits for lazy rendering,
  deduplicates stable element/source IDs, stops at 1–60 images or 40 viewports,
  and restores the original position.
- Captures each active viewport and crops candidates with `OffscreenCanvas`.
  This works for cross-origin CDN images and page-owned blob URLs without
  granting the extension broad network access.
- Sends PNG crops—not page URLs, cookies, captions, or DOM text—to the local
  service.
- Draws fixed, non-interactive badges and exposes per-image details in the
  popup. The label says “AIGC signal” rather than “probability”; physics
  proposal counts, abstention reasons, and residual evidence remain secondary.

### Local inference service

- Binds to `127.0.0.1` by default and requires a bearer token generated at
  startup unless explicitly supplied.
- Accepts raw image bytes at `POST /v1/analyze`; it never fetches a user-provided
  URL, preventing server-side request forgery and cookie leakage.
- Enforces content length, decoded-pixel, format, and timeout-safe error paths.
- Loads one pooled PatchHead runtime and serializes model access with a lock.
- Reuses results through a bounded in-memory SHA-256 cache.
- Can keep one persistent heuristic or learned physics engine when configured;
  the default profile is off for fast feed scanning.
- Loads the bundled residual-aware U-Net by default as a request-time optional
  explainer. Its outputs are never read by PatchHead's verdict path.
- Returns detector/checkpoint provenance, score kind, threshold, verdict,
  extraction limitations, optional physics evidence/DINO spatial association,
  and timing. Request-local filesystem paths are stripped from the API.

## API contract

`GET /v1/health` and `POST /v1/analyze` require:

```text
Authorization: Bearer <local token>
Origin: chrome-extension://... or moz-extension://... (browser clients)
```

The analyze body is raw `image/png`, `image/jpeg`, or `image/webp`. Optional
headers `X-AIGC-Explain: true` and `X-AIGC-Artifact: true` request the configured
explanation sidecars.

Successful responses contain:

```json
{
  "schema_version": "0.2.0",
  "verdict": {
    "is_aigc": true,
    "aigc_score": 0.91,
    "threshold": 0.785,
    "score_kind": "uncalibrated_aigc_classifier_score"
  },
  "detector": {
    "family": "dino_patchhead",
    "checkpoint_sha256": "..."
  },
  "explanation": {
    "patch_evidence": {"grid_shape": [16, 16], "mean": 0.8, "maximum": 0.99},
    "physics": null,
    "artifact": null,
    "physics_affects_detector_score": false,
    "artifact_affects_detector_score": false
  },
  "limitations": []
}
```

## Security and privacy boundaries

- Loopback bind is mandatory unless the operator explicitly supplies another
  host; non-loopback binds produce a warning.
- Random bearer token by default; no token appears in repository files.
- CORS allows extension origins and loopback development pages only.
- No source URL is sent to or fetched by the service.
- Temporary image files are scoped to one request and deleted immediately.
- Model/data artifacts and generated scans stay ignored.
- Browser scans are user-initiated and restricted to rendered pixels in the
  active tab; whole-page mode is bounded and restores its starting scroll.

## Acceptance criteria

### Automated

- Service health/auth/CORS/body-limit tests pass with a fake PatchHead runtime.
- An image request returns the official score formula and a stable response
  contract; a repeated request is a cache hit.
- Optional heuristic physics attaches evidence while preserving primary fields.
- Candidate filtering/deduplication and result-label helpers pass Node tests.
- Existing 83 physics and 18 PatchHead contracts remain green.
- Source imports without downloading checkpoints; no model/data artifact is
  tracked.

### Manual demo

1. Start the local service with the external pooled checkpoint.
2. Load `browser_extension/` as an unpacked Chromium extension.
3. Paste the startup token into extension settings and test the connection.
4. Open a page containing several images and click Scan page images.
5. Confirm each selected region receives a human-visible badge and popup row.
6. Enable physics and restart/configure the server with a physics profile to
   show optional cue details without changing any verdict.
7. Demonstrate an inaccessible/internal page failing with a clear message.

## Known limitations

- Viewport capture analyzes rendered pixels, not the full original-resolution
  asset. Cropping, page overlays, browser scaling, and screenshots can affect
  evidence; the response discloses `viewport_capture` provenance.
- Whole-page mode cannot guarantee every transient item on virtualized or
  continuously mutating feeds; it stops at its configured image/safety cap.
- Animated GIFs and videos are represented by the currently rendered frame or
  poster; this is image detection, not video-temporal forensics.
- PatchHead scores are not calibrated probabilities, and browser badges are not
  proof of authenticity.
- Automatic physics often abstains and has limited natural-scene correspondence
  validation; it is disabled by default.
- Residual evidence is sensitive to ordinary camera and redistribution
  processing, and the bundled mask head has weak recorded localization IoU.
- Browser-managed pages, extension stores, and some protected surfaces cannot
  be scripted or captured.
- Visuals inside nested iframes or shadow DOM may not be discoverable as
  separate crop candidates even though the viewport screenshot contains them.

## Delivery sequence

1. Freeze this architecture and API boundary.
2. Implement/test the local service with a reusable fake runtime.
3. Implement/test pure extension candidate/result helpers.
4. Implement popup, options, active-tab discovery, capture/cropping, API calls,
   overlays, progress, and error handling.
5. Run real pooled-checkpoint API smoke tests and a local static-page scan.
6. Update root/operator documentation and CI paths.
7. Commit the review-ready branch locally. Do not push or open a pull request
   until explicit approval is given.

## Implementation status

Steps 1–6 are implemented on the local review branch. The service and pure
extension contracts are checkpoint-independent and covered by Python/Node
tests; the pooled-checkpoint HTTP boundary and rendered demo wall have also
been validated. The current local revision adds explanation-only artifact
integration, more informative physics abstentions, and bounded whole-page
capture. It remains local pending review and explicit remote-publication
permission.
