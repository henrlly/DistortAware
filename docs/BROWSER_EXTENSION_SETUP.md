# PrismGuard browser extension setup

The Manifest V3 extension can switch between two independent local detector
services:

- **Original PatchHead** — the upstream pooled PatchHead runtime.
- **PrismGuard pure DINO** — a sealed PrismGuard bundle whose score is exactly
  `calibrated_probability(dino_logit)`.

There is also a conspicuously labeled **Wiring demo fixture**. It tests page
capture, local HTTP, model selection, badges, and error handling without a
checkpoint. It is not an AIGC detector and must not be used as accuracy evidence.

## 1. Install the unpacked extension

1. Open `chrome://extensions` (or the equivalent page in your Chromium browser).
2. Enable **Developer mode**.
3. Choose **Load unpacked** and select the repository's `browser_extension/`
   directory. Do not select the ZIP file for unpacked development.
4. Pin **PrismGuard — AIGC Signal Inspector** to the toolbar.
5. After changing extension source, return to the extensions page and press
   **Reload** on the PrismGuard card.

The extension requests only `activeTab`, `scripting`, and `storage`, and may
contact only HTTP loopback hosts. It sends rendered image crops—not page URLs,
cookies, captions, or source URLs—to the selected local service.

## 2. Quick checkpoint-free demo

Use Python 3.10 or newer from the repository root:

```bash
python3 -m browser_product.service \
  --demo-fixture \
  --port 8767 \
  --api-token prismguard-demo-only
```

In a second terminal, serve the deterministic image wall:

```bash
python3 -m http.server 8088 --directory .
```

Open the extension settings and select **Wiring demo fixture — no accuracy
claim**. Keep the default URL `http://127.0.0.1:8767`, enter token
`prismguard-demo-only`, save, and press **Test connection**. It must report a
wiring fixture, not PatchHead or PrismGuard.

Visit `http://127.0.0.1:8088/browser_extension/demo/`, open the extension, and
press **Scan page images**. Expected behavior:

- the 32×32 decorative image is ignored;
- the large rendered fixtures receive badges;
- every result says **Wiring fixture**, never “AIGC signal”;
- the scan note says the run is wiring-only and the AIGC summary is hidden;
- **Clear badges** removes the overlays.

## 2a. One-command verified PatchHead demo

Use the launcher when demonstrating the real original detector. It verifies the
sealed pooled head and pinned DINOv3-L safetensors before starting PatchHead,
the explicitly labeled wiring fixture, and the local image wall. All three
servers bind to loopback, model loading is offline, and both diagnostic profiles
are forced off.

```bash
python3 scripts/launch_browser_demo.py \
  --checkpoint /absolute/path/to/patchhead_pooled.pt \
  --backbone-safetensors /absolute/path/to/model.safetensors \
  --hf-home /absolute/path/to/huggingface-cache \
  --patchhead-token patchhead-demo \
  --fixture-token fixture-demo
```

The launcher prints the exact local URLs and tokens to enter in the extension
settings. Use `--check-only` first to verify artifacts and inspect the redacted
process plan without opening ports. Press Ctrl-C once to stop every child
process.

PrismGuard is deliberately absent unless all four approved inputs are supplied
together: `--prismguard-bundle`, `--prismguard-checkpoint`,
`--prismguard-license-ledger`, and `--prismguard-root`. A partial tuple fails
closed; neither the wiring fixture nor a smoke/proxy bundle is substituted.

## 3. Original PatchHead service

The checkpoint stays outside Git. The frozen DINOv3 backbone must already be in
the model cache if the machine is offline.

```bash
python3 -m browser_product.service \
  --checkpoint /absolute/path/to/patchhead_pooled.pt \
  --port 8765
```

Copy the printed token into the extension's **Original bearer token** field,
select **Original PatchHead**, save, and test the connection. The extension
rejects this endpoint if PrismGuard is selected.

## 4. PrismGuard service

A real PrismGuard run requires all three approved artifacts. It does not fall
back to the rejected handcrafted smoke model.

```bash
python3 -m browser_product.service \
  --prismguard-bundle /absolute/path/to/detector-bundle.json \
  --prismguard-checkpoint /absolute/path/to/dinov3-vitl16-checkpoint \
  --prismguard-license-ledger /absolute/path/to/checkpoint-licenses.json \
  --prismguard-root /absolute/path/to/prismguard \
  --port 8766
```

Copy the printed token into **PrismGuard bearer token**, select **PrismGuard
pure DINO**, save, and test. The health response and every inference result must
identify `prismguard_pure_dino`; a PatchHead or fixture response fails closed.
Physics and artifact diagnostics remain outside this score path.

## 5. Test and package

Run the checkpoint-independent acceptance suite:

```bash
node --test browser_extension/tests/*.test.cjs
node --check browser_extension/lib/shared.js
node --check browser_extension/content.js
node --check browser_extension/service_worker.js
node --check browser_extension/popup/popup.js
node --check browser_extension/options/options.js
python3 -m unittest discover -s browser_product/tests -v
```

The optional heuristic-physics test requires a compatible OpenCV/NumPy build.
The model-switching, PrismGuard invariance, service-authentication, and fixture
tests do not require either detector checkpoint. If physics extras are absent,
run the core backend cases directly and keep that environment limitation in the
test record:

```bash
python3 -m unittest \
  browser_product.tests.test_backend.BrowserInferenceBackendTests.test_demo_fixture_is_explicit_and_deterministic \
  browser_product.tests.test_backend.BrowserInferenceBackendTests.test_prismguard_adapter_keeps_diagnostics_out_of_prediction \
  browser_product.tests.test_backend.BrowserInferenceBackendTests.test_prismguard_adapter_rejects_coupled_diagnostics_profiles \
  browser_product.tests.test_service
```

Build a deterministic review/release archive:

```bash
python3 scripts/package_browser_extension.py
unzip -t dist/prismguard-browser-extension.zip
```

The package script excludes test/demo pages from the installed extension. The
demo wall remains repository-hosted and is served separately over loopback.

## Troubleshooting

- **Selected model does not match endpoint:** verify the selected profile and
  port. This is an intentional fail-closed check.
- **401 Unauthorized:** copy the token printed by the matching service; tokens
  are stored independently for each profile.
- **Cannot scan a protected page:** browser settings, extension stores, and DRM
  surfaces cannot be scripted. Test on an ordinary HTTP(S) page.
- **No eligible images:** images smaller than the configured side/area thresholds
  are deliberately filtered.
- **Code changed but UI did not:** press **Reload** on the extension card, then
  reload the target page.
