const assert = require("node:assert/strict");
const test = require("node:test");

const shared = require("../lib/shared.js");

test("service URL accepts only an origin on HTTP loopback", () => {
  assert.equal(shared.validateServiceUrl("http://127.0.0.1:8765/"), "http://127.0.0.1:8765");
  assert.equal(shared.validateServiceUrl("http://localhost:9000"), "http://localhost:9000");
  for (const value of [
    "https://127.0.0.1:8765",
    "http://example.com:8765",
    "http://127.0.0.1:8765/proxy",
    "http://user:pass@127.0.0.1:8765",
    "http://127.0.0.1:8765?token=leak",
  ]) {
    assert.throws(() => shared.validateServiceUrl(value), /loopback|127\.0\.0\.1/);
  }
});

test("settings normalization is finite and bounded", () => {
  const normalized = shared.normalizeSettings({
    maxImages: "not-a-number",
    minDimension: -4,
    minArea: 999999,
    includePhysics: 1,
    includeArtifacts: 0,
    scanScope: "visible",
    apiToken: "  fixture  ",
  });
  assert.equal(normalized.maxImages, 24);
  assert.equal(normalized.minDimension, 48);
  assert.equal(normalized.minArea, 250000);
  assert.equal(normalized.includePhysics, true);
  assert.equal(normalized.includeArtifacts, false);
  assert.equal(normalized.scanScope, "visible");
  assert.equal(normalized.apiToken, "fixture");
  assert.equal(shared.normalizeSettings({ maxImages: "" }).maxImages, 24);
  assert.equal(shared.normalizeSettings({}).includeArtifacts, true);
  assert.equal(shared.normalizeSettings({ scanScope: "invalid" }).scanScope, "page");
  assert.equal(shared.normalizeSettings({ maxImages: 999 }).maxImages, 60);
});

test("detector profiles are independently stored and selected", () => {
  const settings = shared.normalizeSettings({
    detectorMode: "prismguard",
    originalServiceUrl: "http://127.0.0.1:8765",
    originalApiToken: "original-token",
    prismguardServiceUrl: "http://127.0.0.1:8766",
    prismguardApiToken: "prism-token",
  });
  assert.equal(settings.detectorMode, "prismguard");
  assert.equal(settings.serviceUrl, "http://127.0.0.1:8766");
  assert.equal(settings.apiToken, "prism-token");
  assert.equal(settings.originalApiToken, "original-token");
});

test("service requests route to the active profile without putting tokens in URLs", () => {
  const raw = {
    detectorMode: "prismguard",
    originalServiceUrl: "http://127.0.0.1:8765",
    originalApiToken: "original-secret",
    prismguardServiceUrl: "http://127.0.0.1:8766",
    prismguardApiToken: "prism-secret",
    includePhysics: true,
    includeArtifacts: false,
  };
  const health = shared.healthRequest(raw);
  assert.equal(health.url, "http://127.0.0.1:8766/v1/health");
  assert.equal(health.headers.Authorization, "Bearer prism-secret");
  assert.doesNotMatch(health.url, /secret/);
  const analyze = shared.analyzeRequest(raw);
  assert.equal(analyze.url, "http://127.0.0.1:8766/v1/analyze");
  assert.equal(analyze.headers.Authorization, "Bearer prism-secret");
  assert.equal(analyze.headers["X-AIGC-Explain"], "true");
  assert.equal(analyze.headers["X-AIGC-Artifact"], "false");
  assert.doesNotMatch(analyze.url, /secret/);

  const original = shared.healthRequest({ ...raw, detectorMode: "original_patchhead" });
  assert.equal(original.url, "http://127.0.0.1:8765/v1/health");
  assert.equal(original.headers.Authorization, "Bearer original-secret");
});

test("selected detector must match the endpoint health and result contract", () => {
  const prismHealth = {
    status: "ready",
    detector: { family: "prismguard_pure_dino" },
    prediction_contract: { score_source: "calibrated_dino_logit_only" },
  };
  assert.equal(shared.validateHealthForMode(prismHealth, "prismguard"), prismHealth);
  assert.throws(
    () => shared.validateHealthForMode(prismHealth, "original_patchhead"),
    /different backend/,
  );
  assert.doesNotThrow(() => shared.validateResultForMode({
    detector: { family: "prismguard_pure_dino" },
    verdict: { score_kind: "calibrated_dino_probability" },
  }, "prismguard"));
  assert.throws(() => shared.validateResultForMode({
    detector: { family: "fake_patchhead_for_contract_tests" },
    verdict: { score_kind: "uncalibrated_aigc_classifier_score" },
  }, "prismguard"), /contract mismatch/);

  const demoHealth = {
    status: "ready",
    scientific_status: "plumbing_only_no_aigc_performance_claim",
    detector: { family: "prismguard_browser_wiring_fixture" },
  };
  assert.equal(shared.validateHealthForMode(demoHealth, "demo_fixture"), demoHealth);
  assert.throws(
    () => shared.validateHealthForMode(demoHealth, "original_patchhead"),
    /different backend/,
  );
});

test("full-page scroll planning overlaps viewports and stops exactly at the end", () => {
  assert.deepEqual(
    shared.nextScrollPosition(0, 2400, 800),
    { y: 656, maximum: 1600, atEnd: false },
  );
  assert.deepEqual(
    shared.nextScrollPosition(1500, 2400, 800),
    { y: 1600, maximum: 1600, atEnd: true },
  );
  assert.deepEqual(
    shared.nextScrollPosition(0, 600, 800),
    { y: 0, maximum: 0, atEnd: true },
  );
});

test("candidate selection filters decorations, clips edges, and ranks large regions", () => {
  const viewport = { width: 1000, height: 800 };
  const selected = shared.selectCandidates([
    { kind: "image", source: "hero", rect: { x: 250, y: 120, width: 500, height: 420 } },
    { kind: "image", source: "tiny", rect: { x: 20, y: 20, width: 40, height: 40 } },
    { kind: "background", source: "edge", rect: { x: 900, y: 100, width: 260, height: 260 } },
    { kind: "image", source: "mostly-offscreen", rect: { x: -500, y: 100, width: 600, height: 300 } },
  ], { maxImages: 5, minDimension: 96, minArea: 14000 }, viewport);

  assert.deepEqual(selected.map((item) => item.source), ["hero"]);
  assert.deepEqual(selected[0].visibleRect, { x: 250, y: 120, width: 500, height: 420 });
});

test("candidate selection deduplicates overlapping visual regions", () => {
  const viewport = { width: 800, height: 600 };
  const selected = shared.selectCandidates([
    { kind: "image", source: "a", rect: { x: 100, y: 80, width: 400, height: 300 } },
    { kind: "background", source: "b", rect: { x: 105, y: 85, width: 395, height: 295 } },
    { kind: "image", source: "c", rect: { x: 550, y: 80, width: 180, height: 180 } },
  ], { maxImages: 5 }, viewport);

  assert.equal(selected.length, 2);
  assert.equal(selected[0].source, "a");
  assert.equal(selected[1].source, "c");
  assert.ok(shared.rectIou(selected[0].visibleRect, selected[1].visibleRect) < 0.01);
});

test("labels disclose classifier signal and fail closed on malformed responses", () => {
  assert.deepEqual(
    shared.resultLabel({ verdict: { is_aigc: true, aigc_score: 0.912 } }),
    { text: "AIGC signal", tone: "aigc", detail: "Classifier signal 91.2%", score: 0.912 },
  );
  assert.equal(
    shared.resultLabel({ verdict: { is_aigc: false, aigc_score: 0.12 } }).tone,
    "clear",
  );
  assert.equal(shared.resultLabel({ verdict: { is_aigc: false } }).tone, "error");

  assert.deepEqual(shared.summarizeResults([
    { result: { verdict: { is_aigc: true, aigc_score: 0.9 } } },
    { result: { verdict: { is_aigc: false, aigc_score: 0.1 } } },
    { result: { error: "fixture" } },
  ]), { total: 3, aigc: 1, clear: 1, errors: 1 });
});

test("labels identify a PrismGuard calibrated DINO probability", () => {
  assert.deepEqual(
    shared.resultLabel({
      verdict: {
        is_aigc: false,
        aigc_score: 0.123,
        score_kind: "calibrated_dino_probability",
      },
    }),
    {
      text: "No AIGC signal",
      tone: "clear",
      detail: "Calibrated DINO probability 12.3%",
      score: 0.123,
    },
  );
});

test("demo fixture never renders an AIGC verdict label", () => {
  assert.deepEqual(
    shared.resultLabel({
      scientific_status: "plumbing_only_no_aigc_performance_claim",
      verdict: {
        is_aigc: true,
        aigc_score: 0.8,
        score_kind: "demo_fixture_signal_not_a_probability",
      },
    }),
    {
      text: "Wiring fixture",
      tone: "clear",
      detail: "Wiring fixture signal 80.0%",
      score: 0.8,
    },
  );
});
