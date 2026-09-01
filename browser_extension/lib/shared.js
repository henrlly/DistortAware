(function initializeShared(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  root.AigcShared = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function sharedFactory() {
  "use strict";

  const DEFAULT_SETTINGS = Object.freeze({
    detectorMode: "original_patchhead",
    originalServiceUrl: "http://127.0.0.1:8765",
    originalApiToken: "",
    prismguardServiceUrl: "http://127.0.0.1:8766",
    prismguardApiToken: "",
    demoServiceUrl: "http://127.0.0.1:8767",
    demoApiToken: "prismguard-demo-only",
    scanScope: "page",
    maxImages: 24,
    includePhysics: false,
    includeArtifacts: true,
    minDimension: 96,
    minArea: 14000,
  });

  const DETECTOR_MODES = Object.freeze({
    original_patchhead: Object.freeze({
      label: "Original PatchHead",
      urlKey: "originalServiceUrl",
      tokenKey: "originalApiToken",
    }),
    prismguard: Object.freeze({
      label: "PrismGuard pure DINO",
      urlKey: "prismguardServiceUrl",
      tokenKey: "prismguardApiToken",
    }),
    demo_fixture: Object.freeze({
      label: "Wiring demo fixture",
      urlKey: "demoServiceUrl",
      tokenKey: "demoApiToken",
    }),
  });

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, Number(value)));
  }

  function finiteNumber(value, fallback) {
    if (value === "" || value === null || value === undefined) return fallback;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : fallback;
  }

  function validateServiceUrl(value) {
    let parsed;
    try {
      parsed = new URL(String(value || ""));
    } catch (_error) {
      throw new Error("Service URL must be a valid loopback HTTP URL.");
    }
    if (
      parsed.protocol !== "http:" ||
      !["127.0.0.1", "localhost"].includes(parsed.hostname) ||
      parsed.username ||
      parsed.password ||
      !["", "/"].includes(parsed.pathname) ||
      parsed.search ||
      parsed.hash
    ) {
      throw new Error("Service URL must use http://127.0.0.1 or http://localhost.");
    }
    return parsed.origin;
  }

  function normalizeSettings(raw) {
    const source = raw && typeof raw === "object" ? raw : {};
    const detectorMode = Object.hasOwn(DETECTOR_MODES, source.detectorMode)
      ? source.detectorMode
      : DEFAULT_SETTINGS.detectorMode;
    const normalizedProfiles = {};
    for (const profile of Object.values(DETECTOR_MODES)) {
      const legacyUrl = profile.urlKey === "originalServiceUrl" ? source.serviceUrl : undefined;
      const legacyToken = profile.tokenKey === "originalApiToken" ? source.apiToken : undefined;
      let url = DEFAULT_SETTINGS[profile.urlKey];
      try {
        url = validateServiceUrl(source[profile.urlKey] || legacyUrl || url);
      } catch (_error) {
        url = DEFAULT_SETTINGS[profile.urlKey];
      }
      normalizedProfiles[profile.urlKey] = url;
      const rawToken = source[profile.tokenKey] ?? legacyToken;
      normalizedProfiles[profile.tokenKey] = typeof rawToken === "string"
        ? rawToken.trim()
        : DEFAULT_SETTINGS[profile.tokenKey];
    }
    const activeProfile = DETECTOR_MODES[detectorMode];
    return {
      detectorMode,
      ...normalizedProfiles,
      serviceUrl: normalizedProfiles[activeProfile.urlKey],
      apiToken: normalizedProfiles[activeProfile.tokenKey],
      scanScope: source.scanScope === "visible" ? "visible" : "page",
      maxImages: Math.round(clamp(finiteNumber(source.maxImages, 24), 1, 60)),
      includePhysics: Boolean(source.includePhysics),
      includeArtifacts: source.includeArtifacts === undefined
        ? DEFAULT_SETTINGS.includeArtifacts
        : Boolean(source.includeArtifacts),
      minDimension: Math.round(clamp(finiteNumber(source.minDimension, 96), 48, 320)),
      minArea: Math.round(clamp(finiteNumber(source.minArea, 14000), 2304, 250000)),
    };
  }

  function validateHealthForMode(health, detectorMode) {
    if (!health || health.status !== "ready") {
      throw new Error("The local service is not ready.");
    }
    const family = String(health.detector?.family || "");
    const scoreSource = String(health.prediction_contract?.score_source || "");
    const isDemo = health.scientific_status === "plumbing_only_no_aigc_performance_claim";
    if (detectorMode === "prismguard") {
      if (family !== "prismguard_pure_dino" || scoreSource !== "calibrated_dino_logit_only") {
        throw new Error("Selected PrismGuard, but this endpoint is not a sealed pure-DINO service.");
      }
    } else if (detectorMode === "original_patchhead") {
      if (family === "prismguard_pure_dino" || isDemo) {
        throw new Error("Selected Original PatchHead, but this endpoint serves a different backend.");
      }
    } else if (detectorMode === "demo_fixture") {
      if (!isDemo) {
        throw new Error("Selected Wiring demo, but this endpoint is not an explicit demo fixture.");
      }
    } else {
      throw new Error("Unknown detector selection.");
    }
    return health;
  }

  function validateResultForMode(result, detectorMode) {
    if (!result || result.error) return result;
    const family = String(result.detector?.family || "");
    const scoreKind = String(result.verdict?.score_kind || "");
    const isDemo = result.scientific_status === "plumbing_only_no_aigc_performance_claim";
    if (detectorMode === "prismguard" && (
      family !== "prismguard_pure_dino" || scoreKind !== "calibrated_dino_probability"
    )) {
      throw new Error("PrismGuard response contract mismatch.");
    }
    if (detectorMode === "original_patchhead" && (family === "prismguard_pure_dino" || isDemo)) {
      throw new Error("Original PatchHead response contract mismatch.");
    }
    if (detectorMode === "demo_fixture" && !isDemo) {
      throw new Error("Wiring demo response contract mismatch.");
    }
    return result;
  }

  function healthRequest(settings) {
    const normalized = normalizeSettings(settings);
    if (!normalized.apiToken) throw new Error("Local service token is not configured.");
    return {
      url: `${normalized.serviceUrl}/v1/health`,
      headers: { Authorization: `Bearer ${normalized.apiToken}` },
    };
  }

  function analyzeRequest(settings) {
    const normalized = normalizeSettings(settings);
    if (!normalized.apiToken) throw new Error("Set the selected local service token in extension options.");
    return {
      url: `${normalized.serviceUrl}/v1/analyze`,
      headers: {
        Authorization: `Bearer ${normalized.apiToken}`,
        "Content-Type": "image/png",
        "X-AIGC-Explain": normalized.includePhysics ? "true" : "false",
        "X-AIGC-Artifact": normalized.includeArtifacts ? "true" : "false",
        "X-AIGC-Source-Kind": "viewport_capture",
      },
    };
  }

  function intersectionRect(rect, viewport) {
    const left = clamp(rect.x, 0, viewport.width);
    const top = clamp(rect.y, 0, viewport.height);
    const right = clamp(rect.x + rect.width, 0, viewport.width);
    const bottom = clamp(rect.y + rect.height, 0, viewport.height);
    return {
      x: left,
      y: top,
      width: Math.max(0, right - left),
      height: Math.max(0, bottom - top),
    };
  }

  function rectArea(rect) {
    return Math.max(0, rect.width) * Math.max(0, rect.height);
  }

  function rectIou(left, right) {
    const x1 = Math.max(left.x, right.x);
    const y1 = Math.max(left.y, right.y);
    const x2 = Math.min(left.x + left.width, right.x + right.width);
    const y2 = Math.min(left.y + left.height, right.y + right.height);
    const overlap = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union = rectArea(left) + rectArea(right) - overlap;
    return union > 0 ? overlap / union : 0;
  }

  function candidateScore(candidate, viewport) {
    const rect = candidate.visibleRect;
    const area = rectArea(rect);
    const centerX = rect.x + rect.width / 2;
    const centerY = rect.y + rect.height / 2;
    const dx = Math.abs(centerX - viewport.width / 2) / Math.max(viewport.width / 2, 1);
    const dy = Math.abs(centerY - viewport.height / 2) / Math.max(viewport.height / 2, 1);
    const centrality = 1 - clamp(Math.hypot(dx, dy) / Math.SQRT2, 0, 1);
    const typeWeight = candidate.kind === "image" ? 1.08 :
      ["poster", "video_frame"].includes(candidate.kind) ? 1.03 : 1;
    return area * (0.72 + 0.28 * centrality) * typeWeight;
  }

  function selectCandidates(candidates, settings, viewport) {
    const options = normalizeSettings(settings);
    const minVisibleRatio = clamp(
      finiteNumber(settings?.minVisibleRatio, 0.55),
      0.25,
      0.95,
    );
    const prepared = [];
    for (const candidate of Array.isArray(candidates) ? candidates : []) {
      if (!candidate || !candidate.rect) continue;
      const visibleRect = intersectionRect(candidate.rect, viewport);
      const originalArea = rectArea(candidate.rect);
      const visibleArea = rectArea(visibleRect);
      const visibleRatio = originalArea > 0 ? visibleArea / originalArea : 0;
      const aspect = visibleRect.width / Math.max(visibleRect.height, 1);
      if (
        visibleRect.width < options.minDimension ||
        visibleRect.height < options.minDimension ||
        visibleArea < options.minArea ||
        visibleRatio < minVisibleRatio ||
        aspect < 0.18 ||
        aspect > 5.5
      ) {
        continue;
      }
      prepared.push({
        ...candidate,
        visibleRect,
        visibleRatio,
        rankScore: candidateScore({ ...candidate, visibleRect }, viewport),
      });
    }
    prepared.sort((left, right) => right.rankScore - left.rankScore);
    const selected = [];
    for (const candidate of prepared) {
      const duplicate = selected.some((existing) => {
        const sameSource = candidate.source && candidate.source === existing.source;
        return rectIou(candidate.visibleRect, existing.visibleRect) >= 0.82 ||
          (sameSource && rectIou(candidate.visibleRect, existing.visibleRect) >= 0.45);
      });
      if (!duplicate) selected.push(candidate);
      if (selected.length >= options.maxImages) break;
    }
    return selected;
  }

  function nextScrollPosition(currentY, scrollHeight, viewportHeight, overlap = 0.18) {
    const viewport = Math.max(1, finiteNumber(viewportHeight, 1));
    const maximum = Math.max(0, finiteNumber(scrollHeight, viewport) - viewport);
    const current = clamp(finiteNumber(currentY, 0), 0, maximum);
    if (current >= maximum - 1) return { y: maximum, maximum, atEnd: true };
    const step = Math.max(1, viewport * (1 - clamp(finiteNumber(overlap, 0.18), 0, 0.8)));
    const target = Math.min(maximum, Math.max(current + 1, current + step));
    return { y: target, maximum, atEnd: target >= maximum - 1 };
  }

  function resultLabel(result) {
    if (!result || result.error) {
      return { text: "Scan failed", tone: "error", detail: result?.error || "Unknown error" };
    }
    const verdict = result.verdict || {};
    const numericScore = Number(verdict.aigc_score);
    const score = Number.isFinite(numericScore) ? numericScore : null;
    if (typeof verdict.is_aigc !== "boolean" || score === null) {
      return { text: "Invalid result", tone: "error", detail: "The service response was incomplete." };
    }
    const isDemo = result.scientific_status === "plumbing_only_no_aigc_performance_claim";
    const scoreLabel = verdict.score_kind === "calibrated_dino_probability"
      ? "Calibrated DINO probability"
      : verdict.score_kind === "demo_fixture_signal_not_a_probability"
        ? "Wiring fixture signal"
        : "Classifier signal";
    return {
      text: isDemo ? "Wiring fixture" : verdict.is_aigc ? "AIGC signal" : "No AIGC signal",
      tone: isDemo ? "clear" : verdict.is_aigc ? "aigc" : "clear",
      detail: `${scoreLabel} ${(100 * score).toFixed(1)}%`,
      score,
    };
  }

  function summarizeResults(results) {
    const summary = { total: 0, aigc: 0, clear: 0, errors: 0 };
    for (const item of Array.isArray(results) ? results : []) {
      summary.total += 1;
      const label = resultLabel(item.result || item);
      if (label.tone === "aigc") summary.aigc += 1;
      else if (label.tone === "clear") summary.clear += 1;
      else summary.errors += 1;
    }
    return summary;
  }

  return {
    DEFAULT_SETTINGS,
    DETECTOR_MODES,
    clamp,
    intersectionRect,
    healthRequest,
    analyzeRequest,
    normalizeSettings,
    nextScrollPosition,
    rectIou,
    resultLabel,
    selectCandidates,
    summarizeResults,
    validateHealthForMode,
    validateResultForMode,
    validateServiceUrl,
  };
});
