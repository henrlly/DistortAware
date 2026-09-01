/* global AigcShared */
"use strict";

importScripts("lib/shared.js");

const activeScans = new Set();

async function loadSettings() {
  const stored = await chrome.storage.local.get(AigcShared.DEFAULT_SETTINGS);
  return AigcShared.normalizeSettings(stored);
}

async function injectContent(tabId) {
  await chrome.scripting.executeScript({
    target: { tabId },
    files: ["lib/shared.js", "content.js"],
  });
}

async function sendTabMessage(tabId, message) {
  const response = await chrome.tabs.sendMessage(tabId, message);
  if (!response?.ok) throw new Error(response?.error || "The active page did not respond.");
  return response;
}

async function captureViewport(tabId, windowId) {
  const before = await chrome.tabs.query({ active: true, windowId });
  if (before[0]?.id !== tabId) {
    throw new Error("Keep the page tab active while its viewport is captured.");
  }
  const dataUrl = await chrome.tabs.captureVisibleTab(windowId, { format: "png" });
  const after = await chrome.tabs.query({ active: true, windowId });
  if (after[0]?.id !== tabId) {
    throw new Error("The active tab changed during capture; return to the page and rescan.");
  }
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  return createImageBitmap(blob);
}

function clampedCrop(candidate, viewport, bitmap) {
  const rect = candidate.rect;
  const scaleX = bitmap.width / Math.max(viewport.width, 1);
  const scaleY = bitmap.height / Math.max(viewport.height, 1);
  const x = Math.max(0, Math.floor(rect.x * scaleX));
  const y = Math.max(0, Math.floor(rect.y * scaleY));
  const right = Math.min(bitmap.width, Math.ceil((rect.x + rect.width) * scaleX));
  const bottom = Math.min(bitmap.height, Math.ceil((rect.y + rect.height) * scaleY));
  return { x, y, width: Math.max(1, right - x), height: Math.max(1, bottom - y) };
}

async function cropCandidate(bitmap, candidate, viewport) {
  const source = clampedCrop(candidate, viewport, bitmap);
  const maximumDimension = 1280;
  const scale = Math.min(1, maximumDimension / Math.max(source.width, source.height));
  const width = Math.max(1, Math.round(source.width * scale));
  const height = Math.max(1, Math.round(source.height * scale));
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) throw new Error("Browser canvas capture is unavailable.");
  context.drawImage(
    bitmap,
    source.x,
    source.y,
    source.width,
    source.height,
    0,
    0,
    width,
    height,
  );
  return canvas.convertToBlob({ type: "image/png" });
}

async function analyzeBlob(blob, settings) {
  const request = AigcShared.analyzeRequest(settings);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch(request.url, {
      method: "POST",
      headers: request.headers,
      body: blob,
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(payload?.error?.message || `Local service returned HTTP ${response.status}.`);
    }
    return AigcShared.validateResultForMode(payload, settings.detectorMode);
  } finally {
    clearTimeout(timeoutId);
  }
}

async function workerPool(items, concurrency, callback) {
  const results = new Array(items.length);
  let nextIndex = 0;
  async function worker() {
    while (true) {
      const index = nextIndex;
      nextIndex += 1;
      if (index >= items.length) return;
      results[index] = await callback(items[index], index);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
  return results;
}

async function emitProgress(completed, total, phase = "analyzing", extra = {}) {
  try {
    await chrome.runtime.sendMessage({
      type: "scan_progress",
      completed,
      total,
      phase,
      ...extra,
    });
  } catch (_error) {
    // The popup may have closed; scanning and page overlays should still finish.
  }
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function analyzeCandidates(
  tabId,
  windowId,
  discovery,
  settings,
  progress,
) {
  if (!discovery.candidates.length) return [];
  const bitmap = await captureViewport(tabId, windowId);
  try {
    return await workerPool(discovery.candidates, 2, async (candidate) => {
      let result;
      try {
        const crop = await cropCandidate(bitmap, candidate, discovery.viewport);
        result = await analyzeBlob(crop, settings);
      } catch (error) {
        const message = error?.name === "AbortError"
          ? "Local inference timed out."
          : error?.message || String(error);
        result = { error: message };
      }
      progress.completed += 1;
      await emitProgress(
        progress.completed,
        progress.total,
        "analyzing",
        { pass: progress.pass },
      );
      return { candidate, result };
    });
  } finally {
    bitmap.close();
  }
}

async function scanTab(tabId, windowId) {
  if (activeScans.has(tabId)) throw new Error("A scan is already running for this tab.");
  activeScans.add(tabId);
  try {
    return await scanTabOnce(tabId, windowId);
  } finally {
    activeScans.delete(tabId);
  }
}

async function scanTabOnce(tabId, windowId) {
  const settings = await loadSettings();
  await injectContent(tabId);
  if (settings.scanScope === "page") {
    return scanFullPage(tabId, windowId, settings);
  }
  return scanVisiblePage(tabId, windowId, settings);
}

async function scanVisiblePage(tabId, windowId, settings) {
  const discovery = await sendTabMessage(tabId, {
    type: "discover_candidates",
    settings,
  });
  if (!discovery.candidates.length) {
    const empty = {
      tabId,
      results: [],
      summary: AigcShared.summarizeResults([]),
      scannedAt: Date.now(),
      scanScope: "visible",
      detectorMode: settings.detectorMode,
    };
    await chrome.storage.session.set({ latestScan: empty });
    return empty;
  }
  const progress = {
    completed: 0,
    total: discovery.candidates.length,
    pass: 1,
  };
  const results = await analyzeCandidates(
    tabId,
    windowId,
    discovery,
    settings,
    progress,
  );
  await sendTabMessage(tabId, { type: "render_results", results });
  const latestScan = {
    tabId,
    results,
    summary: AigcShared.summarizeResults(results),
    scannedAt: Date.now(),
    scanScope: "visible",
    detectorMode: settings.detectorMode,
    viewportsVisited: 1,
    truncated: false,
    scrollRestored: true,
    physicsRequested: settings.includePhysics,
    artifactsRequested: settings.includeArtifacts,
  };
  await chrome.storage.session.set({ latestScan });
  return latestScan;
}

async function scanFullPage(tabId, windowId, settings) {
  const maximumViewports = 40;
  const seen = new Set();
  const results = [];
  const progress = { completed: 0, total: settings.maxImages, pass: 0 };
  let viewportsVisited = 0;
  let truncated = false;
  let restored = false;

  await sendTabMessage(tabId, { type: "page_scan_start" });
  try {
    await delay(450);
    while (
      viewportsVisited < maximumViewports &&
      results.length < settings.maxImages
    ) {
      viewportsVisited += 1;
      progress.pass = viewportsVisited;
      await emitProgress(
        progress.completed,
        progress.total,
        "discovering",
        { pass: viewportsVisited },
      );
      const remaining = settings.maxImages - results.length;
      const discovery = await sendTabMessage(tabId, {
        type: "discover_candidates",
        settings: { ...settings, maxImages: remaining },
        append: true,
        excludeIds: [...seen],
      });
      for (const candidate of discovery.candidates) seen.add(candidate.id);
      const batch = await analyzeCandidates(
        tabId,
        windowId,
        discovery,
        settings,
        progress,
      );
      results.push(...batch);
      if (results.length >= settings.maxImages) {
        truncated = true;
        break;
      }
      const advanced = await sendTabMessage(tabId, { type: "page_scan_advance" });
      if (!advanced.moved) break;
      await delay(450);
    }
    if (viewportsVisited >= maximumViewports) truncated = true;
  } finally {
    try {
      const response = await sendTabMessage(tabId, { type: "page_scan_restore" });
      restored = Boolean(response.restored);
      await delay(80);
    } catch (_error) {
      restored = false;
    }
  }

  await sendTabMessage(tabId, { type: "render_results", results });
  const latestScan = {
    tabId,
    results,
    summary: AigcShared.summarizeResults(results),
    scannedAt: Date.now(),
    scanScope: "page",
    detectorMode: settings.detectorMode,
    viewportsVisited,
    truncated,
    scrollRestored: restored,
    physicsRequested: settings.includePhysics,
    artifactsRequested: settings.includeArtifacts,
  };
  await chrome.storage.session.set({ latestScan });
  return latestScan;
}

async function health(settingsOverride) {
  const settings = settingsOverride
    ? AigcShared.normalizeSettings(settingsOverride)
    : await loadSettings();
  const request = AigcShared.healthRequest(settings);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch(request.url, {
      headers: request.headers,
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(payload?.error?.message || `Health check failed (${response.status}).`);
    }
    return AigcShared.validateHealthForMode(payload, settings.detectorMode);
  } catch (error) {
    if (error?.name === "AbortError") throw new Error("Local service health check timed out.");
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!["scan_tab", "service_health", "clear_tab"].includes(message?.type)) {
    return false;
  }
  (async () => {
    if (message?.type === "scan_tab") {
      return scanTab(message.tabId, message.windowId);
    }
    if (message?.type === "service_health") {
      return health(message.settings);
    }
    if (message?.type === "clear_tab") {
      await injectContent(message.tabId);
      await sendTabMessage(message.tabId, { type: "clear_results" });
      await chrome.storage.session.remove("latestScan");
      return { cleared: true };
    }
    throw new Error("Unsupported extension request.");
  })()
    .then((result) => sendResponse({ ok: true, result }))
    .catch((error) => sendResponse({ ok: false, error: error?.message || String(error) }));
  return true;
});
