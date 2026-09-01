(function initializeContentScript() {
  "use strict";

  if (globalThis.__aigcSignalInspectorLoaded) return;
  globalThis.__aigcSignalInspectorLoaded = true;

  const shared = globalThis.AigcShared;
  const candidateElements = new Map();
  const elementCandidateIds = new WeakMap();
  const latestElementIds = new WeakMap();
  const renderedResults = new Map();
  let nextCandidateNumber = 1;
  let overlayHost = null;
  let overlayRoot = null;
  let positionFrame = null;
  let pageScanState = null;

  function computedVisible(element, rect) {
    if (!rect || rect.width <= 0 || rect.height <= 0) return false;
    const style = getComputedStyle(element);
    return style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity) > 0.05;
  }

  function rectObject(rect) {
    return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
  }

  function backgroundSource(value) {
    if (!value || value === "none") return null;
    const match = value.match(/url\(["']?(.*?)["']?\)/i);
    return match ? match[1] : null;
  }

  function rawCandidates() {
    const candidates = [];
    for (const image of document.images) {
      const rect = image.getBoundingClientRect();
      if (!computedVisible(image, rect)) continue;
      candidates.push({
        element: image,
        kind: "image",
        source: image.currentSrc || image.src || "rendered-image",
        rect: rectObject(rect),
        naturalWidth: image.naturalWidth || null,
        naturalHeight: image.naturalHeight || null,
        alt: image.alt ? image.alt.slice(0, 120) : "",
      });
    }
    let videoIndex = 0;
    for (const video of document.querySelectorAll("video")) {
      const rect = video.getBoundingClientRect();
      if (!computedVisible(video, rect)) continue;
      candidates.push({
        element: video,
        kind: video.poster ? "poster" : "video_frame",
        source: video.poster || `rendered-video-frame-${videoIndex}`,
        rect: rectObject(rect),
        naturalWidth: video.videoWidth || null,
        naturalHeight: video.videoHeight || null,
        alt: video.poster ? "Video poster" : "Current video frame",
      });
      videoIndex += 1;
    }
    const elements = document.querySelectorAll("body *");
    const ceiling = Math.min(elements.length, 2500);
    for (let index = 0; index < ceiling; index += 1) {
      const element = elements[index];
      if (element instanceof HTMLImageElement || element instanceof HTMLVideoElement) continue;
      const style = getComputedStyle(element);
      const source = backgroundSource(style.backgroundImage);
      if (!source) continue;
      const rect = element.getBoundingClientRect();
      if (!computedVisible(element, rect)) continue;
      candidates.push({
        element,
        kind: "background",
        source,
        rect: rectObject(rect),
        naturalWidth: null,
        naturalHeight: null,
        alt: element.getAttribute("aria-label")?.slice(0, 120) || "Background image",
      });
    }
    return candidates;
  }

  function candidateId(element, kind, source) {
    let identifiers = elementCandidateIds.get(element);
    if (!identifiers) {
      identifiers = new Map();
      elementCandidateIds.set(element, identifiers);
    }
    const key = `${kind}\n${source || "rendered"}`;
    if (!identifiers.has(key)) {
      identifiers.set(key, `aigc-${nextCandidateNumber}`);
      nextCandidateNumber += 1;
    }
    return identifiers.get(key);
  }

  function discover(settings, { append = false, excludeIds = [] } = {}) {
    if (!append) clearAll();
    const viewport = { width: innerWidth, height: innerHeight };
    const excluded = new Set(Array.isArray(excludeIds) ? excludeIds : []);
    const raw = rawCandidates()
      .map((candidate) => ({
        ...candidate,
        stableId: candidateId(candidate.element, candidate.kind, candidate.source),
      }))
      .filter((candidate) => !excluded.has(candidate.stableId));
    const selected = shared.selectCandidates(
      raw,
      {
        ...settings,
        minVisibleRatio: settings?.scanScope === "page" ? 0.3 : 0.55,
      },
      viewport,
    );
    const serialized = selected.map((candidate) => {
      const id = candidate.stableId;
      candidateElements.set(id, candidate.element);
      latestElementIds.set(candidate.element, id);
      return {
        id,
        kind: candidate.kind,
        rect: candidate.visibleRect,
        visibleRatio: candidate.visibleRatio,
        naturalWidth: candidate.naturalWidth,
        naturalHeight: candidate.naturalHeight,
        alt: candidate.alt,
        pageTop: Math.round(scrollY + candidate.visibleRect.y),
      };
    });
    return {
      candidates: serialized,
      viewport: {
        ...viewport,
        devicePixelRatio: devicePixelRatio || 1,
      },
    };
  }

  function pageState() {
    const viewportHeight = Math.max(1, innerHeight);
    const scrollHeight = Math.max(
      document.documentElement.scrollHeight,
      document.body?.scrollHeight || 0,
      viewportHeight,
    );
    const maximum = Math.max(0, scrollHeight - viewportHeight);
    return {
      scrollY: window.scrollY,
      scrollHeight,
      viewportHeight,
      maximum,
      atEnd: window.scrollY >= maximum - 1,
    };
  }

  function startPageScan() {
    if (pageScanState) restorePageScan();
    clearAll();
    const root = document.documentElement;
    pageScanState = {
      x: window.scrollX,
      y: window.scrollY,
      scrollBehavior: root.style.scrollBehavior,
    };
    root.style.scrollBehavior = "auto";
    window.scrollTo({ left: pageScanState.x, top: 0, behavior: "auto" });
    return pageState();
  }

  function advancePageScan() {
    if (!pageScanState) throw new Error("No full-page scan is active.");
    const before = pageState();
    const next = shared.nextScrollPosition(
      before.scrollY,
      before.scrollHeight,
      before.viewportHeight,
    );
    window.scrollTo({ left: pageScanState.x, top: next.y, behavior: "auto" });
    return {
      ...pageState(),
      targetY: next.y,
      moved: Math.abs(next.y - before.scrollY) > 1,
      atEnd: next.atEnd,
    };
  }

  function restorePageScan() {
    if (!pageScanState) return { restored: false, ...pageState() };
    const saved = pageScanState;
    pageScanState = null;
    window.scrollTo({ left: saved.x, top: saved.y, behavior: "auto" });
    document.documentElement.style.scrollBehavior = saved.scrollBehavior;
    return { restored: true, ...pageState() };
  }

  function ensureOverlay() {
    if (overlayHost?.isConnected && overlayRoot) return;
    overlayHost = document.createElement("div");
    overlayHost.id = "aigc-signal-inspector-overlays";
    overlayHost.style.cssText = "position:fixed;inset:0;z-index:2147483647;pointer-events:none;";
    overlayRoot = overlayHost.attachShadow({ mode: "closed" });
    const style = document.createElement("style");
    style.textContent = `
      .badge { position: fixed; transform: translate(6px, 6px); max-width: 190px;
        padding: 7px 9px; border-radius: 8px; color: #fff; font: 600 12px/1.2
        -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; box-shadow:
        0 4px 16px rgba(0,0,0,.28); border: 1px solid rgba(255,255,255,.4);
        backdrop-filter: blur(5px); }
      .badge small { display:block; margin-top:3px; font-size:10px; font-weight:500; opacity:.9; }
      .aigc { background: rgba(185, 28, 28, .94); }
      .clear { background: rgba(21, 128, 61, .94); }
      .error { background: rgba(161, 98, 7, .94); }
    `;
    overlayRoot.append(style);
    document.documentElement.append(overlayHost);
  }

  function updatePositions() {
    positionFrame = null;
    if (!overlayRoot) return;
    for (const [id, item] of renderedResults) {
      const element = candidateElements.get(id);
      const badge = item.badge;
      if (!element?.isConnected || latestElementIds.get(element) !== id) {
        badge.style.display = "none";
        continue;
      }
      const rect = element.getBoundingClientRect();
      const visible = rect.bottom > 0 && rect.right > 0 && rect.top < innerHeight && rect.left < innerWidth;
      badge.style.display = visible ? "block" : "none";
      badge.style.left = `${Math.max(0, rect.left)}px`;
      badge.style.top = `${Math.max(0, rect.top)}px`;
    }
  }

  function schedulePositionUpdate() {
    if (positionFrame === null) positionFrame = requestAnimationFrame(updatePositions);
  }

  function clearOverlays() {
    renderedResults.clear();
    if (overlayHost?.isConnected) overlayHost.remove();
    overlayHost = null;
    overlayRoot = null;
  }

  function clearAll() {
    clearOverlays();
    candidateElements.clear();
  }

  function render(results) {
    clearOverlays();
    ensureOverlay();
    for (const item of Array.isArray(results) ? results : []) {
      const label = shared.resultLabel(item.result);
      const badge = document.createElement("div");
      badge.className = `badge ${label.tone}`;
      badge.textContent = label.text;
      const detail = document.createElement("small");
      detail.textContent = label.detail;
      badge.append(detail);
      overlayRoot.append(badge);
      renderedResults.set(item.candidate.id, { badge, item });
    }
    schedulePositionUpdate();
    return { rendered: renderedResults.size };
  }

  addEventListener("scroll", schedulePositionUpdate, { passive: true, capture: true });
  addEventListener("resize", schedulePositionUpdate, { passive: true });

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    try {
      if (message?.type === "discover_candidates") {
        sendResponse({
          ok: true,
          ...discover(message.settings, {
            append: Boolean(message.append),
            excludeIds: message.excludeIds,
          }),
        });
      } else if (message?.type === "page_scan_start") {
        sendResponse({ ok: true, ...startPageScan() });
      } else if (message?.type === "page_scan_advance") {
        sendResponse({ ok: true, ...advancePageScan() });
      } else if (message?.type === "page_scan_restore") {
        sendResponse({ ok: true, ...restorePageScan() });
      } else if (message?.type === "render_results") {
        sendResponse({ ok: true, ...render(message.results) });
      } else if (message?.type === "clear_results") {
        clearAll();
        sendResponse({ ok: true });
      }
    } catch (error) {
      sendResponse({ ok: false, error: error?.message || String(error) });
    }
    return false;
  });
})();
