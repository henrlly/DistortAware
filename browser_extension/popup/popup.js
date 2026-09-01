/* global AigcShared */
"use strict";

const elements = {
  settings: document.querySelector("#settings-button"),
  scan: document.querySelector("#scan-button"),
  clear: document.querySelector("#clear-button"),
  serviceDot: document.querySelector("#service-dot"),
  serviceTitle: document.querySelector("#service-title"),
  serviceDetail: document.querySelector("#service-detail"),
  progressPanel: document.querySelector("#progress-panel"),
  progressLabel: document.querySelector("#progress-label"),
  progressCount: document.querySelector("#progress-count"),
  progressBar: document.querySelector("#progress-bar"),
  summaryPanel: document.querySelector("#summary-panel"),
  summaryTotal: document.querySelector("#summary-total"),
  summaryAigc: document.querySelector("#summary-aigc"),
  summaryClear: document.querySelector("#summary-clear"),
  summaryErrors: document.querySelector("#summary-errors"),
  empty: document.querySelector("#empty-message"),
  scanNote: document.querySelector("#scan-note"),
  results: document.querySelector("#result-list"),
};

function setService(tone, title, detail) {
  elements.serviceDot.className = `status-dot ${tone}`;
  elements.serviceTitle.textContent = title;
  elements.serviceDetail.textContent = detail;
}

function setBusy(busy) {
  elements.scan.disabled = busy;
  elements.clear.disabled = busy;
  elements.progressPanel.hidden = !busy;
  if (!busy) {
    elements.progressBar.max = 1;
    elements.progressBar.value = 0;
    elements.progressCount.textContent = "";
  }
}

async function send(message) {
  const response = await chrome.runtime.sendMessage(message);
  if (!response?.ok) throw new Error(response?.error || "Extension request failed.");
  return response.result;
}

async function currentTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  const tab = tabs[0];
  if (!Number.isInteger(tab?.id) || !Number.isInteger(tab?.windowId)) {
    throw new Error("No active browser tab is available.");
  }
  const protocol = (() => {
    try { return new URL(tab.url || "").protocol; } catch (_error) { return ""; }
  })();
  if (!["http:", "https:", "file:"].includes(protocol)) {
    throw new Error("This protected browser page cannot be captured. Open a regular webpage.");
  }
  return tab;
}

function compactKind(kind) {
  return {
    image: "Image",
    poster: "Video poster",
    video_frame: "Video frame",
    background: "Background image",
  }[kind] || "Visual region";
}

function physicsDescription(result) {
  const explanation = result?.explanation;
  const physics = explanation?.physics;
  if (physics?.aggregate) {
    const aggregate = physics.aggregate;
    const applicable = Array.isArray(aggregate.applicable_cues)
      ? aggregate.applicable_cues.length
      : 0;
    const aligned = explanation.dino_physics_alignment?.applicable
      ? " · DINO/physics spatial association available"
      : "";
    const cues = physics.cues || {};
    const shadow = cues.cast_shadow?.measurements || {};
    const reflection = cues.reflection?.measurements || {};
    const proposals = ` · shadow ${Number(shadow.accepted_shadow_pairs) || 0}/${Number(shadow.required_pairs) || 3} pairs · reflection ${Number(reflection.accepted_reflection_pairs) || 0}/${Number(reflection.required_pairs) || 3} pairs`;
    return `Physics: ${String(aggregate.status || "indeterminate").replaceAll("_", " ")} · ${applicable} applicable cue${applicable === 1 ? "" : "s"}${proposals}${aligned}`;
  }
  if (explanation?.physics_requested) {
    return `Physics requested · service profile ${explanation.physics_profile || "off"}`;
  }
  return "Physics not requested (faster default)";
}

function physicsCueDescription(result) {
  const cues = result?.explanation?.physics?.cues;
  if (!cues || typeof cues !== "object") return "";
  const statuses = Object.entries(cues).map(([name, cue]) => {
    const label = name.replaceAll("_", " ");
    const status = String(cue?.status || "indeterminate").replaceAll("_", " ");
    const measurements = cue?.measurements || {};
    let counts = "";
    if (name === "cast_shadow") {
      counts = ` (${Number(measurements.candidate_shadow_regions) || 0} region(s), ${Number(measurements.accepted_shadow_pairs) || 0}/${Number(measurements.required_pairs) || 3} pairs)`;
    } else if (name === "reflection") {
      const backend = measurements.feature_backend?.backend
        ? ` via ${String(measurements.feature_backend.backend).replaceAll("_", " ")}`
        : "";
      const selection = measurements.feature_selection;
      const fallback = selection
        ? `; shared DINO had ${Number(selection.primary_accepted_pairs) || 0}, appearance fallback had ${Number(selection.fallback_accepted_pairs) || 0}`
        : "";
      counts = ` (${Number(measurements.candidate_mirror_regions) || 0} region(s), ${Number(measurements.accepted_reflection_pairs) || 0}/${Number(measurements.required_pairs) || 3} matches${backend}${fallback})`;
    } else if (name === "perspective") {
      counts = ` (${Number(measurements.retained_count) || 0} structural lines retained)`;
    }
    const reason = cue?.summary ? ` — ${cue.summary}` : "";
    return `${label}: ${status}${counts}${reason}`;
  });
  return statuses.length ? `Physics cues — ${statuses.join("; ")}. ` : "";
}

function artifactDescription(result) {
  const explanation = result?.explanation;
  const artifact = explanation?.artifact;
  if (artifact?.status === "available") {
    const label = String(artifact.predicted_class || "unknown").replaceAll("_", " ");
    const signal = Number(artifact.ai_signal_score);
    const area = Number(artifact.evidence_mask?.area_fraction_at_0_5);
    return `Residual artifacts: ${label}${Number.isFinite(signal) ? ` · AI signal ${(100 * signal).toFixed(1)}%` : ""}${Number.isFinite(area) ? ` · highlighted area ${(100 * area).toFixed(1)}%` : ""} · explanation only`;
  }
  if (explanation?.artifact_requested) {
    return `Residual artifacts requested · ${artifact?.reason || `service profile ${explanation.artifact_profile || "off"}`}`;
  }
  return "Residual artifacts not requested";
}

function artifactEvidenceDescription(result) {
  const artifact = result?.explanation?.artifact;
  if (artifact?.status !== "available") return "";
  const scores = artifact.class_scores || {};
  const entries = Object.entries(scores).map(
    ([name, value]) => `${name.replaceAll("_", " ")} ${(100 * Number(value)).toFixed(1)}%`,
  );
  const bbox = artifact.evidence_mask?.normalized_bbox_at_0_5;
  const boxText = Array.isArray(bbox)
    ? ` Weak mask bounding box (normalised x1,y1,x2,y2): ${bbox.join(", ")}.`
    : " No mask area crossed the 0.5 display threshold.";
  return `Residual class scores — ${entries.join("; ")}.${boxText} This sidecar does not vote on the verdict. `;
}

function resultCard(item, index) {
  const candidate = item.candidate || {};
  const result = item.result || {};
  const label = AigcShared.resultLabel(result);
  const card = document.createElement("li");
  card.className = "result-card";

  const head = document.createElement("div");
  head.className = "result-head";
  const name = document.createElement("span");
  name.className = "result-name";
  name.textContent = `${index + 1}. ${compactKind(candidate.kind)}${candidate.alt ? ` · ${candidate.alt}` : ""}`;
  const pill = document.createElement("span");
  pill.className = `pill ${label.tone}`;
  pill.textContent = label.text;
  head.append(name, pill);

  const detail = document.createElement("p");
  detail.className = "result-detail";
  detail.textContent = label.detail;

  const rect = candidate.rect || {};
  const timing = Number(result.timing_ms);
  const metadata = document.createElement("p");
  metadata.className = "result-meta";
  const size = Number.isFinite(rect.width) && Number.isFinite(rect.height)
    ? `${Math.round(rect.width)}×${Math.round(rect.height)} visible pixels`
    : "visible viewport crop";
  metadata.textContent = `${size}${Number.isFinite(timing) ? ` · ${timing.toFixed(0)} ms` : ""}`;

  const physics = document.createElement("p");
  physics.className = "result-meta";
  physics.textContent = result.error ? "No explanation available" : physicsDescription(result);

  const artifact = document.createElement("p");
  artifact.className = "result-meta";
  artifact.textContent = result.error ? "No artifact evidence available" : artifactDescription(result);

  card.append(head, detail, metadata, physics, artifact);

  if (!result.error) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Evidence and limitations";
    const copy = document.createElement("p");
    const patch = result.explanation?.patch_evidence;
    const patchText = patch
      ? `Weak patch map ${patch.grid_shape?.join("×") || "available"}; mean ${Number(patch.mean).toFixed(3)}, max ${Number(patch.maximum).toFixed(3)}. `
      : "No patch summary. ";
    const cueText = physicsCueDescription(result);
    const artifactText = artifactEvidenceDescription(result);
    const limitation = Array.isArray(result.limitations) && result.limitations.length
      ? result.limitations[0]
      : "Treat this result as supporting evidence, not proof.";
    copy.textContent = patchText + cueText + artifactText + limitation;
    details.append(summary, copy);
    card.append(details);
  }
  return card;
}

function renderScan(scan) {
  elements.results.replaceChildren();
  const results = Array.isArray(scan?.results) ? scan.results : [];
  const summary = scan?.summary || AigcShared.summarizeResults(results);
  elements.summaryTotal.textContent = String(summary.total || 0);
  elements.summaryAigc.textContent = String(summary.aigc || 0);
  elements.summaryClear.textContent = String(summary.clear || 0);
  elements.summaryErrors.textContent = String(summary.errors || 0);
  elements.summaryPanel.hidden = !scan || scan?.detectorMode === "demo_fixture";
  elements.empty.hidden = results.length > 0;
  const pageMode = scan?.scanScope === "page";
  elements.empty.textContent = scan
    ? pageMode
      ? "No eligible rendered image regions were found within the bounded page scan."
      : "No eligible image regions were visible. Scroll to an image and scan again."
    : "Click Scan to inspect rendered image regions in the active tab.";
  elements.scanNote.hidden = !scan;
  if (scan) {
    const scope = pageMode ? "Whole-page auto-scroll" : "Visible viewport";
    const passes = Number(scan.viewportsVisited) || 1;
    const limit = scan.truncated ? " · stopped at the configured safety limit" : "";
    const restored = pageMode
      ? scan.scrollRestored ? " · original scroll position restored" : " · could not confirm scroll restoration"
      : "";
    const fixture = scan.detectorMode === "demo_fixture"
      ? " · WIRING FIXTURE ONLY — no AIGC conclusions"
      : "";
    elements.scanNote.textContent = `${scope} · ${passes} viewport${passes === 1 ? "" : "s"}${limit}${restored}${fixture}.`;
  }
  results.forEach((item, index) => elements.results.append(resultCard(item, index)));
}

async function refreshHealth() {
  const settings = AigcShared.normalizeSettings(
    await chrome.storage.local.get(AigcShared.DEFAULT_SETTINGS),
  );
  if (!settings.apiToken) {
    setService("idle", "Service token not configured", "Open settings and paste the token printed by the local service.");
    return;
  }
  try {
    const health = await send({ type: "service_health" });
    const detector = health.detector || {};
    const detail = `${detector.arch || detector.family || "PatchHead"} · physics ${health.physics_profile || "off"} · artifacts ${health.artifact?.status || "disabled"}`;
    const demo = health.scientific_status === "plumbing_only_no_aigc_performance_claim";
    setService(
      "ready",
      demo ? "Wiring demo ready — not a detector" : "Local detector ready",
      detail,
    );
  } catch (error) {
    setService("error", "Local detector unavailable", error?.message || String(error));
  }
}

chrome.runtime.onMessage.addListener((message) => {
  if (message?.type !== "scan_progress") return;
  const completed = Number(message.completed) || 0;
  const total = Math.max(1, Number(message.total) || 1);
  elements.progressPanel.hidden = false;
  elements.progressLabel.textContent = message.phase === "discovering"
    ? `Scanning page viewport ${Number(message.pass) || 1}…`
    : "Running local detector and explainers…";
  elements.progressBar.max = total;
  elements.progressBar.value = completed;
  elements.progressCount.textContent = `${completed}/${total}`;
});

elements.scan.addEventListener("click", async () => {
  setBusy(true);
  elements.progressLabel.textContent = "Finding rendered image regions…";
  try {
    const tab = await currentTab();
    const scan = await send({ type: "scan_tab", tabId: tab.id, windowId: tab.windowId });
    renderScan(scan);
    await refreshHealth();
  } catch (error) {
    setService("error", "Scan could not start", error?.message || String(error));
  } finally {
    setBusy(false);
  }
});

elements.clear.addEventListener("click", async () => {
  try {
    const tab = await currentTab();
    await send({ type: "clear_tab", tabId: tab.id });
    renderScan(null);
  } catch (error) {
    setService("error", "Could not clear page badges", error?.message || String(error));
  }
});

elements.settings.addEventListener("click", () => chrome.runtime.openOptionsPage());

(async function initialize() {
  try {
    const settings = AigcShared.normalizeSettings(
      await chrome.storage.local.get(AigcShared.DEFAULT_SETTINGS),
    );
    elements.scan.textContent = settings.scanScope === "page"
      ? "Scan page images"
      : "Scan visible images";
    const tab = await currentTab();
    const stored = await chrome.storage.session.get("latestScan");
    renderScan(stored.latestScan?.tabId === tab.id ? stored.latestScan : null);
  } catch (_error) {
    renderScan(null);
  }
  await refreshHealth();
})();
