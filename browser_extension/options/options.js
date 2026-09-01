/* global AigcShared */
"use strict";

const form = document.querySelector("#settings-form");
const fields = {
  detectorMode: document.querySelector("#detector-mode"),
  originalServiceUrl: document.querySelector("#original-service-url"),
  originalApiToken: document.querySelector("#original-api-token"),
  prismguardServiceUrl: document.querySelector("#prismguard-service-url"),
  prismguardApiToken: document.querySelector("#prismguard-api-token"),
  demoServiceUrl: document.querySelector("#demo-service-url"),
  demoApiToken: document.querySelector("#demo-api-token"),
  scanScope: document.querySelector("#scan-scope"),
  maxImages: document.querySelector("#max-images"),
  minDimension: document.querySelector("#min-dimension"),
  minArea: document.querySelector("#min-area"),
  includePhysics: document.querySelector("#include-physics"),
  includeArtifacts: document.querySelector("#include-artifacts"),
};
const testButton = document.querySelector("#test-button");
const connectionStatus = document.querySelector("#connection-status");
const saveStatus = document.querySelector("#save-status");

function formSettings() {
  const settings = AigcShared.normalizeSettings({
    detectorMode: fields.detectorMode.value,
    originalServiceUrl: AigcShared.validateServiceUrl(fields.originalServiceUrl.value),
    originalApiToken: fields.originalApiToken.value,
    prismguardServiceUrl: AigcShared.validateServiceUrl(fields.prismguardServiceUrl.value),
    prismguardApiToken: fields.prismguardApiToken.value,
    demoServiceUrl: AigcShared.validateServiceUrl(fields.demoServiceUrl.value),
    demoApiToken: fields.demoApiToken.value,
    scanScope: fields.scanScope.value,
    maxImages: fields.maxImages.value,
    minDimension: fields.minDimension.value,
    minArea: fields.minArea.value,
    includePhysics: fields.includePhysics.checked,
    includeArtifacts: fields.includeArtifacts.checked,
  });
  if (!settings.apiToken) throw new Error("The selected detector token cannot be empty.");
  return settings;
}

function fill(settings) {
  fields.detectorMode.value = settings.detectorMode;
  fields.originalServiceUrl.value = settings.originalServiceUrl;
  fields.originalApiToken.value = settings.originalApiToken;
  fields.prismguardServiceUrl.value = settings.prismguardServiceUrl;
  fields.prismguardApiToken.value = settings.prismguardApiToken;
  fields.demoServiceUrl.value = settings.demoServiceUrl;
  fields.demoApiToken.value = settings.demoApiToken;
  fields.scanScope.value = settings.scanScope;
  fields.maxImages.value = String(settings.maxImages);
  fields.minDimension.value = String(settings.minDimension);
  fields.minArea.value = String(settings.minArea);
  fields.includePhysics.checked = settings.includePhysics;
  fields.includeArtifacts.checked = settings.includeArtifacts;
}

function show(element, message, tone = "") {
  element.textContent = message;
  element.className = element === connectionStatus ? `status ${tone}` : tone;
}

testButton.addEventListener("click", async () => {
  testButton.disabled = true;
  show(connectionStatus, "Connecting to the local service…");
  try {
    const settings = formSettings();
    const response = await chrome.runtime.sendMessage({ type: "service_health", settings });
    if (!response?.ok) throw new Error(response?.error || "Health check failed.");
    const health = response.result;
    AigcShared.validateHealthForMode(health, settings.detectorMode);
    const detector = health.detector || {};
    const hash = detector.checkpoint_sha256 ? detector.checkpoint_sha256.slice(0, 12) : "unreported";
    show(
      connectionStatus,
      `Ready: ${AigcShared.DETECTOR_MODES[settings.detectorMode].label} · ${detector.arch || detector.family || "detector"}, checkpoint ${hash}…, physics ${health.physics_profile || "off"}, artifacts ${health.artifact?.status || "disabled"}.`,
      "success",
    );
  } catch (error) {
    show(connectionStatus, error?.message || String(error), "error");
  } finally {
    testButton.disabled = false;
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const settings = formSettings();
    await chrome.storage.local.set(settings);
    fill(settings);
    show(saveStatus, "Saved locally.", "success");
  } catch (error) {
    show(saveStatus, error?.message || String(error), "error");
  }
});

(async function initialize() {
  const stored = await chrome.storage.local.get(AigcShared.DEFAULT_SETTINGS);
  fill(AigcShared.normalizeSettings(stored));
})();
