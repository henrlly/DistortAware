"use strict";

const MANIFEST_URL = "../../data/wildfake_validation_demo/manifest.json";
const gallery = document.querySelector("#gallery");
const summary = document.querySelector("#summary");
const errorPanel = document.querySelector("#error-panel");
const errorCopy = document.querySelector("#error-copy");
const truthToggle = document.querySelector("#truth-toggle");

function groundTruthCopy(item) {
  return item.ground_truth === "real" ? "REAL" : "AIGC";
}

function sampleCard(item) {
  const figure = document.createElement("figure");
  figure.className = "sample-card";

  const image = document.createElement("img");
  image.src = new URL(item.file, new URL(MANIFEST_URL, location.href)).href;
  image.alt = `Evaluation image ${item.id}`;
  image.loading = "eager";
  image.decoding = "async";

  const caption = document.createElement("figcaption");
  const number = document.createElement("span");
  number.className = "sample-number";
  number.textContent = `Evaluation image ${item.id}`;

  const truthRow = document.createElement("div");
  truthRow.className = "truth-row";
  const truthLabel = document.createElement("span");
  truthLabel.className = "truth-label";
  truthLabel.textContent = "Human-visible ground truth";
  const truthValue = document.createElement("strong");
  truthValue.className = `truth-value ${item.ground_truth}`;
  truthValue.textContent = groundTruthCopy(item);
  truthRow.append(truthLabel, truthValue);

  const source = document.createElement("p");
  source.className = "source";
  source.textContent = item.source_display;
  caption.append(number, truthRow, source);
  figure.append(image, caption);
  return figure;
}

async function loadManifest() {
  const response = await fetch(MANIFEST_URL, { cache: "no-store" });
  if (!response.ok) throw new Error(`manifest request returned HTTP ${response.status}`);
  const manifest = await response.json();
  if (manifest.training_allowed !== false || !Array.isArray(manifest.items)) {
    throw new Error("manifest does not declare the required validation-only boundary");
  }
  gallery.replaceChildren(...manifest.items.map(sampleCard));
  const real = manifest.items.filter((item) => item.ground_truth === "real").length;
  const aigc = manifest.items.filter((item) => item.ground_truth === "aigc").length;
  summary.textContent = `${manifest.items.length} images · ${real} real · ${aigc} AIGC`;
}

truthToggle.addEventListener("click", () => {
  const hidden = document.body.classList.toggle("ground-truth-hidden");
  truthToggle.textContent = hidden ? "Reveal ground truth" : "Hide ground truth";
  truthToggle.setAttribute("aria-pressed", String(!hidden));
});

loadManifest().catch((error) => {
  summary.textContent = "Local validation subset unavailable";
  gallery.hidden = true;
  errorPanel.hidden = false;
  errorCopy.textContent = `The page could not load its generated manifest: ${error.message}. From the repository root, run:`;
});
