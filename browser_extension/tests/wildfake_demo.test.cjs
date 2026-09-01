"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const repoRoot = path.resolve(__dirname, "../..");
const html = fs.readFileSync(
  path.join(repoRoot, "browser_extension/wildfake_demo/index.html"),
  "utf8",
);
const app = fs.readFileSync(
  path.join(repoRoot, "browser_extension/wildfake_demo/app.js"),
  "utf8",
);

test("WildFake wall states the no-training boundary", () => {
  assert.match(html, /must not be used for training or threshold tuning/i);
  assert.match(app, /manifest\.training_allowed !== false/);
});

test("image candidates receive label-neutral alt text and filenames", () => {
  assert.match(app, /image\.alt = `Evaluation image \$\{item\.id\}`/);
  assert.match(app, /image\.src = new URL\(item\.file/);
  assert.doesNotMatch(app, /image\.(alt|title|dataset)[^\n]*ground_truth/);
});

test("ground truth is attached to a sibling caption, not the image", () => {
  assert.match(app, /const caption = document\.createElement\("figcaption"\)/);
  assert.match(app, /truthValue\.textContent = groundTruthCopy\(item\)/);
  assert.match(app, /figure\.append\(image, caption\)/);
});
