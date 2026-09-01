const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const extensionRoot = path.resolve(__dirname, "..");
const manifest = JSON.parse(fs.readFileSync(path.join(extensionRoot, "manifest.json"), "utf8"));

test("manifest keeps least-privilege permissions and loopback-only hosts", () => {
  assert.equal(manifest.manifest_version, 3);
  assert.deepEqual(manifest.permissions.sort(), ["activeTab", "scripting", "storage"]);
  assert.deepEqual(manifest.host_permissions.sort(), [
    "http://127.0.0.1/*",
    "http://localhost/*",
  ]);
  assert.ok(!manifest.permissions.includes("tabs"));
  assert.ok(!manifest.host_permissions.some((item) => item.includes("*://*")));
});

test("every declared extension entrypoint exists", () => {
  const relativePaths = [
    manifest.background.service_worker,
    manifest.action.default_popup,
    manifest.options_page,
    "content.js",
    "lib/shared.js",
  ];
  for (const relativePath of relativePaths) {
    assert.ok(fs.existsSync(path.join(extensionRoot, relativePath)), relativePath);
  }
});

test("extension source contains no remote executable scripts", () => {
  for (const relativePath of [
    "service_worker.js",
    "content.js",
    "popup/popup.js",
    "options/options.js",
  ]) {
    const source = fs.readFileSync(path.join(extensionRoot, relativePath), "utf8");
    assert.doesNotMatch(source, /<script[^>]+https?:|importScripts\(["']https?:/i);
  }
});
