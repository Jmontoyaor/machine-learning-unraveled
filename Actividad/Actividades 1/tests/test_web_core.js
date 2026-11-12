// Run: node "Actividad/Actividades 1/tests/test_web_core.js"  (after python export_web.py)
"use strict";
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const docs = path.resolve(__dirname, "..", "..", "..", "docs", "mnist");
const core = require(path.join(docs, "mnist_core.js"));
const sandbox = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(docs, "model_data.js"), "utf8"), sandbox);
const data = sandbox.window.MNIST_DATA;

let passed = 0;
function test(name, fn) { fn(); passed++; console.log("ok -", name); }

function bar(width, height, bg, fg) {
  const img = new Uint8Array(width * height).fill(bg);
  for (let y = 15; y < 85; y++) for (let x = 44; x < 56; x++) img[y * width + x] = fg;
  return img;
}

test("JS forward pass matches Keras on real MNIST test images", () => {
  const images = core.decodeFloat32(data.verify_images);
  for (const kind of ["dense", "cnn"]) {
    const layers = core.loadModel(data.models[kind]);
    const expected = core.decodeFloat32(data.models[kind].verify_expected);
    let maxDiff = 0;
    for (let n = 0; n < data.verify_count; n++) {
      const probs = core.predict(layers, images.subarray(n * 784, (n + 1) * 784)).probabilities;
      for (let k = 0; k < 10; k++) maxDiff = Math.max(maxDiff, Math.abs(probs[k] - expected[n * 10 + k]));
    }
    assert(maxDiff < 1e-4, `${kind}: max probability difference ${maxDiff}`);
    console.log(`   ${kind}: max |JS - Keras| = ${maxDiff.toExponential(2)}`);
  }
});

test("white-on-black and black-on-white give the same canonical image", () => {
  const a = core.preprocess(bar(100, 100, 0, 255), 100, 100).image;
  const r = core.preprocess(bar(100, 100, 255, 0), 100, 100);
  assert.strictEqual(r.info.inverted, true);
  for (let i = 0; i < 784; i++) assert(Math.abs(a[i] - r.image[i]) < 1e-6);
  let max = 0, min = 1;
  for (const v of a) { max = Math.max(max, v); min = Math.min(min, v); }
  assert(max > 0.9 && min === 0);
});

test("empty canvases are rejected in both polarities", () => {
  for (const level of [0, 255]) {
    assert.throws(() => core.preprocess(new Uint8Array(10000).fill(level), 100, 100), /empty/);
  }
});

test("camera cleaning removes grey paper and shadow", () => {
  const img = bar(100, 100, 200, 30);                        // dark digit on grey paper
  for (let y = 0; y < 100; y++) for (let x = 70; x < 100; x++) if (img[y * 100 + x] === 200) img[y * 100 + x] = 150; // shadow
  const r = core.preprocess(img, 100, 100, { clean: true });
  assert.strictEqual(r.info.inverted, true);
  const background = r.image[0] + r.image[27] + r.image[783];
  assert.strictEqual(background, 0);
});

test("digit is centred by mass", () => {
  const img = new Uint8Array(10000);
  for (let y = 10; y < 40; y++) for (let x = 12; x < 20; x++) img[y * 100 + x] = 255;
  const out = core.preprocess(img, 100, 100).image;
  let t = 0, sr = 0, sc = 0;
  for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) { const v = out[r * 28 + c]; t += v; sr += r * v; sc += c * v; }
  assert(Math.abs(sr / t - 14) < 1 && Math.abs(sc / t - 14) < 1);
});

console.log(`${passed} tests passed`);
