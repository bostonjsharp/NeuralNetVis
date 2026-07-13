// Trains the 784→16→16→10 MLP the app visualizes and exports its weights.
//
//   node scripts/train-mnist.mjs
//
// Downloads MNIST once into .mnist-cache/ (gitignored), trains with plain
// seeded SGD, and writes the committed runtime assets:
//   src/assets/weights.json  — base64 Float32 W/b per layer + metadata
//   src/assets/samples.json  — 60 test digits (6 per class) for the attract loop
//
// The app never trains or downloads anything; these assets are the contract.

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { get } from "node:https";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const CACHE = join(ROOT, ".mnist-cache");
const ASSETS = join(ROOT, "src", "assets");

const MIRRORS = [
  "https://ossci-datasets.s3.amazonaws.com/mnist/",
  "https://storage.googleapis.com/cvdf-datasets/mnist/",
];
const FILES = {
  trainImages: "train-images-idx3-ubyte.gz",
  trainLabels: "train-labels-idx1-ubyte.gz",
  testImages: "t10k-images-idx3-ubyte.gz",
  testLabels: "t10k-labels-idx1-ubyte.gz",
};

// ---------------------------------------------------------------- download

function fetchBuffer(url) {
  return new Promise((resolve, reject) => {
    get(url, (res) => {
      if (res.statusCode !== 200) {
        res.resume();
        reject(new Error(`${url} → HTTP ${res.statusCode}`));
        return;
      }
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => resolve(Buffer.concat(chunks)));
      res.on("error", reject);
    }).on("error", reject);
  });
}

async function ensureFile(name) {
  const path = join(CACHE, name);
  if (existsSync(path)) return path;
  mkdirSync(CACHE, { recursive: true });
  let lastErr;
  for (const mirror of MIRRORS) {
    try {
      console.log(`downloading ${mirror}${name} ...`);
      const buf = await fetchBuffer(mirror + name);
      writeFileSync(path, buf);
      return path;
    } catch (err) {
      lastErr = err;
      console.warn(`  mirror failed: ${err.message}`);
    }
  }
  throw lastErr;
}

// ------------------------------------------------------------------- parse

function parseIdxImages(path) {
  const buf = gunzipSync(readFileSync(path));
  if (buf.readUInt32BE(0) !== 2051) throw new Error(`${path}: bad magic`);
  const n = buf.readUInt32BE(4);
  const rows = buf.readUInt32BE(8);
  const cols = buf.readUInt32BE(12);
  if (rows !== 28 || cols !== 28) throw new Error(`${path}: not 28×28`);
  return { n, pixels: new Uint8Array(buf.buffer, buf.byteOffset + 16, n * 784) };
}

function parseIdxLabels(path) {
  const buf = gunzipSync(readFileSync(path));
  if (buf.readUInt32BE(0) !== 2049) throw new Error(`${path}: bad magic`);
  const n = buf.readUInt32BE(4);
  return new Uint8Array(buf.buffer, buf.byteOffset + 8, n);
}

// ------------------------------------------------------------------- model

const SHAPE = [784, 16, 16, 10];

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Layers as { W: Float32Array(out*in) row-major [out][in], b: Float32Array(out) }. */
function initNet(rand) {
  const layers = [];
  for (let l = 0; l < SHAPE.length - 1; l++) {
    const [nIn, nOut] = [SHAPE[l], SHAPE[l + 1]];
    const W = new Float32Array(nOut * nIn);
    const std = Math.sqrt(2 / nIn); // He init for ReLU
    for (let i = 0; i < W.length; i++) {
      // Box–Muller from two uniform draws
      const u = Math.max(rand(), 1e-12);
      const v = rand();
      W[i] = std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    layers.push({ W, b: new Float32Array(nOut) });
  }
  return layers;
}

/**
 * Forward + backward for one sample; accumulates gradients in-place.
 * Returns 1 if prediction was correct (for train accuracy logging).
 */
function trainSample(layers, grads, acts, x, label) {
  const L = layers.length;
  acts[0].set(x);
  // forward: ReLU on hidden layers, raw logits on the last
  for (let l = 0; l < L; l++) {
    const { W, b } = layers[l];
    const [aIn, aOut] = [acts[l], acts[l + 1]];
    const nIn = SHAPE[l];
    for (let o = 0; o < SHAPE[l + 1]; o++) {
      let sum = b[o];
      const row = o * nIn;
      for (let i = 0; i < nIn; i++) sum += W[row + i] * aIn[i];
      aOut[o] = l < L - 1 ? Math.max(0, sum) : sum;
    }
  }
  const logits = acts[L];
  let maxLogit = -Infinity;
  let argmax = 0;
  for (let o = 0; o < 10; o++) {
    if (logits[o] > maxLogit) {
      maxLogit = logits[o];
      argmax = o;
    }
  }
  let expSum = 0;
  const delta = new Float32Array(10);
  for (let o = 0; o < 10; o++) {
    delta[o] = Math.exp(logits[o] - maxLogit);
    expSum += delta[o];
  }
  // softmax cross-entropy gradient: p - onehot
  for (let o = 0; o < 10; o++) delta[o] = delta[o] / expSum - (o === label ? 1 : 0);

  // backward
  let dOut = delta;
  for (let l = L - 1; l >= 0; l--) {
    const { W } = layers[l];
    const g = grads[l];
    const aIn = acts[l];
    const nIn = SHAPE[l];
    const nOut = SHAPE[l + 1];
    const dIn = l > 0 ? new Float32Array(nIn) : null;
    for (let o = 0; o < nOut; o++) {
      const d = dOut[o];
      if (d === 0) continue;
      const row = o * nIn;
      g.b[o] += d;
      const gW = g.W;
      for (let i = 0; i < nIn; i++) {
        gW[row + i] += d * aIn[i];
        if (dIn) dIn[i] += d * W[row + i];
      }
    }
    if (dIn) {
      // ReLU derivative w.r.t. the hidden activation
      for (let i = 0; i < nIn; i++) if (acts[l][i] <= 0) dIn[i] = 0;
      dOut = dIn;
    }
  }
  return argmax === label ? 1 : 0;
}

function predict(layers, acts, x) {
  acts[0].set(x);
  const L = layers.length;
  for (let l = 0; l < L; l++) {
    const { W, b } = layers[l];
    const [aIn, aOut] = [acts[l], acts[l + 1]];
    const nIn = SHAPE[l];
    for (let o = 0; o < SHAPE[l + 1]; o++) {
      let sum = b[o];
      const row = o * nIn;
      for (let i = 0; i < nIn; i++) sum += W[row + i] * aIn[i];
      aOut[o] = l < L - 1 ? Math.max(0, sum) : sum;
    }
  }
  const logits = acts[L];
  let best = 0;
  for (let o = 1; o < 10; o++) if (logits[o] > logits[best]) best = o;
  return best;
}

// -------------------------------------------------------------------- main

async function main() {
  const paths = {};
  for (const [key, name] of Object.entries(FILES)) paths[key] = await ensureFile(name);

  const train = parseIdxImages(paths.trainImages);
  const trainLabels = parseIdxLabels(paths.trainLabels);
  const test = parseIdxImages(paths.testImages);
  const testLabels = parseIdxLabels(paths.testLabels);
  console.log(`train ${train.n}, test ${test.n}`);

  const rand = mulberry32(0x5eed);
  const layers = initNet(rand);
  const acts = SHAPE.map((n) => new Float32Array(n));

  const EPOCHS = 20;
  const BATCH = 64;
  const order = Uint32Array.from({ length: train.n }, (_, i) => i);
  const x = new Float32Array(784);

  const testAccuracy = () => {
    let correct = 0;
    for (let s = 0; s < test.n; s++) {
      for (let i = 0; i < 784; i++) x[i] = test.pixels[s * 784 + i] / 255;
      if (predict(layers, acts, x) === testLabels[s]) correct++;
    }
    return correct / test.n;
  };

  let acc = 0;
  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    // Fisher–Yates with the seeded PRNG so runs are reproducible
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    const lr = 0.1 / (1 + 0.08 * (epoch - 1));
    const grads = layers.map(({ W, b }) => ({
      W: new Float32Array(W.length),
      b: new Float32Array(b.length),
    }));
    let trainCorrect = 0;
    for (let start = 0; start < train.n; start += BATCH) {
      const count = Math.min(BATCH, train.n - start);
      for (const g of grads) {
        g.W.fill(0);
        g.b.fill(0);
      }
      for (let k = 0; k < count; k++) {
        const s = order[start + k];
        for (let i = 0; i < 784; i++) x[i] = train.pixels[s * 784 + i] / 255;
        trainCorrect += trainSample(layers, grads, acts, x, trainLabels[s]);
      }
      const scale = lr / count;
      for (let l = 0; l < layers.length; l++) {
        const { W, b } = layers[l];
        for (let i = 0; i < W.length; i++) W[i] -= scale * grads[l].W[i];
        for (let i = 0; i < b.length; i++) b[i] -= scale * grads[l].b[i];
      }
    }
    acc = testAccuracy();
    console.log(
      `epoch ${epoch}/${EPOCHS}  lr ${lr.toFixed(4)}  ` +
        `train ${((trainCorrect / train.n) * 100).toFixed(2)}%  ` +
        `test ${(acc * 100).toFixed(2)}%`
    );
  }

  if (acc < 0.92) {
    throw new Error(`test accuracy ${(acc * 100).toFixed(2)}% is below the 92% bar — not exporting`);
  }

  // ---- export weights.json
  const b64 = (arr) => Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString("base64");
  mkdirSync(ASSETS, { recursive: true });
  const weightsJson = {
    shape: SHAPE,
    testAccuracy: Number(acc.toFixed(4)),
    trainedAt: new Date().toISOString().slice(0, 10),
    layers: layers.map(({ W, b }) => ({ W: b64(W), b: b64(b) })),
  };
  writeFileSync(join(ASSETS, "weights.json"), JSON.stringify(weightsJson));
  console.log(`wrote src/assets/weights.json (test accuracy ${(acc * 100).toFixed(2)}%)`);

  // ---- export samples.json: per class, 5 confident-correct + 1 ambiguous-correct
  const softmaxConfidence = (s) => {
    for (let i = 0; i < 784; i++) x[i] = test.pixels[s * 784 + i] / 255;
    const pred = predict(layers, acts, x);
    const logits = acts[SHAPE.length - 1];
    let max = -Infinity;
    for (let o = 0; o < 10; o++) max = Math.max(max, logits[o]);
    let sum = 0;
    for (let o = 0; o < 10; o++) sum += Math.exp(logits[o] - max);
    return { pred, confidence: Math.exp(logits[pred] - max) / sum };
  };
  const samples = [];
  for (let digit = 0; digit < 10; digit++) {
    const correct = [];
    for (let s = 0; s < test.n; s++) {
      if (testLabels[s] !== digit) continue;
      const { pred, confidence } = softmaxConfidence(s);
      if (pred === digit) correct.push({ s, confidence });
    }
    correct.sort((a, b) => b.confidence - a.confidence);
    const picks = [...correct.slice(0, 5), correct[correct.length - 1]];
    for (const { s } of picks) {
      samples.push({
        label: digit,
        pixels: Buffer.from(test.pixels.subarray(s * 784, s * 784 + 784)).toString("base64"),
      });
    }
  }
  writeFileSync(join(ASSETS, "samples.json"), JSON.stringify({ samples }));
  console.log(`wrote src/assets/samples.json (${samples.length} digits)`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
