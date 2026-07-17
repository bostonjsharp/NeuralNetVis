// Trains the MLP variants the app visualizes and exports their weights.
//
//   node scripts/train-mnist.mjs            # classic (today's default)
//   node scripts/train-mnist.mjs wide       # one variant
//   node scripts/train-mnist.mjs all        # every variant
//
// Downloads MNIST once into .mnist-cache/ (gitignored), trains with plain
// seeded SGD, and writes the committed runtime assets:
//   src/assets/weights-<id>.json — base64 Float32 W/b per layer + metadata
//   src/assets/weights.json      — the classic variant, kept under its
//                                  original name (the app's default import)
//   src/assets/samples.json      — 60 test digits (6 per class) for the
//                                  attract loop, picked by the classic net
//
// The app never trains or downloads anything; these assets are the contract.

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

/**
 * The brains the exhibit can swap between. `minAcc` is an honest floor per
 * architecture — a linear softmax simply cannot reach the classic bar, and
 * that gap is the whole point of showing it.
 */
const VARIANTS = {
  linear: { shape: [784, 10], minAcc: 0.9, label: "Straight-through" },
  tiny: { shape: [784, 8, 10], minAcc: 0.9, label: "Tiny" },
  classic: { shape: [784, 16, 16, 10], minAcc: 0.92, label: "Classic" },
  wide: { shape: [784, 32, 32, 10], minAcc: 0.95, label: "Wide" },
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
function initNet(shape, rand) {
  const layers = [];
  for (let l = 0; l < shape.length - 1; l++) {
    const [nIn, nOut] = [shape[l], shape[l + 1]];
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
function trainSample(shape, layers, grads, acts, x, label) {
  const L = layers.length;
  const nClasses = shape[shape.length - 1];
  acts[0].set(x);
  // forward: ReLU on hidden layers, raw logits on the last
  for (let l = 0; l < L; l++) {
    const { W, b } = layers[l];
    const [aIn, aOut] = [acts[l], acts[l + 1]];
    const nIn = shape[l];
    for (let o = 0; o < shape[l + 1]; o++) {
      let sum = b[o];
      const row = o * nIn;
      for (let i = 0; i < nIn; i++) sum += W[row + i] * aIn[i];
      aOut[o] = l < L - 1 ? Math.max(0, sum) : sum;
    }
  }
  const logits = acts[L];
  let maxLogit = -Infinity;
  let argmax = 0;
  for (let o = 0; o < nClasses; o++) {
    if (logits[o] > maxLogit) {
      maxLogit = logits[o];
      argmax = o;
    }
  }
  let expSum = 0;
  const delta = new Float32Array(nClasses);
  for (let o = 0; o < nClasses; o++) {
    delta[o] = Math.exp(logits[o] - maxLogit);
    expSum += delta[o];
  }
  // softmax cross-entropy gradient: p - onehot
  for (let o = 0; o < nClasses; o++) delta[o] = delta[o] / expSum - (o === label ? 1 : 0);

  // backward
  let dOut = delta;
  for (let l = L - 1; l >= 0; l--) {
    const { W } = layers[l];
    const g = grads[l];
    const aIn = acts[l];
    const nIn = shape[l];
    const nOut = shape[l + 1];
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

function predict(shape, layers, acts, x) {
  acts[0].set(x);
  const L = layers.length;
  const nClasses = shape[shape.length - 1];
  for (let l = 0; l < L; l++) {
    const { W, b } = layers[l];
    const [aIn, aOut] = [acts[l], acts[l + 1]];
    const nIn = shape[l];
    for (let o = 0; o < shape[l + 1]; o++) {
      let sum = b[o];
      const row = o * nIn;
      for (let i = 0; i < nIn; i++) sum += W[row + i] * aIn[i];
      aOut[o] = l < L - 1 ? Math.max(0, sum) : sum;
    }
  }
  const logits = acts[L];
  let best = 0;
  for (let o = 1; o < nClasses; o++) if (logits[o] > logits[best]) best = o;
  return best;
}

// ---------------------------------------------------------------- training

function trainVariant(id, { shape, minAcc, label }, data) {
  const { train, trainLabels, test, testLabels } = data;
  console.log(`\n=== ${id} [${shape.join("→")}] ===`);

  const rand = mulberry32(0x5eed);
  const layers = initNet(shape, rand);
  const acts = shape.map((n) => new Float32Array(n));

  const EPOCHS = 20;
  const BATCH = 64;
  const order = Uint32Array.from({ length: train.n }, (_, i) => i);
  const x = new Float32Array(784);

  const testAccuracy = () => {
    let correct = 0;
    for (let s = 0; s < test.n; s++) {
      for (let i = 0; i < 784; i++) x[i] = test.pixels[s * 784 + i] / 255;
      if (predict(shape, layers, acts, x) === testLabels[s]) correct++;
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
        trainCorrect += trainSample(shape, layers, grads, acts, x, trainLabels[s]);
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

  if (acc < minAcc) {
    throw new Error(
      `${id}: test accuracy ${(acc * 100).toFixed(2)}% is below the ` +
        `${(minAcc * 100).toFixed(0)}% bar — not exporting`
    );
  }

  // ---- export weights-<id>.json (classic also keeps the original filename)
  const b64 = (arr) => Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString("base64");
  mkdirSync(ASSETS, { recursive: true });
  const weightsJson = {
    id,
    label,
    shape,
    testAccuracy: Number(acc.toFixed(4)),
    trainedAt: new Date().toISOString().slice(0, 10),
    layers: layers.map(({ W, b }) => ({ W: b64(W), b: b64(b) })),
  };
  const json = JSON.stringify(weightsJson);
  writeFileSync(join(ASSETS, `weights-${id}.json`), json);
  console.log(`wrote src/assets/weights-${id}.json (test accuracy ${(acc * 100).toFixed(2)}%)`);
  if (id === "classic") {
    writeFileSync(join(ASSETS, "weights.json"), json);
    console.log("wrote src/assets/weights.json (same classic net)");
  }

  return { shape, layers, acts, x, testAccuracy: acc };
}

// ---- samples.json: per class, 5 confident-correct + 1 ambiguous-correct,
//      always picked by the CLASSIC net (the attract loop's shared deck)

function exportSamples(trained, data) {
  const { shape, layers, acts, x } = trained;
  const { test, testLabels } = data;
  const softmaxConfidence = (s) => {
    for (let i = 0; i < 784; i++) x[i] = test.pixels[s * 784 + i] / 255;
    const pred = predict(shape, layers, acts, x);
    const logits = acts[shape.length - 1];
    const nClasses = shape[shape.length - 1];
    let max = -Infinity;
    for (let o = 0; o < nClasses; o++) max = Math.max(max, logits[o]);
    let sum = 0;
    for (let o = 0; o < nClasses; o++) sum += Math.exp(logits[o] - max);
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

// -------------------------------------------------------------------- main

async function main() {
  const arg = process.argv[2] ?? "classic";
  const ids = arg === "all" ? Object.keys(VARIANTS) : [arg];
  for (const id of ids) {
    if (!VARIANTS[id]) {
      throw new Error(`unknown variant "${id}" — use ${Object.keys(VARIANTS).join("/")} or all`);
    }
  }

  const paths = {};
  for (const [key, name] of Object.entries(FILES)) paths[key] = await ensureFile(name);
  const data = {
    train: parseIdxImages(paths.trainImages),
    trainLabels: parseIdxLabels(paths.trainLabels),
    test: parseIdxImages(paths.testImages),
    testLabels: parseIdxLabels(paths.testLabels),
  };
  console.log(`train ${data.train.n}, test ${data.test.n}`);

  for (const id of ids) {
    const trained = trainVariant(id, VARIANTS[id], data);
    if (id === "classic") exportSamples(trained, data);
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
