// Stages the MediaPipe hand-tracking assets into public/ so the built app
// is fully offline: the WASM runtime is copied out of node_modules and the
// hand landmark model is downloaded once from Google's model zoo.
//
//   node scripts/fetch-gesture-assets.mjs

import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { get } from "node:https";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const WASM_SRC = join(ROOT, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const WASM_DEST = join(ROOT, "public", "mediapipe-wasm");
const MODELS = [
  {
    // Hand landmarks + a trained canned-gesture classifier (Closed_Fist /
    // Open_Palm with confidences). Supersedes hand_landmarker.task: curl
    // ratios computed by hand were not robust to sleeves at 2m.
    dest: join(ROOT, "public", "models", "gesture_recognizer.task"),
    url: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
  },
  {
    // Body landmarks read at far greater range than hands — used to see
    // the arms-crossed ✕ from across the room
    dest: join(ROOT, "public", "models", "pose_landmarker_lite.task"),
    url: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  },
];

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

mkdirSync(WASM_DEST, { recursive: true });
for (const name of readdirSync(WASM_SRC)) {
  copyFileSync(join(WASM_SRC, name), join(WASM_DEST, name));
  console.log(`copied ${name}`);
}

for (const { dest, url } of MODELS) {
  const name = dest.split(/[\\/]/).pop();
  if (existsSync(dest) && statSync(dest).size > 1_000_000) {
    console.log(`${name} already present`);
    continue;
  }
  mkdirSync(dirname(dest), { recursive: true });
  console.log(`downloading ${name} ...`);
  writeFileSync(dest, await fetchBuffer(url));
  console.log(`wrote ${dest}`);
}
