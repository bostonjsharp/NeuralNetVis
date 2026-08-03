import { FramePipeline, type DrivingTier, type PipelineFrameInput } from "./framePipeline";
import {
  createVisionTasks,
  STRIDES,
  type VisionTasks,
  type VisionWorkerRequest,
  type VisionWorkerResponse,
} from "./visionTasks";

/**
 * Dedicated worker that runs MediaPipe inference off the main thread, so
 * WASM never steals frame budget from the Three.js render loop. The main
 * thread pumps ImageBitmap camera frames in (one in flight at a time);
 * this folds them through the shared FramePipeline and posts GestureState
 * back. Any throw is reported and the facade falls back to the inline
 * main-thread path — the kiosk must never lose gestures to a worker bug.
 */

const post = (message: VisionWorkerResponse) =>
  (self as unknown as { postMessage(m: unknown): void }).postMessage(message);

// MediaPipe's WASM loader is a classic script (`var ModuleFactory = ...`):
// its global only exists if the script runs via importScripts. Vite serves
// workers as ES modules, where the native importScripts throws and
// MediaPipe's dynamic-import fallback leaves the factory module-scoped —
// "ModuleFactory not set." Restore classic semantics: fetch synchronously
// and evaluate in global scope (indirect eval), so top-level `var` lands
// on globalThis exactly as a classic worker would run it.
(self as unknown as { importScripts(...urls: (string | URL)[]): void }).importScripts = (
  ...urls: (string | URL)[]
) => {
  for (const url of urls) {
    const xhr = new XMLHttpRequest();
    xhr.open("GET", url.toString(), false);
    xhr.send();
    if (xhr.status !== 200) {
      throw new Error(`importScripts polyfill: ${xhr.status} for ${url}`);
    }
    (0, eval)(xhr.responseText);
  }
};

let tasks: VisionTasks | null = null;
let pipeline: FramePipeline | null = null;
let tier: DrivingTier = null;
let handCountdown = 0;
let poseCountdown = 0;

self.onmessage = async (e: MessageEvent<VisionWorkerRequest>) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      pipeline = new FramePipeline();
      try {
        tasks = await createVisionTasks(msg, "GPU");
      } catch (gpuErr) {
        // No usable GPU context in this worker (kiosk driver quirks) —
        // CPU inference off-main-thread still beats GPU inference on it.
        console.warn("vision worker GPU delegate failed — retrying on CPU:", gpuErr);
        tasks = await createVisionTasks(msg, "CPU");
      }
      post({ type: "ready" });
      return;
    }
    if (msg.type === "frame") {
      const { bitmap, t, wantDebug } = msg;
      if (!tasks || !pipeline) {
        bitmap.close();
        return;
      }
      const strides = tier === "close" ? STRIDES.close : STRIDES.far;
      let allHands: PipelineFrameInput["allHands"] = [];
      let allGestures: PipelineFrameInput["allGestures"] = [];
      let gestureMs: number | undefined;
      if (--handCountdown <= 0) {
        handCountdown = strides.hand;
        const gestureStart = performance.now();
        const result = tasks.recognizer.recognizeForVideo(bitmap, t);
        gestureMs = performance.now() - gestureStart;
        allHands = result.landmarks ?? [];
        allGestures = result.gestures ?? [];
      }
      let poseRan = false;
      let bodyPose;
      let poseMs: number | undefined;
      if (tasks.poseLandmarker && --poseCountdown <= 0) {
        poseCountdown = strides.pose;
        poseRan = true;
        const poseStart = performance.now();
        bodyPose = tasks.poseLandmarker.detectForVideo(bitmap, t).landmarks?.[0];
        poseMs = performance.now() - poseStart;
      }
      bitmap.close();
      const out = pipeline.update({
        allHands,
        allGestures,
        poseRan,
        bodyPose,
        nowMs: t,
        wantDebug,
      });
      tier = out.tier;
      post({ type: "state", state: out.state, debug: out.debug, gestureMs, poseMs });
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
