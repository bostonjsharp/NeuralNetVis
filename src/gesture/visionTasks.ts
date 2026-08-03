import { FilesetResolver, GestureRecognizer, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { GestureState, HandDebugFrame } from "./framePipeline";

/**
 * MediaPipe task construction shared by the vision worker and the
 * main-thread fallback path — one place for the tuned recognizer options,
 * so the two paths cannot drift apart.
 */

/** Absolute URLs — a worker resolves relative paths against its own script
 *  URL, not the page, so the main thread resolves these before handoff. */
export interface VisionAssets {
  wasmBase: string;
  gestureModelUrl: string;
  poseModelUrl: string;
}

/** Per-tier inference strides. Close tier: hands per-frame (a stroking
 *  fist is fast and blur-sensitive), pose at ~10Hz — it only feeds the ✕
 *  there, which must be HELD 800ms. Far tier (or nobody in frame): the
 *  arrangement inverts — pose feeds the cursor so it runs per-frame, and
 *  the hand model's only job is noticing someone stepping close. */
export const STRIDES = {
  close: { hand: 1, pose: 6 },
  far: { hand: 4, pose: 1 },
} as const;

export interface VisionTasks {
  recognizer: GestureRecognizer;
  poseLandmarker: PoseLandmarker | null;
}

export async function createVisionTasks(
  assets: VisionAssets,
  delegate: "GPU" | "CPU"
): Promise<VisionTasks> {
  const fileset = await FilesetResolver.forVisionTasks(assets.wasmBase);
  // GestureRecognizer = HandLandmarker + a *trained* fist/palm classifier.
  // Hand-rolled finger-curl ratios could not survive a knit sleeve at 2m;
  // this ships the same 21 landmarks plus a confidence per canned gesture.
  const recognizer = await GestureRecognizer.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: assets.gestureModelUrl, delegate },
    runningMode: "VIDEO",
    numHands: 2, // second hand only matters for the arms-crossed clear
    // Hands are small in frame at 2m — the defaults (0.5) drop them
    minHandDetectionConfidence: 0.35,
    minHandPresenceConfidence: 0.35,
    minTrackingConfidence: 0.35,
    // Report low-confidence guesses too: PenLatch consumes the score, and a
    // weak-but-real fist should still push the pen down. Thresholding here
    // would throw away exactly the gradient the latch is built to integrate.
    cannedGesturesClassifierOptions: { scoreThreshold: 0.1 },
  });
  // Body pose reads at far greater range than hands; it carries the ✕
  // detection (forearm segments crossing). Optional — hands-only if the
  // model asset is missing.
  let poseLandmarker: PoseLandmarker | null = null;
  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: assets.poseModelUrl, delegate },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  } catch (err) {
    console.warn("pose model unavailable — ✕ falls back to two-hand wrist distance:", err);
  }
  return { recognizer, poseLandmarker };
}

// ---- worker protocol ----

export interface VisionWorkerInit extends VisionAssets {
  type: "init";
}

export interface VisionWorkerFrame {
  type: "frame";
  bitmap: ImageBitmap;
  /** Capture timestamp (main-thread performance.now(), monotonic). */
  t: number;
  wantDebug: boolean;
}

export type VisionWorkerRequest = VisionWorkerInit | VisionWorkerFrame;

export type VisionWorkerResponse =
  | { type: "ready" }
  | {
      type: "state";
      state: GestureState | null;
      debug: HandDebugFrame | null;
      /** Inference costs measured inside the worker, for the perf overlay.
       *  Absent when the hand model was strided out this frame. */
      gestureMs?: number;
      poseMs?: number;
    }
  | { type: "error"; message: string };
