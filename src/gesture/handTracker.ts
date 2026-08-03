import { perf } from "../app/perf";
import {
  FramePipeline,
  type DrivingTier,
  type GestureState,
  type HandDebugFrame,
  type PipelineFrameInput,
} from "./framePipeline";
import {
  createVisionTasks,
  STRIDES,
  type VisionAssets,
  type VisionWorkerResponse,
} from "./visionTasks";

export type { GestureState, HandDebugFrame } from "./framePipeline";

export interface HandDebugSink {
  /** Called once with the (playing) camera video element. */
  attachVideo(video: HTMLVideoElement): void;
  /** Polled per frame — building a HandDebugFrame allocates, so the tracker
   *  skips it entirely while the overlay is closed. */
  wantFrames(): boolean;
  onFrame(frame: HandDebugFrame): void;
}

export interface HandTrackingOptions {
  /** Polled per camera frame: minimum ms between inference captures, 0 for
   *  every frame. Lets the app halve inference load in attract mode (a 5s
   *  wake hold doesn't need 60Hz) while strokes keep the full rate. */
  captureIntervalMs?: () => number;
}

/**
 * Opens the webcam and reports a smoothed, latched gesture state every
 * frame. MediaPipe inference runs in a dedicated worker so it can never
 * stall the render loop; if the worker path fails for any reason (at
 * startup or mid-run) it falls back to inline main-thread inference with
 * the identical FramePipeline. Throws if there is no camera or permission
 * is denied — callers treat gestures as an optional enhancement and fall
 * back to mouse.
 */
export async function startHandTracking(
  onState: (state: GestureState) => void,
  debug?: HandDebugSink,
  options?: HandTrackingOptions
): Promise<() => void> {
  // Ask for 60fps: a fast hand is only blurry because the exposure is long,
  // and a higher frame rate forces the sensor to shorten it. Blur is what
  // collapses Closed_Fist confidence mid-stroke (which drains the pen latch),
  // so this buys both a sharper fist and twice the cursor samples. `ideal`
  // keeps a 30fps-only webcam working rather than failing the constraint.
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 360 },
      frameRate: { ideal: 60, min: 24 },
      facingMode: "user",
    },
  });
  const video = document.createElement("video");
  video.srcObject = stream;
  video.playsInline = true;
  video.muted = true;
  await video.play();
  debug?.attachVideo(video);

  // The worker resolves nothing relative to the page — hand it final URLs.
  const assets: VisionAssets = {
    wasmBase: new URL("mediapipe-wasm", document.baseURI).href,
    gestureModelUrl: new URL("models/gesture_recognizer.task", document.baseURI).href,
    poseModelUrl: new URL("models/pose_landmarker_lite.task", document.baseURI).href,
  };

  let stopped = false;
  let stopPath: (() => void) | null = null;
  const stopCamera = () => {
    for (const track of stream.getTracks()) track.stop();
  };

  const runInline = () =>
    startInlinePath(video, assets, onState, debug, options).then((stop) => {
      if (stopped) stop();
      else stopPath = stop;
    });

  try {
    stopPath = await startWorkerPath(video, assets, onState, debug, options, () => {
      // Fatal worker error mid-run: the camera is still live — swap the
      // inference engine underneath it without dropping the session.
      if (stopped) return;
      runInline().catch((err: unknown) => {
        console.warn("inline fallback after worker failure also failed:", err);
      });
    });
  } catch (workerErr) {
    console.warn("vision worker unavailable — hand tracking on the main thread:", workerErr);
    try {
      await runInline();
    } catch (inlineErr) {
      stopCamera();
      throw inlineErr;
    }
  }

  return () => {
    stopped = true;
    stopPath?.();
    stopCamera();
  };
}

/**
 * Worker path: pump ImageBitmap frames to the vision worker, one in flight
 * at a time — a busy worker simply drops camera frames, so backpressure
 * can never queue up latency.
 */
function startWorkerPath(
  video: HTMLVideoElement,
  assets: VisionAssets,
  onState: (state: GestureState) => void,
  debug: HandDebugSink | undefined,
  options: HandTrackingOptions | undefined,
  onFatal: () => void
): Promise<() => void> {
  if (typeof Worker === "undefined" || typeof createImageBitmap !== "function") {
    return Promise.reject(new Error("Worker/createImageBitmap unsupported"));
  }
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./visionWorker.ts", import.meta.url), {
      type: "module",
    });
    let ready = false;
    let disposed = false;
    let busy = false;
    let cancelPump = () => {};

    const stop = () => {
      disposed = true;
      cancelPump();
      worker.terminate();
    };

    const readyTimer = window.setTimeout(
      () => fatal(new Error("vision worker never became ready (10s)")),
      10_000
    );

    const fatal = (err: Error) => {
      if (disposed) return;
      window.clearTimeout(readyTimer);
      stop();
      if (ready) {
        console.warn("vision worker failed mid-run:", err);
        onFatal();
      } else {
        reject(err);
      }
    };

    worker.onerror = (e) => fatal(new Error(e.message || "vision worker crashed"));
    worker.onmessage = (e: MessageEvent<VisionWorkerResponse>) => {
      const msg = e.data;
      if (msg.type === "ready") {
        ready = true;
        window.clearTimeout(readyTimer);
        startPump();
        resolve(stop);
      } else if (msg.type === "state") {
        busy = false;
        if (msg.gestureMs !== undefined) perf.recordMark("gesture", msg.gestureMs);
        if (msg.poseMs !== undefined) perf.recordMark("pose", msg.poseMs);
        if (msg.state) onState(msg.state);
        if (msg.debug) debug?.onFrame(msg.debug);
      } else {
        fatal(new Error(msg.message));
      }
    };

    worker.postMessage({ type: "init", ...assets });

    let lastCapture = 0;
    const capture = () => {
      if (disposed || busy) return;
      const now = performance.now();
      if (now - lastCapture < (options?.captureIntervalMs?.() ?? 0)) return;
      lastCapture = now;
      busy = true;
      createImageBitmap(video).then(
        (bitmap) => {
          if (disposed) {
            bitmap.close();
            return;
          }
          worker.postMessage(
            { type: "frame", bitmap, t: performance.now(), wantDebug: debug?.wantFrames() ?? false },
            [bitmap]
          );
        },
        () => {
          busy = false; // a torn video frame — just wait for the next one
        }
      );
    };

    const startPump = () => {
      // lib.dom types rVFC as always-present; older engines may lack it
      const v = video as HTMLVideoElement & {
        requestVideoFrameCallback?: HTMLVideoElement["requestVideoFrameCallback"];
      };
      if (typeof v.requestVideoFrameCallback === "function") {
        let handle = 0;
        const onFrame = () => {
          if (disposed) return;
          capture();
          handle = video.requestVideoFrameCallback(onFrame);
        };
        handle = video.requestVideoFrameCallback(onFrame);
        cancelPump = () => video.cancelVideoFrameCallback(handle);
      } else {
        let raf = 0;
        let lastVideoTime = -1;
        const loop = () => {
          if (disposed) return;
          raf = requestAnimationFrame(loop);
          if (video.currentTime === lastVideoTime) return;
          lastVideoTime = video.currentTime;
          capture();
        };
        loop();
        cancelPump = () => cancelAnimationFrame(raf);
      }
    };
  });
}

/**
 * Inline fallback: today's pre-worker behavior — inference on the main
 * thread inside a rAF loop, same recognizer options, same FramePipeline.
 */
async function startInlinePath(
  video: HTMLVideoElement,
  assets: VisionAssets,
  onState: (state: GestureState) => void,
  debug?: HandDebugSink,
  options?: HandTrackingOptions
): Promise<() => void> {
  const { recognizer, poseLandmarker } = await createVisionTasks(assets, "GPU");
  const pipeline = new FramePipeline();
  let tier: DrivingTier = null;
  let handCountdown = 0;
  let poseCountdown = 0;
  let raf = 0;
  let lastVideoTime = -1;
  let lastCapture = 0;
  let disposed = false;

  const loop = () => {
    if (disposed) return;
    raf = requestAnimationFrame(loop);
    if (video.currentTime === lastVideoTime) return; // no new camera frame yet
    lastVideoTime = video.currentTime;
    const now = performance.now();
    if (now - lastCapture < (options?.captureIntervalMs?.() ?? 0)) return;
    lastCapture = now;
    const strides = tier === "close" ? STRIDES.close : STRIDES.far;
    let allHands: PipelineFrameInput["allHands"] = [];
    let allGestures: PipelineFrameInput["allGestures"] = [];
    if (--handCountdown <= 0) {
      handCountdown = strides.hand;
      const gestureStart = performance.now();
      const result = recognizer.recognizeForVideo(video, now);
      perf.recordMark("gesture", performance.now() - gestureStart);
      allHands = result.landmarks ?? [];
      allGestures = result.gestures ?? [];
    }
    let poseRan = false;
    let bodyPose;
    if (poseLandmarker && --poseCountdown <= 0) {
      poseCountdown = strides.pose;
      poseRan = true;
      const poseStart = performance.now();
      bodyPose = poseLandmarker.detectForVideo(video, now).landmarks?.[0];
      perf.recordMark("pose", performance.now() - poseStart);
    }
    const out = pipeline.update({
      allHands,
      allGestures,
      poseRan,
      bodyPose,
      nowMs: now,
      wantDebug: debug?.wantFrames() ?? false,
    });
    tier = out.tier;
    if (out.state) onState(out.state);
    if (out.debug) debug?.onFrame(out.debug);
  };
  loop();

  return () => {
    disposed = true;
    cancelAnimationFrame(raf);
    recognizer.close();
    poseLandmarker?.close();
  };
}
