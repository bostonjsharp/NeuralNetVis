import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { classifyHand, mapToPad, PoseStabilizer, type HandPose } from "./handGesture";

export interface GestureState {
  present: boolean;
  /** Normalized pad-space position (0..1, already mirrored + region-mapped). */
  x: number;
  y: number;
  /** Debounced pose: ✊ fist = pen down, ✋ open = pen lifted, ✌️ two = clear. */
  pose: Exclude<HandPose, "unknown">;
}

/** Exponential smoothing factor for the cursor (higher = snappier). */
const SMOOTHING = 0.35;
/** Palm anchor: middle-finger MCP tracks the hand's center steadily. */
const PALM = 9;

/**
 * Opens the webcam, runs MediaPipe HandLandmarker (assets served locally
 * from public/), and reports a smoothed, debounced gesture state every
 * frame. Throws if there is no camera or permission is denied — callers
 * treat gestures as an optional enhancement and fall back to mouse.
 */
export async function startHandTracking(
  onState: (state: GestureState) => void
): Promise<() => void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 360, facingMode: "user" },
  });
  const video = document.createElement("video");
  video.srcObject = stream;
  video.playsInline = true;
  video.muted = true;
  await video.play();

  const fileset = await FilesetResolver.forVisionTasks("mediapipe-wasm");
  const landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "models/hand_landmarker.task", delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 1,
  });

  const stabilizer = new PoseStabilizer();
  let smoothX = 0.5;
  let smoothY = 0.5;
  let hadHand = false;
  let raf = 0;
  let lastVideoTime = -1;
  let disposed = false;

  const loop = () => {
    if (disposed) return;
    raf = requestAnimationFrame(loop);
    if (video.currentTime === lastVideoTime) return; // no new camera frame yet
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    const landmarks = result.landmarks?.[0];
    if (!landmarks) {
      if (hadHand) {
        hadHand = false;
        stabilizer.reset();
        onState({ present: false, x: smoothX, y: smoothY, pose: "open" });
      }
      return;
    }
    const mapped = mapToPad(landmarks[PALM].x, landmarks[PALM].y);
    if (!hadHand) {
      // Snap on reacquire so the cursor doesn't glide in from its old spot
      smoothX = mapped.x;
      smoothY = mapped.y;
      hadHand = true;
    } else {
      smoothX += (mapped.x - smoothX) * SMOOTHING;
      smoothY += (mapped.y - smoothY) * SMOOTHING;
    }
    const pose = stabilizer.update(classifyHand(landmarks)) as Exclude<HandPose, "unknown">;
    onState({ present: true, x: smoothX, y: smoothY, pose });
  };
  loop();

  return () => {
    disposed = true;
    cancelAnimationFrame(raf);
    landmarker.close();
    for (const track of stream.getTracks()) track.stop();
  };
}
