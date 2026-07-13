import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import {
  classifyHand,
  isCloseEnough,
  isRaised,
  mapToPad,
  PALM,
  palmsClose,
  PoseStabilizer,
  type HandPose,
} from "./handGesture";

export interface GestureState {
  present: boolean;
  /** Normalized pad-space position (0..1, already mirrored + region-mapped). */
  x: number;
  y: number;
  /** Debounced pose of the primary hand: ✊ fist = pen down, ✋ open = pen lifted. */
  pose: Exclude<HandPose, "unknown">;
  /** Palm held high in the camera frame (the wake gesture). */
  raised: boolean;
  /** Both hands up with palms together/crossed — the ✕ clear gesture. */
  crossed: boolean;
}

/** Exponential smoothing factor for the cursor (higher = snappier). */
const SMOOTHING = 0.35;

/** Consecutive hand-less camera frames tolerated before reporting absence.
 *  Tracking drops out for a frame or two constantly at 2m — without this
 *  grace a 5-second wake hold would reset on every blip. */
const ABSENCE_GRACE_FRAMES = 14;

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
    numHands: 2, // second hand only matters for the arms-crossed clear
  });

  const stabilizer = new PoseStabilizer();
  let smoothX = 0.5;
  let smoothY = 0.5;
  let hadHand = false;
  let missedFrames = 0;
  let raf = 0;
  let lastVideoTime = -1;
  let disposed = false;

  const loop = () => {
    if (disposed) return;
    raf = requestAnimationFrame(loop);
    if (video.currentTime === lastVideoTime) return; // no new camera frame yet
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    // Hands too small in frame are far beyond the visitor zone — ignore them
    const hands = (result.landmarks ?? []).filter(isCloseEnough);
    if (hands.length === 0) {
      if (hadHand && ++missedFrames > ABSENCE_GRACE_FRAMES) {
        hadHand = false;
        stabilizer.reset();
        onState({
          present: false,
          x: smoothX,
          y: smoothY,
          pose: "open",
          raised: false,
          crossed: false,
        });
      }
      return;
    }
    missedFrames = 0;
    const primary = hands[0];
    const mapped = mapToPad(primary[PALM].x, primary[PALM].y);
    if (!hadHand) {
      // Snap on reacquire so the cursor doesn't glide in from its old spot
      smoothX = mapped.x;
      smoothY = mapped.y;
      hadHand = true;
    } else {
      smoothX += (mapped.x - smoothX) * SMOOTHING;
      smoothY += (mapped.y - smoothY) * SMOOTHING;
    }
    const pose = stabilizer.update(classifyHand(primary)) as Exclude<HandPose, "unknown">;
    const crossed = hands.length >= 2 && palmsClose(hands[0][PALM], hands[1][PALM]);
    onState({ present: true, x: smoothX, y: smoothY, pose, raised: isRaised(primary), crossed });
  };
  loop();

  return () => {
    disposed = true;
    cancelAnimationFrame(raf);
    landmarker.close();
    for (const track of stream.getTracks()) track.stop();
  };
}
