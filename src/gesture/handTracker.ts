import { FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";
import {
  classifyHand,
  forearmsCrossed,
  handSpan,
  isCloseEnough,
  isRaised,
  PadMapper,
  PALM,
  pickPrimaryHand,
  POSE,
  PoseStabilizer,
  RAISE_LINE,
  WRIST,
  wristsClose,
  type HandPose,
  type Landmark,
  type PoseLandmark,
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
const ABSENCE_GRACE_FRAMES = 20;

/** Per-frame diagnostics for the dev overlay (G key / ?debug). */
export interface HandDebugFrame {
  hands: {
    landmarks: Landmark[];
    span: number;
    closeEnough: boolean;
    rawPose: HandPose;
  }[];
  /** Which entry in `hands` is the pen hand (-1 when none). */
  primaryIndex: number;
  /** Forearm segments from the body-pose model, when a body is visible. */
  forearms: {
    leftElbow: Landmark;
    leftWrist: Landmark;
    rightElbow: Landmark;
    rightWrist: Landmark;
    crossed: boolean;
  } | null;
  /** True while the pen is being steered by the body-pose wrist because
   *  hand detection lost the fist. */
  penFallback: boolean;
  crossed: boolean;
  /** Last state actually emitted to the app. */
  state: GestureState | null;
}

export interface HandDebugSink {
  /** Called once with the (playing) camera video element. */
  attachVideo(video: HTMLVideoElement): void;
  onFrame(frame: HandDebugFrame): void;
}

/**
 * Opens the webcam, runs MediaPipe HandLandmarker (assets served locally
 * from public/), and reports a smoothed, debounced gesture state every
 * frame. Throws if there is no camera or permission is denied — callers
 * treat gestures as an optional enhancement and fall back to mouse.
 */
export async function startHandTracking(
  onState: (state: GestureState) => void,
  debug?: HandDebugSink
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
    // Hands are small in frame at 2m — the defaults (0.5) drop them
    minHandDetectionConfidence: 0.35,
    minHandPresenceConfidence: 0.35,
    minTrackingConfidence: 0.35,
  });
  // Body pose reads at far greater range than hands; it carries the ✕
  // detection (forearm segments crossing). Optional — hands-only if the
  // model asset is missing.
  let poseLandmarker: PoseLandmarker | null = null;
  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "models/pose_landmarker_lite.task", delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  } catch (err) {
    console.warn("pose model unavailable — ✕ falls back to two-hand wrist distance:", err);
  }
  debug?.attachVideo(video);

  const stabilizer = new PoseStabilizer();
  const mapper = new PadMapper();
  let smoothX = 0.5;
  let smoothY = 0.5;
  let smoothSpan = 0;
  let hadHand = false;
  let missedFrames = 0;
  let lastPrimaryPalm: Landmark | null = null;
  // Pose-wrist fallback: hand detection routinely loses a clenched fist at
  // 2m, but the body model keeps tracking the wrist. While both are
  // visible we learn which body wrist is the pen and the palm's offset
  // from it; on hand loss the pen keeps moving from the pose wrist.
  let penSide: "left" | "right" | null = null;
  const palmOffset = { x: 0, y: 0 };
  let fallbackFrames = 0;
  const FALLBACK_MAX_FRAMES = 90; // ~3s of pose-steered pen before giving up
  let raf = 0;
  let lastVideoTime = -1;
  let disposed = false;

  let lastEmitted: GestureState | null = null;
  const emit = (state: GestureState) => {
    lastEmitted = state;
    onState(state);
  };

  const loop = () => {
    if (disposed) return;
    raf = requestAnimationFrame(loop);
    if (video.currentTime === lastVideoTime) return; // no new camera frame yet
    lastVideoTime = video.currentTime;
    const now = performance.now();
    const result = landmarker.detectForVideo(video, now);
    const allHands = result.landmarks ?? [];
    // Hands too small in frame are far beyond the visitor zone — ignore them
    const hands = allHands.filter(isCloseEnough);

    // ✕ detection: forearm segments crossing (body pose, long range),
    // with two-hands-wrists-together as the fallback signal
    let bodyPose: PoseLandmark[] | undefined;
    if (poseLandmarker) {
      bodyPose = poseLandmarker.detectForVideo(video, now).landmarks?.[0];
    }
    const poseCrossed = bodyPose ? forearmsCrossed(bodyPose) : false;
    const crossed =
      poseCrossed || (hands.length >= 2 && wristsClose(hands[0][WRIST], hands[1][WRIST]));

    const primaryIndex =
      hands.length > 0 ? pickPrimaryHand(hands.map((h) => h[PALM]), lastPrimaryPalm) : -1;

    const penWrist =
      bodyPose && penSide
        ? bodyPose[penSide === "left" ? POSE.leftWrist : POSE.rightWrist]
        : undefined;
    const canFallback =
      hands.length === 0 &&
      hadHand &&
      !!penWrist &&
      (penWrist.visibility ?? 1) > 0.5 &&
      fallbackFrames < FALLBACK_MAX_FRAMES;

    if (debug) {
      const primaryLandmarks = primaryIndex >= 0 ? hands[primaryIndex] : null;
      debug.onFrame({
        hands: allHands.map((landmarks) => ({
          landmarks,
          span: handSpan(landmarks),
          closeEnough: isCloseEnough(landmarks),
          rawPose: classifyHand(landmarks),
        })),
        primaryIndex: primaryLandmarks ? allHands.indexOf(primaryLandmarks) : -1,
        forearms: bodyPose
          ? {
              leftElbow: bodyPose[POSE.leftElbow],
              leftWrist: bodyPose[POSE.leftWrist],
              rightElbow: bodyPose[POSE.rightElbow],
              rightWrist: bodyPose[POSE.rightWrist],
              crossed: poseCrossed,
            }
          : null,
        penFallback: canFallback,
        crossed,
        state: lastEmitted,
      });
    }
    if (hands.length === 0) {
      if (canFallback && penWrist) {
        // Steer the pen from the body wrist; keep the current pose (a
        // lost hand mid-draw is almost always still a fist — open hands
        // redetect immediately).
        fallbackFrames++;
        missedFrames = 0;
        const estimate = { x: penWrist.x + palmOffset.x, y: penWrist.y + palmOffset.y };
        const mapped = mapper.update(estimate, smoothSpan || 0.04);
        smoothX += (mapped.x - smoothX) * SMOOTHING;
        smoothY += (mapped.y - smoothY) * SMOOTHING;
        lastPrimaryPalm = estimate;
        emit({
          present: true,
          x: smoothX,
          y: smoothY,
          pose: lastEmitted?.present ? lastEmitted.pose : "open",
          raised: estimate.y < RAISE_LINE,
          crossed,
        });
        return;
      }
      if (hadHand && ++missedFrames > ABSENCE_GRACE_FRAMES) {
        hadHand = false;
        stabilizer.reset();
        mapper.reset();
        lastPrimaryPalm = null;
        penSide = null;
        emit({
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
    fallbackFrames = 0;
    const primary = hands[primaryIndex];
    lastPrimaryPalm = { x: primary[PALM].x, y: primary[PALM].y };
    // Learn which body wrist is the pen and the palm's offset from it
    if (bodyPose) {
      const palm = primary[PALM];
      const leftWrist = bodyPose[POSE.leftWrist];
      const rightWrist = bodyPose[POSE.rightWrist];
      const dLeft = Math.hypot(leftWrist.x - palm.x, leftWrist.y - palm.y);
      const dRight = Math.hypot(rightWrist.x - palm.x, rightWrist.y - palm.y);
      penSide = dLeft < dRight ? "left" : "right";
      const wristLm = dLeft < dRight ? leftWrist : rightWrist;
      palmOffset.x += (palm.x - wristLm.x - palmOffset.x) * 0.2;
      palmOffset.y += (palm.y - wristLm.y - palmOffset.y) * 0.2;
    }
    // Smooth the span so a single misread frame doesn't warp the mapper box
    const span = handSpan(primary);
    smoothSpan = smoothSpan === 0 ? span : smoothSpan + (span - smoothSpan) * 0.1;
    const mapped = mapper.update(primary[PALM], smoothSpan);
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
    emit({ present: true, x: smoothX, y: smoothY, pose, raised: isRaised(primary), crossed });
  };
  loop();

  return () => {
    disposed = true;
    cancelAnimationFrame(raf);
    landmarker.close();
    poseLandmarker?.close();
    for (const track of stream.getTracks()) track.stop();
  };
}
