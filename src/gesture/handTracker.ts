import { FilesetResolver, GestureRecognizer, PoseLandmarker } from "@mediapipe/tasks-vision";
import {
  forearmsCrossed,
  handSpan,
  isCloseEnough,
  isRaised,
  NO_EVIDENCE,
  OneEuroFilter,
  PadMapper,
  PALM,
  penEvidence,
  PenLatch,
  pickPrimaryHand,
  POSE,
  WRIST,
  wristsClose,
  type GestureCategory,
  type Landmark,
  type PenPose,
  type PoseLandmark,
} from "./handGesture";

export interface GestureState {
  present: boolean;
  /** Normalized pad-space position (0..1, already mirrored + region-mapped). */
  x: number;
  y: number;
  /** Latched pose of the primary hand: ✊ fist = pen down, ✋ open = pen lifted. */
  pose: PenPose;
  /** Palm held high in the camera frame (the wake gesture). */
  raised: boolean;
  /** Both hands up with palms together/crossed — the ✕ clear gesture. */
  crossed: boolean;
}

/** Fallback timestep when the camera clock gives us nothing useful. */
const NOMINAL_DT = 1 / 30;

/** Consecutive hand-less camera frames tolerated before reporting absence.
 *  Tracking drops out for a frame or two constantly at 2m — without this
 *  grace a 5-second wake hold would reset on every blip. */
const ABSENCE_GRACE_FRAMES = 20;

/** Body pose runs every Nth camera frame (~10Hz at a 60fps camera). It only
 *  feeds the ✕ clear, which must be HELD for 800ms — per-frame detection
 *  buys nothing there, and at 60fps it doubles the GPU inference load
 *  contending with the bloom-heavy render loop. The held result is at most
 *  ~83ms stale, an order of magnitude inside the hold window. The hand
 *  model stays per-frame: a stroking fist is fast and blur-sensitive. */
const POSE_FRAME_STRIDE = 6;

/** Per-frame diagnostics for the dev overlay (G key / ?debug). */
export interface HandDebugFrame {
  hands: {
    landmarks: Landmark[];
    span: number;
    closeEnough: boolean;
    /** Top canned gesture the recognizer named for this hand. */
    gesture: string;
    score: number;
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
  /** Pen latch charge (0..1) — how close the pen is to pressing/lifting. */
  charge: number;
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
 * Opens the webcam, runs MediaPipe GestureRecognizer (assets served locally
 * from public/), and reports a smoothed, latched gesture state every frame.
 * Throws if there is no camera or permission is denied — callers treat
 * gestures as an optional enhancement and fall back to mouse.
 */
export async function startHandTracking(
  onState: (state: GestureState) => void,
  debug?: HandDebugSink
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

  const fileset = await FilesetResolver.forVisionTasks("mediapipe-wasm");
  // GestureRecognizer = HandLandmarker + a *trained* fist/palm classifier.
  // Hand-rolled finger-curl ratios could not survive a knit sleeve at 2m;
  // this ships the same 21 landmarks plus a confidence per canned gesture.
  const recognizer = await GestureRecognizer.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "models/gesture_recognizer.task", delegate: "GPU" },
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
      baseOptions: { modelAssetPath: "models/pose_landmarker_lite.task", delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  } catch (err) {
    console.warn("pose model unavailable — ✕ falls back to two-hand wrist distance:", err);
  }
  debug?.attachVideo(video);

  const latch = new PenLatch();
  const mapper = new PadMapper();
  // Speed-adaptive smoothing: steady when the hand hovers, near-transparent
  // when it strokes. A fixed factor could only ever be one or the other.
  const filterX = new OneEuroFilter();
  const filterY = new OneEuroFilter();
  let lastFrameTime = 0;
  let smoothX = 0.5;
  let smoothY = 0.5;
  let smoothSpan = 0;
  let hadHand = false;
  let missedFrames = 0;
  let lastPrimaryPalm: Landmark | null = null;
  let poseFrame = 0;
  let lastBodyPose: PoseLandmark[] | undefined;
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
    const result = recognizer.recognizeForVideo(video, now);
    const allHands = result.landmarks ?? [];
    const allGestures: GestureCategory[][] = result.gestures ?? [];
    // Hands too small in frame are far beyond the visitor zone — ignore them.
    // Keep each hand's gesture scores paired with it through the filter.
    const hands = allHands
      .map((landmarks, i) => ({ landmarks, gestures: allGestures[i] ?? [] }))
      .filter((h) => isCloseEnough(h.landmarks));

    // ✕ detection: forearm segments crossing (body pose, long range),
    // with two-hands-wrists-together as the fallback signal. This is the
    // ONLY thing the body-pose model drives — it is far too noisy through a
    // sleeve to be trusted with the pen.
    if (poseLandmarker && poseFrame++ % POSE_FRAME_STRIDE === 0) {
      lastBodyPose = poseLandmarker.detectForVideo(video, now).landmarks?.[0];
    }
    const bodyPose = lastBodyPose;
    const poseCrossed = bodyPose ? forearmsCrossed(bodyPose) : false;
    const crossed =
      poseCrossed ||
      (hands.length >= 2 && wristsClose(hands[0].landmarks[WRIST], hands[1].landmarks[WRIST]));

    const primaryIndex =
      hands.length > 0
        ? pickPrimaryHand(
            hands.map((h) => h.landmarks[PALM]),
            lastPrimaryPalm
          )
        : -1;

    if (debug) {
      const primaryHand = primaryIndex >= 0 ? hands[primaryIndex] : null;
      debug.onFrame({
        hands: allHands.map((landmarks, i) => {
          const top = (allGestures[i] ?? [])[0];
          return {
            landmarks,
            span: handSpan(landmarks),
            closeEnough: isCloseEnough(landmarks),
            gesture: top?.categoryName ?? "None",
            score: top?.score ?? 0,
          };
        }),
        primaryIndex: primaryHand ? allHands.indexOf(primaryHand.landmarks) : -1,
        forearms: bodyPose
          ? {
              leftElbow: bodyPose[POSE.leftElbow],
              leftWrist: bodyPose[POSE.leftWrist],
              rightElbow: bodyPose[POSE.rightElbow],
              rightWrist: bodyPose[POSE.rightWrist],
              crossed: poseCrossed,
            }
          : null,
        charge: latch.level(),
        crossed,
        state: lastEmitted,
      });
    }

    if (hands.length === 0) {
      if (!hadHand) return;
      // No hand is not a fist. Bleed the latch down instead of freezing the
      // last pose: a brief dropout rides through, a real absence lifts the
      // pen well before the cursor itself is withdrawn.
      const pose = latch.update(NO_EVIDENCE);
      if (++missedFrames > ABSENCE_GRACE_FRAMES) {
        hadHand = false;
        latch.reset();
        mapper.reset();
        lastFrameTime = 0;
        lastPrimaryPalm = null;
        emit({
          present: false,
          x: smoothX,
          y: smoothY,
          pose: "open",
          raised: false,
          crossed: false,
        });
        return;
      }
      // Hold the cursor where it was — never invent a position from a model
      // that cannot see through a sleeve.
      emit({ present: true, x: smoothX, y: smoothY, pose, raised: false, crossed });
      return;
    }

    missedFrames = 0;
    const primary = hands[primaryIndex];
    const palm = primary.landmarks[PALM];
    lastPrimaryPalm = { x: palm.x, y: palm.y };
    // Smooth the span so a single misread frame doesn't warp the mapper box
    const span = handSpan(primary.landmarks);
    smoothSpan = smoothSpan === 0 ? span : smoothSpan + (span - smoothSpan) * 0.1;
    const mapped = mapper.update(palm, smoothSpan);
    if (!hadHand) {
      // Snap on reacquire so the cursor doesn't glide in from its old spot
      filterX.reset();
      filterY.reset();
      hadHand = true;
    }
    // Real elapsed time, not an assumed 30fps — the camera's rate varies with
    // light, and a filter fed the wrong dt mis-scales its own speed estimate.
    const dt = lastFrameTime ? (now - lastFrameTime) / 1000 : NOMINAL_DT;
    lastFrameTime = now;
    smoothX = filterX.filter(mapped.x, dt);
    smoothY = filterY.filter(mapped.y, dt);
    const pose = latch.update(penEvidence(primary.gestures));
    emit({
      present: true,
      x: smoothX,
      y: smoothY,
      pose,
      raised: isRaised(primary.landmarks),
      crossed,
    });
  };
  loop();

  return () => {
    disposed = true;
    cancelAnimationFrame(raf);
    recognizer.close();
    poseLandmarker?.close();
    for (const track of stream.getTracks()) track.stop();
  };
}
