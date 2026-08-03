import {
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
  thumbEvidence,
  WRIST,
  wristsClose,
  type GestureCategory,
  type Landmark,
  type PenPose,
} from "./handGesture";
import {
  armOut,
  forearmsCrossed,
  pickActiveArm,
  POSE,
  POSE_PAD_REACH,
  poseRaised,
  reachEvidence,
  shoulderWidth,
  wristOf,
  type PoseLandmark,
  type Side,
} from "./poseGesture";

/**
 * The per-frame folding of raw MediaPipe readings into the app's gesture
 * state — every latch, filter, and grace rule in one pure class. It runs
 * identically inside the vision worker and on the main-thread fallback
 * path, so the tuned 2m-camera behavior can't drift between the two.
 */

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
  /** Latched 👍 on the primary hand — the brain-switch verb. */
  thumbsUp: boolean;
}

/** Which signal source produced the frame's state. "close" includes the
 *  close tier's absence-grace window — the far tier may only engage once
 *  the close tier has fully released. */
export type DrivingTier = "close" | "far" | null;

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
  tier: DrivingTier;
  /** Held body pose, for skeleton drawing. */
  bodyPose: PoseLandmark[] | null;
  /** Far-tier reach latch readout (the on-site tuning instrument). */
  farReach: { side: Side; charge: number } | null;
  /** Last state actually emitted to the app. */
  state: GestureState | null;
}

/** Fallback timestep when the camera clock gives us nothing useful. */
const NOMINAL_DT = 1 / 30;

/** Consecutive hand-less camera frames tolerated before reporting absence.
 *  Tracking drops out for a frame or two constantly at 2m — without this
 *  grace a 5-second wake hold would reset on every blip. */
export const ABSENCE_GRACE_FRAMES = 20;

export interface PipelineFrameInput {
  /** Raw hand landmarks straight off the recognizer. */
  allHands: Landmark[][];
  /** Gesture guesses paired index-for-index with allHands. */
  allGestures: GestureCategory[][];
  /** True when the pose model ran this frame; its reading replaces the held one. */
  poseRan: boolean;
  bodyPose?: PoseLandmark[];
  nowMs: number;
  wantDebug: boolean;
}

export interface PipelineFrameOutput {
  /** State to report to the app — null when nothing should be emitted. */
  state: GestureState | null;
  debug: HandDebugFrame | null;
  tier: DrivingTier;
}

export class FramePipeline {
  private readonly latch = new PenLatch();
  // Same evidence-latch machinery, tuned snappier: engaging the 👍 hold only
  // starts a 1.5s dwell (the dwell is the real filter), and releasing fast
  // keeps the ring honest when the hand moves on.
  private readonly thumbLatch = new PenLatch({
    press: 0.3,
    release: 0.5,
    decay: 0.06,
    downAt: 0.6,
    upAt: 0.25,
  });
  private readonly mapper = new PadMapper();
  // Speed-adaptive smoothing: steady when the hand hovers, near-transparent
  // when it strokes. A fixed factor could only ever be one or the other.
  private readonly filterX = new OneEuroFilter();
  private readonly filterY = new OneEuroFilter();
  private lastFrameTime = 0;
  private smoothX = 0.5;
  private smoothY = 0.5;
  private smoothSpan = 0;
  private hadHand = false;
  private missedFrames = 0;
  private lastPrimaryPalm: Landmark | null = null;
  private lastBodyPose: PoseLandmark[] | undefined;
  private lastEmitted: GestureState | null = null;

  // ---- far tier: separate instances so tier handoff can't smear state ----
  private readonly farLatch = new PenLatch();
  private readonly farArmOutLatch = new PenLatch({
    press: 0.3,
    release: 0.5,
    decay: 0.06,
    downAt: 0.6,
    upAt: 0.25,
  });
  private readonly farMapper = new PadMapper(POSE_PAD_REACH);
  private readonly farFilterX = new OneEuroFilter();
  private readonly farFilterY = new OneEuroFilter();
  private farSmoothSw = 0;
  private farActive = false;
  private farMissed = 0;
  private activeArm: Side | null = null;
  private tier: DrivingTier = null;

  update(input: PipelineFrameInput): PipelineFrameOutput {
    const { allHands, allGestures, nowMs } = input;
    // Hands too small in frame are far beyond the visitor zone — ignore them.
    // Keep each hand's gesture scores paired with it through the filter.
    const hands = allHands
      .map((landmarks, i) => ({ landmarks, gestures: allGestures[i] ?? [] }))
      .filter((h) => isCloseEnough(h.landmarks));

    // ✕ detection: forearm segments crossing (body pose, long range),
    // with two-hands-wrists-together as the fallback signal. This is the
    // ONLY thing the body-pose model drives — it is far too noisy through a
    // sleeve to be trusted with the pen.
    if (input.poseRan) this.lastBodyPose = input.bodyPose;
    const bodyPose = this.lastBodyPose;
    const poseCrossed = bodyPose ? forearmsCrossed(bodyPose) : false;
    const crossed =
      poseCrossed ||
      (hands.length >= 2 && wristsClose(hands[0].landmarks[WRIST], hands[1].landmarks[WRIST]));

    const primaryIndex =
      hands.length > 0
        ? pickPrimaryHand(
            hands.map((h) => h.landmarks[PALM]),
            this.lastPrimaryPalm
          )
        : -1;

    let debug: HandDebugFrame | null = null;
    const buildDebug = input.wantDebug
      ? (): void => {
          const primaryHand = primaryIndex >= 0 ? hands[primaryIndex] : null;
          debug = {
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
            charge: this.latch.level(),
            crossed,
            tier: this.tier,
            bodyPose: bodyPose ?? null,
            farReach: this.activeArm
              ? { side: this.activeArm, charge: this.farLatch.level() }
              : null,
            state: this.lastEmitted,
          };
        }
      : null;

    if (hands.length === 0) {
      if (!this.hadHand) {
        // Close tier idle → the far tier reads the whole vocabulary from
        // body pose. Same GestureState out, so nothing downstream can tell.
        const sw = bodyPose ? shoulderWidth(bodyPose) : 0;
        const arm = bodyPose && sw > 0 ? pickActiveArm(bodyPose, this.activeArm) : null;
        if (arm === null) {
          if (!this.farActive) {
            this.tier = null;
            buildDebug?.();
            return { state: null, debug, tier: null };
          }
          // Far dropout: same shape as a vanished hand — bleed the latches,
          // hold the cursor, report absence only after the grace window.
          const pose = this.farLatch.update(NO_EVIDENCE);
          const thumbsUp = this.farArmOutLatch.update(NO_EVIDENCE) === "fist";
          if (++this.farMissed > ABSENCE_GRACE_FRAMES) {
            this.resetFar();
            this.lastFrameTime = 0;
            this.tier = null;
            const state = this.emit({
              present: false,
              x: this.smoothX,
              y: this.smoothY,
              pose: "open",
              raised: false,
              crossed: false,
              thumbsUp: false,
            });
            buildDebug?.();
            return { state, debug, tier: null };
          }
          this.tier = "far";
          const state = this.emit({
            present: true,
            x: this.smoothX,
            y: this.smoothY,
            pose,
            raised: false,
            crossed,
            thumbsUp,
          });
          buildDebug?.();
          return { state, debug, tier: "far" };
        }
        this.farMissed = 0;
        this.activeArm = arm;
        this.tier = "far";
        const wrist = wristOf(bodyPose!, arm);
        // Smooth the ruler like the close tier smooths hand-span.
        this.farSmoothSw =
          this.farSmoothSw === 0 ? sw : this.farSmoothSw + (sw - this.farSmoothSw) * 0.1;
        const mapped = this.farMapper.update(wrist, this.farSmoothSw);
        if (!this.farActive) {
          this.farFilterX.reset();
          this.farFilterY.reset();
          this.farActive = true;
        }
        const dt = this.lastFrameTime ? (nowMs - this.lastFrameTime) / 1000 : NOMINAL_DT;
        this.lastFrameTime = nowMs;
        this.smoothX = this.farFilterX.filter(mapped.x, dt);
        this.smoothY = this.farFilterY.filter(mapped.y, dt);
        const pose = this.farLatch.update(reachEvidence(bodyPose!, arm));
        const thumbsUp =
          this.farArmOutLatch.update(
            armOut(bodyPose!, arm) ? { fist: 1, open: 0 } : { fist: 0, open: 1 }
          ) === "fist";
        const state = this.emit({
          present: true,
          x: this.smoothX,
          y: this.smoothY,
          pose,
          raised: poseRaised(bodyPose!, arm),
          crossed,
          thumbsUp,
        });
        buildDebug?.();
        return { state, debug, tier: "far" };
      }
      // No hand is not a fist. Bleed the latches down instead of freezing
      // the last pose: a brief dropout rides through, a real absence lifts
      // the pen well before the cursor itself is withdrawn.
      const pose = this.latch.update(NO_EVIDENCE);
      const thumbsUp = this.thumbLatch.update(NO_EVIDENCE) === "fist";
      if (++this.missedFrames > ABSENCE_GRACE_FRAMES) {
        this.hadHand = false;
        this.latch.reset();
        this.thumbLatch.reset();
        this.mapper.reset();
        this.lastFrameTime = 0;
        this.lastPrimaryPalm = null;
        this.tier = null;
        const state = this.emit({
          present: false,
          x: this.smoothX,
          y: this.smoothY,
          pose: "open",
          raised: false,
          crossed: false,
          thumbsUp: false,
        });
        buildDebug?.();
        return { state, debug, tier: null };
      }
      // Hold the cursor where it was — never invent a position from a model
      // that cannot see through a sleeve.
      this.tier = "close";
      const state = this.emit({
        present: true,
        x: this.smoothX,
        y: this.smoothY,
        pose,
        raised: false,
        crossed,
        thumbsUp,
      });
      buildDebug?.();
      return { state, debug, tier: "close" };
    }

    this.missedFrames = 0;
    if (this.farActive || this.activeArm) this.resetFar();
    this.tier = "close";
    const primary = hands[primaryIndex];
    const palm = primary.landmarks[PALM];
    this.lastPrimaryPalm = { x: palm.x, y: palm.y };
    // Smooth the span so a single misread frame doesn't warp the mapper box
    const span = handSpan(primary.landmarks);
    this.smoothSpan = this.smoothSpan === 0 ? span : this.smoothSpan + (span - this.smoothSpan) * 0.1;
    const mapped = this.mapper.update(palm, this.smoothSpan);
    if (!this.hadHand) {
      // Snap on reacquire so the cursor doesn't glide in from its old spot
      this.filterX.reset();
      this.filterY.reset();
      this.hadHand = true;
    }
    // Real elapsed time, not an assumed 30fps — the camera's rate varies with
    // light, and a filter fed the wrong dt mis-scales its own speed estimate.
    const dt = this.lastFrameTime ? (nowMs - this.lastFrameTime) / 1000 : NOMINAL_DT;
    this.lastFrameTime = nowMs;
    this.smoothX = this.filterX.filter(mapped.x, dt);
    this.smoothY = this.filterY.filter(mapped.y, dt);
    const pose = this.latch.update(penEvidence(primary.gestures));
    const thumbsUp = this.thumbLatch.update(thumbEvidence(primary.gestures)) === "fist";
    const state = this.emit({
      present: true,
      x: this.smoothX,
      y: this.smoothY,
      pose,
      raised: isRaised(primary.landmarks),
      crossed,
      thumbsUp,
    });
    buildDebug?.();
    return { state, debug, tier: "close" };
  }

  private emit(state: GestureState): GestureState {
    this.lastEmitted = state;
    return state;
  }

  private resetFar(): void {
    this.farActive = false;
    this.farMissed = 0;
    this.activeArm = null;
    this.farLatch.reset();
    this.farArmOutLatch.reset();
    this.farMapper.reset();
    this.farFilterX.reset();
    this.farFilterY.reset();
    this.farSmoothSw = 0;
  }
}
