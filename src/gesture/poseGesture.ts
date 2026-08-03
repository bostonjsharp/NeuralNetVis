import { NO_EVIDENCE, type Landmark, type PenEvidence } from "./handGesture";

/**
 * Pure body-pose math — the pose-side twin of handGesture.ts, no camera,
 * no MediaPipe imports, unit-testable. Body pose reads at far greater
 * range than hands; past ~2m it is the only signal the pipeline has.
 */

export interface PoseLandmark extends Landmark {
  /** Depth relative to the hip midpoint, roughly x-scale units; more
   *  negative = toward the camera. The noisiest channel MediaPipe ships —
   *  only ever consumed through a PenLatch's hysteresis. */
  z?: number;
  visibility?: number;
}

/** MediaPipe pose landmark indices (the upper-body subset we read). */
export const POSE = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
} as const;

export const POSE_VISIBILITY_MIN = 0.5;

export type Side = "left" | "right";

export function visible(p: PoseLandmark | undefined): p is PoseLandmark {
  return !!p && (p.visibility ?? 1) >= POSE_VISIBILITY_MIN;
}

/** Shoulder-to-shoulder distance in normalized units — the far tier's size
 *  ruler, playing the role hand-span plays up close. 0 when unreadable. */
export function shoulderWidth(pose: PoseLandmark[]): number {
  const l = pose[POSE.leftShoulder];
  const r = pose[POSE.rightShoulder];
  if (!visible(l) || !visible(r)) return 0;
  return Math.hypot(l.x - r.x, l.y - r.y);
}

export function wristOf(pose: PoseLandmark[], side: Side): PoseLandmark {
  return pose[side === "left" ? POSE.leftWrist : POSE.rightWrist];
}

function shoulderOf(pose: PoseLandmark[], side: Side): PoseLandmark {
  return pose[side === "left" ? POSE.leftShoulder : POSE.rightShoulder];
}

function hipOf(pose: PoseLandmark[], side: Side): PoseLandmark {
  return pose[side === "left" ? POSE.leftHip : POSE.rightHip];
}

/** Wrist above the same-side shoulder by this margin = the wake pose. The
 *  margin keeps a forward reach (wrist near shoulder height) from reading
 *  as a wake. */
export const POSE_RAISE_MARGIN = 0.05;

export function poseRaised(pose: PoseLandmark[], side: Side): boolean {
  const wrist = wristOf(pose, side);
  const shoulder = shoulderOf(pose, side);
  if (!visible(wrist) || !visible(shoulder)) return false;
  return wrist.y < shoulder.y - POSE_RAISE_MARGIN;
}

/** Arm straight out to the side — the far-tier brain-cycle verb. Bounds in
 *  shoulder-widths so the pose reads the same at any distance. Geometrically
 *  orthogonal to wake (up) and reach (forward) by construction. */
export const ARM_OUT_LATERAL = 0.8;
export const ARM_OUT_VERTICAL = 0.5;

export function armOut(pose: PoseLandmark[], side: Side): boolean {
  const wrist = wristOf(pose, side);
  const shoulder = shoulderOf(pose, side);
  const sw = shoulderWidth(pose);
  if (sw === 0 || !visible(wrist) || !visible(shoulder)) return false;
  const lateral = Math.abs(wrist.x - shoulder.x) / sw;
  const vertical = Math.abs(wrist.y - shoulder.y) / sw;
  return lateral > ARM_OUT_LATERAL && vertical < ARM_OUT_VERTICAL;
}

/** Reach ratio ((shoulder.z − wrist.z) / shoulderWidth) where the pen-down
 *  drive begins, and where it saturates. THE on-site tuning pair: watch the
 *  reach bar in the debug overlay and move these until a relaxed arm sits
 *  at zero and a comfortable reach pins the bar. */
export const REACH_REST = 0.5;
export const REACH_FULL = 1.5;

/**
 * Reach-toward-the-screen as pen evidence, shaped for PenLatch (`fist`
 * = pen-down drive, `open` = pen-up drive, same convention as
 * thumbEvidence). Unreadable input yields NO_EVIDENCE so the latch's decay
 * lifts the pen on its own — uncertainty must stay silent.
 */
export function reachEvidence(pose: PoseLandmark[], side: Side): PenEvidence {
  const wrist = wristOf(pose, side);
  const shoulder = shoulderOf(pose, side);
  const sw = shoulderWidth(pose);
  if (sw === 0 || !visible(wrist) || !visible(shoulder)) return NO_EVIDENCE;
  if (wrist.z === undefined || shoulder.z === undefined) return NO_EVIDENCE;
  // An arm hanging past the hip can't be a reach, whatever z claims.
  const hip = hipOf(pose, side);
  if (visible(hip) && wrist.y > hip.y) return { fist: 0, open: 1 };
  const reach = (shoulder.z - wrist.z) / sw;
  const t = (reach - REACH_REST) / (REACH_FULL - REACH_REST);
  if (t >= 1) return { fist: 1, open: 0 };
  if (t > 0) return { fist: t, open: 0 };
  return { fist: 0, open: Math.min(1, (REACH_REST - reach) / REACH_REST) };
}

/** Pad box width in shoulder-widths — PadMapper's `reach` parameter when
 *  the far tier feeds it wrist + shoulderWidth instead of palm + hand-span,
 *  so drawing needs the same comfortable arm travel at any distance. */
export const POSE_PAD_REACH = 2.5;

/** Sticky active-arm choice, mirroring pickPrimaryHand: keep the current
 *  arm while its wrist reads; otherwise the higher visible wrist wins.
 *  (Pose sides are stable identities — no distance-matching needed.) */
export function pickActiveArm(pose: PoseLandmark[], last: Side | null): Side | null {
  const leftOk = visible(pose[POSE.leftWrist]);
  const rightOk = visible(pose[POSE.rightWrist]);
  if (last === "left" && leftOk) return "left";
  if (last === "right" && rightOk) return "right";
  if (leftOk && rightOk) {
    return pose[POSE.leftWrist].y <= pose[POSE.rightWrist].y ? "left" : "right";
  }
  if (leftOk) return "left";
  if (rightOk) return "right";
  return null;
}

/** Strict 2D segment intersection (shared endpoints/collinear don't count). */
export function segmentsIntersect(
  a1: Landmark,
  a2: Landmark,
  b1: Landmark,
  b2: Landmark
): boolean {
  const cross = (o: Landmark, p: Landmark, q: Landmark) =>
    (p.x - o.x) * (q.y - o.y) - (p.y - o.y) * (q.x - o.x);
  const d1 = cross(b1, b2, a1);
  const d2 = cross(b1, b2, a2);
  const d3 = cross(a1, a2, b1);
  const d4 = cross(a1, a2, b2);
  return d1 > 0 !== d2 > 0 && d3 > 0 !== d4 > 0;
}

/** The literal ✕ test: do the two forearm segments (elbow→wrist) cross? */
export function forearmsCrossed(pose: PoseLandmark[]): boolean {
  const points = [
    pose[POSE.leftElbow],
    pose[POSE.leftWrist],
    pose[POSE.rightElbow],
    pose[POSE.rightWrist],
  ];
  if (points.some((p) => !p || (p.visibility ?? 1) < POSE_VISIBILITY_MIN)) return false;
  return segmentsIntersect(
    pose[POSE.leftElbow],
    pose[POSE.leftWrist],
    pose[POSE.rightElbow],
    pose[POSE.rightWrist]
  );
}
