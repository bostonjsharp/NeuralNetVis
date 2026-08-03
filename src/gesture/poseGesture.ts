import type { Landmark } from "./handGesture";

/**
 * Pure body-pose math — the pose-side twin of handGesture.ts, no camera,
 * no MediaPipe imports, unit-testable. Body pose reads at far greater
 * range than hands; past ~2m it is the only signal the pipeline has.
 */

export interface PoseLandmark extends Landmark {
  visibility?: number;
}

/** MediaPipe pose landmark indices. */
export const POSE = {
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
} as const;

export const POSE_VISIBILITY_MIN = 0.5;

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
