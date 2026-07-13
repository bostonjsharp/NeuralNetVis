/**
 * Pure hand-gesture math — no camera, no MediaPipe imports, unit-testable.
 * The tracker feeds MediaPipe's 21 hand landmarks through these.
 *
 * Gesture language: ✊ closed fist = pen down, ✋ open hand = pen up.
 */

export interface Landmark {
  x: number;
  y: number;
}

export type HandPose = "fist" | "open" | "two" | "unknown";

/** Finger [PIP, TIP] landmark indices — thumb excluded (unreliable folder). */
const FINGERS: readonly [number, number][] = [
  [6, 8], // index
  [10, 12], // middle
  [14, 16], // ring
  [18, 20], // pinky
];
const WRIST = 0;

/**
 * A finger is extended when its tip is farther from the wrist than its
 * middle joint. ≤1 extended reads as a fist, ≥3 as open, exactly
 * index+middle (✌️) is the clear gesture; anything else is ambiguous
 * (mid-curl) and callers keep their previous state.
 */
export function classifyHand(landmarks: Landmark[]): HandPose {
  const wrist = landmarks[WRIST];
  const dist = (a: Landmark) => Math.hypot(a.x - wrist.x, a.y - wrist.y);
  const extended = FINGERS.map(([pip, tip]) => dist(landmarks[tip]) > dist(landmarks[pip]));
  const count = extended.filter(Boolean).length;
  if (count === 2 && extended[0] && extended[1]) return "two";
  if (count <= 1) return "fist";
  if (count >= 3) return "open";
  return "unknown";
}

/**
 * Maps a normalized camera-space hand position to normalized pad space.
 * Mirrored (webcams are selfie-view), and only the central region of the
 * camera frame is used so reaching the pad's corners doesn't require
 * stretching to the edge of the camera's view.
 */
const REGION_MIN = 0.22;
const REGION_MAX = 0.78;

export function mapToPad(cameraX: number, cameraY: number): { x: number; y: number } {
  const span = REGION_MAX - REGION_MIN;
  const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
  return {
    x: clamp01((1 - cameraX - REGION_MIN) / span),
    y: clamp01((cameraY - REGION_MIN) / span),
  };
}

/** Frames a pose flip must hold before it's reported. ✌️ (clear) needs a
 *  deliberate hold because fingers pass through two-extended mid-curl. */
const DEFAULT_HOLD_FRAMES: Record<Exclude<HandPose, "unknown">, number> = {
  fist: 3,
  open: 3,
  two: 12,
};

/**
 * Debounces raw per-frame poses: a state flip must hold for a streak of
 * consecutive frames before it's reported, so a mid-curl flicker doesn't
 * chop one stroke into many or fire an accidental clear.
 */
export class PoseStabilizer {
  private stable: HandPose = "open";
  private candidate: HandPose | null = null;
  private streak = 0;

  constructor(private readonly holdFrames = DEFAULT_HOLD_FRAMES) {}

  update(raw: HandPose): HandPose {
    if (raw === "unknown" || raw === this.stable) {
      this.candidate = null;
      this.streak = 0;
      return this.stable;
    }
    if (raw !== this.candidate) {
      this.candidate = raw;
      this.streak = 1;
    } else if (++this.streak >= this.holdFrames[raw]) {
      this.stable = raw;
      this.candidate = null;
      this.streak = 0;
    }
    return this.stable;
  }

  reset(): void {
    this.stable = "open";
    this.candidate = null;
    this.streak = 0;
  }
}
