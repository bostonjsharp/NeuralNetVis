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

export type HandPose = "fist" | "open" | "unknown";

/** Finger [PIP, TIP] landmark indices — thumb excluded (unreliable folder). */
const FINGERS: readonly [number, number][] = [
  [6, 8], // index
  [10, 12], // middle
  [14, 16], // ring
  [18, 20], // pinky
];
export const WRIST = 0;

/**
 * Deliberately strict margins for a camera 2+ meters away: a finger only
 * counts as EXTENDED when its tip is clearly past its middle joint
 * (fully straightened), and only as FOLDED when clearly curled back.
 * Half-open hands land in neither bucket, so casual poses read as
 * "unknown" and the stabilizer holds the previous state — visitors must
 * really clench or really splay.
 */
const EXTENDED_RATIO = 1.25;
/** Generous: at 2m a clenched fist's fingers read as barely folded, and a
 *  lost fist mid-stroke chops the drawing. */
const FOLDED_RATIO = 1.12;

export function classifyHand(landmarks: Landmark[]): HandPose {
  const wrist = landmarks[WRIST];
  const dist = (a: Landmark) => Math.hypot(a.x - wrist.x, a.y - wrist.y);
  let extendedCount = 0;
  let foldedCount = 0;
  for (const [pip, tip] of FINGERS) {
    const ratio = dist(landmarks[tip]) / Math.max(dist(landmarks[pip]), 1e-6);
    if (ratio > EXTENDED_RATIO) extendedCount++;
    if (ratio < FOLDED_RATIO) foldedCount++;
  }
  if (foldedCount >= 3 && extendedCount === 0) return "fist";
  if (extendedCount >= 3 && foldedCount === 0) return "open";
  return "unknown";
}

/** Palm anchor: middle-finger MCP tracks the hand's center steadily. */
export const PALM = 9;

/** A hand is "raised" when the palm sits in the upper part of the camera
 *  frame — an arm held up, not a hand at waist height. Any pose counts;
 *  at 2m the 5-second hold is the false-positive filter, not the pose. */
export const RAISE_LINE = 0.45;

export function isRaised(landmarks: Landmark[]): boolean {
  return landmarks[PALM].y < RAISE_LINE;
}

/**
 * Wrist→palm span in normalized camera units — a proxy for how close the
 * hand is. A visitor's hand at ~2m spans roughly 0.03–0.05 of a typical
 * webcam frame; anything below this is someone much farther out (or a
 * misdetection) and is ignored entirely.
 */
export const MIN_HAND_SPAN = 0.022;

export function handSpan(landmarks: Landmark[]): number {
  const wrist = landmarks[WRIST];
  const palm = landmarks[PALM];
  return Math.hypot(palm.x - wrist.x, palm.y - wrist.y);
}

export function isCloseEnough(landmarks: Landmark[]): boolean {
  return handSpan(landmarks) >= MIN_HAND_SPAN;
}

/** Two WRISTS near each other — arms crossed in an ✕. Wrists are the
 *  right anchor: when forearms cross, the wrists touch while the palms
 *  splay far apart in opposite directions. */
export const CROSS_DISTANCE = 0.25;

export function wristsClose(a: Landmark, b: Landmark): boolean {
  return Math.hypot(a.x - b.x, a.y - b.y) < CROSS_DISTANCE;
}

/**
 * Maps hand position to pad space RELATIVE to the hand's apparent size,
 * so drawing needs the same comfortable physical movement at any distance
 * from the camera. A virtual box `reach` hand-spans wide is anchored where
 * the hand appears (cursor starts at pad center); moving within the box
 * moves the cursor, and pushing past an edge drags the box along. X is
 * mirrored for selfie view.
 */
export class PadMapper {
  private anchorX: number | null = null;
  private anchorY = 0;

  /** `reach` = hand-spans of travel to cross the whole pad. */
  constructor(private readonly reach = 5) {}

  update(palm: Landmark, span: number): { x: number; y: number } {
    // Floor keeps the box usable if span misreads tiny for a frame
    const width = Math.max(this.reach * span, 0.1);
    if (this.anchorX === null) {
      this.anchorX = palm.x;
      this.anchorY = palm.y;
    }
    let ox = (palm.x - this.anchorX) / width;
    if (ox > 0.5) {
      this.anchorX = palm.x - 0.5 * width;
      ox = 0.5;
    } else if (ox < -0.5) {
      this.anchorX = palm.x + 0.5 * width;
      ox = -0.5;
    }
    let oy = (palm.y - this.anchorY) / width;
    if (oy > 0.5) {
      this.anchorY = palm.y - 0.5 * width;
      oy = 0.5;
    } else if (oy < -0.5) {
      this.anchorY = palm.y + 0.5 * width;
      oy = -0.5;
    }
    return { x: 0.5 - ox, y: 0.5 + oy };
  }

  reset(): void {
    this.anchorX = null;
  }
}

/** Frames a pose flip must hold before it's reported. Asymmetric: the pen
 *  presses quickly but releases slowly, so a noisy frame or two mid-stroke
 *  never chops the drawing. */
const DEFAULT_HOLD_FRAMES: Record<Exclude<HandPose, "unknown">, number> = {
  fist: 4,
  open: 8,
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
