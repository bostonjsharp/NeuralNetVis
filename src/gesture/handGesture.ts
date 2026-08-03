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

/** What the pen is doing. There is no third "unknown" state on purpose —
 *  see PenLatch: uncertainty is an absence of drive, not a pose. */
export type PenPose = "fist" | "open";

export const WRIST = 0;

/** One canned-gesture guess from MediaPipe's GestureRecognizer. */
export interface GestureCategory {
  categoryName: string;
  score: number;
}

/** How strongly this frame argues for pen-down vs pen-up, each in 0..1. */
export interface PenEvidence {
  fist: number;
  open: number;
}

export const NO_EVIDENCE: PenEvidence = { fist: 0, open: 0 };

/** Splayed fingers — unambiguously not a pen grip. */
const OPEN_GESTURES = new Set(["Open_Palm", "Victory", "Pointing_Up"]);

/**
 * Turns the recognizer's canned-gesture scores into pen evidence.
 *
 * Anything we can't read (`None`, an empty list, or a curled-but-not-fist
 * shape like Thumb_Up) yields NO_EVIDENCE rather than a guess. That matters:
 * the old ratio classifier bucketed every uncertain frame as "unknown", and
 * the stabilizer treated "unknown" as "hold current state" — so a hand the
 * model half-saw kept the pen welded down. Uncertainty must be *silent*, so
 * the latch below can decay the pen back up on its own.
 */
export function penEvidence(gestures: GestureCategory[]): PenEvidence {
  let fist = 0;
  let open = 0;
  for (const { categoryName, score } of gestures) {
    if (categoryName === "Closed_Fist") fist = Math.max(fist, score);
    else if (OPEN_GESTURES.has(categoryName)) open = Math.max(open, score);
  }
  return { fist, open };
}

/**
 * Evidence for the 👍 brain-switch verb, shaped for PenLatch (which is a
 * generic evidence-integrating Schmitt latch — here `fist` carries "is a
 * thumbs-up" and `open` carries "is confidently something else"). Any other
 * NAMED gesture argues against; an unreadable hand argues nothing, so the
 * latch's decay handles 2m tracking flicker the same way it does for the pen.
 */
export function thumbEvidence(gestures: GestureCategory[]): PenEvidence {
  let thumb = 0;
  let other = 0;
  for (const { categoryName, score } of gestures) {
    if (categoryName === "Thumb_Up") thumb = Math.max(thumb, score);
    else if (categoryName !== "None") other = Math.max(other, score);
  }
  return { fist: thumb, open: other };
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
 * Sticky pen-hand selection. On first acquisition the LIFTED hand (highest
 * palm in frame) becomes the pen; afterwards the pen identity follows
 * whichever detected hand is nearest the pen's last position, so a second
 * hand entering the frame never steals the cursor.
 */
export function pickPrimaryHand(palms: Landmark[], lastPalm: Landmark | null): number {
  if (palms.length <= 1) return 0;
  let best = 0;
  if (lastPalm === null) {
    for (let i = 1; i < palms.length; i++) {
      if (palms[i].y < palms[best].y) best = i; // camera y grows downward
    }
    return best;
  }
  let bestDist = Infinity;
  for (let i = 0; i < palms.length; i++) {
    const d = Math.hypot(palms[i].x - lastPalm.x, palms[i].y - lastPalm.y);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
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

/** A plain exponential low-pass, re-seedable. Building block of OneEuro. */
class LowPass {
  private value: number | null = null;

  filter(x: number, alpha: number): number {
    this.value = this.value === null ? x : this.value + alpha * (x - this.value);
    return this.value;
  }

  reset(): void {
    this.value = null;
  }
}

export interface OneEuroOptions {
  /** Cutoff (Hz) when the hand is still — lower = steadier, but laggier. */
  minCutoff: number;
  /** How aggressively the cutoff opens up with speed — the anti-lag term. */
  beta: number;
  /** Cutoff (Hz) for the speed estimate itself. */
  dCutoff: number;
}

/**
 * Tuned in pad units (0..1) at camera rate, by the filter's own recipe: set
 * `minCutoff` for the jitter you accept while the hand hovers, then raise
 * `beta` until fast strokes stop lagging.
 *
 * A quick digit stroke crosses the pad in ~0.3s (speed ≈ 3/s). At beta 3 that
 * lifts the cutoff from 1Hz to ~10Hz, leaving under 0.05 pad units of lag —
 * about a frame of travel, versus 0.19 (a fifth of the pad!) under the fixed
 * 0.35 factor this replaces. A hovering hand barely moves, so its cutoff
 * stays near 1Hz and it is smoothed *harder* than before.
 */
const DEFAULT_EURO: OneEuroOptions = { minCutoff: 1.0, beta: 3.0, dCutoff: 1.0 };

/**
 * One Euro filter (Casiez, Roussel & Vogel, CHI 2012).
 *
 * Fixed exponential smoothing forces a single choice between jitter at rest
 * and lag in motion — its tracking error grows in direct proportion to hand
 * speed, which is why fast strokes trailed and cut their corners. This makes
 * the cutoff frequency a function of the estimated speed: it filters hard
 * when the hand is nearly still, and opens up (approaching raw passthrough)
 * as the hand moves, so quick strokes land where they were drawn.
 */
export class OneEuroFilter {
  private readonly x = new LowPass();
  private readonly dx = new LowPass();
  private previous: number | null = null;

  constructor(private readonly options: OneEuroOptions = DEFAULT_EURO) {}

  /** Smoothing factor for a cutoff frequency over a timestep. */
  private static alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  /** @param dt seconds since the previous sample. */
  filter(value: number, dt: number): number {
    const step = dt > 0 ? dt : 1 / 30; // a stalled clock must not divide by zero
    const speed = this.previous === null ? 0 : (value - this.previous) / step;
    this.previous = value;
    const smoothSpeed = this.dx.filter(speed, OneEuroFilter.alpha(this.options.dCutoff, step));
    const cutoff = this.options.minCutoff + this.options.beta * Math.abs(smoothSpeed);
    return this.x.filter(value, OneEuroFilter.alpha(cutoff, step));
  }

  reset(): void {
    this.x.reset();
    this.dx.reset();
    this.previous = null;
  }
}

export interface PenLatchOptions {
  /** Charge gained per frame at full fist confidence. */
  press: number;
  /** Charge shed per frame at full open confidence. */
  release: number;
  /** Charge shed per frame when the hand reads as nothing at all. */
  decay: number;
  /** Schmitt trigger: charge at/above this puts the pen down. */
  downAt: number;
  /** …and at/below this lifts it. The gap is the anti-flicker margin. */
  upAt: number;
}

/** Press ≈4 confident frames; release ≈2. An open hand must always be able
 *  to win faster than a fist can grab — the opposite bias made the pen
 *  impossible to put down. `decay` alone lifts the pen in ~0.5s at 30fps,
 *  which is the safety net for a hand the model simply cannot read. */
const DEFAULT_LATCH: PenLatchOptions = {
  press: 0.25,
  release: 0.4,
  decay: 0.05,
  downAt: 0.7,
  upAt: 0.3,
};

/**
 * Integrates per-frame gesture confidence into a pen up/down decision.
 *
 * A charge in 0..1 is driven up by fist evidence, down by open evidence, and
 * *always* bled down by `decay` when the frame carries no evidence either
 * way. A Schmitt trigger reads the charge, so a stroke rides through the
 * intermittent misses that sleeves and 2m of distance guarantee — but a hand
 * the model can no longer read as a fist will always, eventually, let go.
 *
 * This replaces a consecutive-frame streak counter, which could be reset to
 * zero by a single uncertain frame and therefore never completed a release.
 */
export class PenLatch {
  private charge = 0;
  private down = false;

  constructor(private readonly options: PenLatchOptions = DEFAULT_LATCH) {}

  update(evidence: PenEvidence): PenPose {
    const { press, release, decay, downAt, upAt } = this.options;
    const net = evidence.fist - evidence.open;
    if (net > 0) this.charge += net * press;
    else if (net < 0) this.charge += net * release; // net is negative
    else this.charge -= decay; // no evidence, or a perfect tie → fail safe
    this.charge = Math.min(1, Math.max(0, this.charge));

    if (this.charge >= downAt) this.down = true;
    else if (this.charge <= upAt) this.down = false;
    return this.pose();
  }

  pose(): PenPose {
    return this.down ? "fist" : "open";
  }

  /** Live charge, for the debug overlay. */
  level(): number {
    return this.charge;
  }

  reset(): void {
    this.charge = 0;
    this.down = false;
  }
}
