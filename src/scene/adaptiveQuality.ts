import { STAGE_HEIGHT, STAGE_WIDTH } from "../app/constants";

/**
 * Measured-performance quality governor. The wall canvas is authored at
 * 2736×1216, but the GPU actually driving it varies wildly (the kiosk, a
 * dev laptop, a borrowed projector PC). Two pure pieces keep every one of
 * them smooth:
 *
 *  - renderResolution: never rasterize more pixels than the display shows.
 *  - AdaptiveQuality: walk down a fixed ladder of effect settings whenever
 *    sustained frame times blow the budget. Down only, never back up — a
 *    kiosk that oscillates between quality levels looks broken.
 */

export interface QualityLevel {
  /** MSAA samples on the composer's HDR target (0 disables). */
  msaa: number;
  /** Bloom runs at buffer/bloomScale — it's a haze, cheap to undersample. */
  bloomScale: number;
  /** Extra multiplier on the drawing-buffer size (last resort). */
  renderScale: number;
}

/** Costliest first. MSAA goes before bloom sharpness: on bandwidth-starved
 *  iGPUs the 4× HalfFloat resolve dwarfs everything else in the frame. */
export const QUALITY_LADDER: QualityLevel[] = [
  { msaa: 4, bloomScale: 2, renderScale: 1 },
  { msaa: 0, bloomScale: 2, renderScale: 1 },
  { msaa: 0, bloomScale: 4, renderScale: 1 },
  { msaa: 0, bloomScale: 4, renderScale: 0.75 },
];

/** Entry rungs for the ?quality= query param ("auto" adapts from the top). */
export function startLevelFor(quality: "auto" | "high" | "low"): number {
  return quality === "low" ? 2 : 0;
}

/** Buffer never drops below this fraction of native — text on the input
 *  plane must stay legible even in a tiny dev preview window. */
const MIN_RESOLUTION_SCALE = 0.3;

/**
 * Drawing-buffer size for the current display. `stageScale` is the CSS
 * scale Stage applies to fit the wall canvas into the viewport; `dpr` the
 * devicePixelRatio. Rendering above displayed-physical-pixels is invisible
 * waste (the browser downsamples it anyway), so cap there — and at native,
 * because the wall itself (scale 1) must never be exceeded either.
 */
export function renderResolution(
  stageScale: number,
  dpr: number,
  renderScale: number
): { width: number; height: number } {
  const scale = Math.max(
    MIN_RESOLUTION_SCALE,
    Math.min(1, stageScale * dpr) * renderScale
  );
  return {
    width: Math.round(STAGE_WIDTH * scale),
    height: Math.round(STAGE_HEIGHT * scale),
  };
}

/** Sustained p95 above this triggers a step down. Comfortably past the
 *  16.7ms/60fps budget so vsync jitter alone can't demote a healthy GPU. */
const P95_BUDGET_MS = 22;
/** Frame-time readings in the first seconds include shader compiles and
 *  model loads — never judge the GPU on its warmup. */
const WARMUP_MS = 6_000;
/** After a step the perf ring still holds pre-change frames; wait for the
 *  window to refill with post-change truth before judging again. */
const COOLDOWN_MS = 12_000;
/** A p95 needs a real population behind it. */
const MIN_FRAMES = 120;

export interface AdaptiveQualityOptions {
  startLevel: number;
  /** Explicit ?quality= pins the rung — measurement never moves it. */
  locked?: boolean;
}

export class AdaptiveQuality {
  level: number;
  private readonly locked: boolean;
  private lastChangeAt: number;

  constructor(options: AdaptiveQualityOptions) {
    this.level = options.startLevel;
    this.locked = options.locked ?? false;
    this.lastChangeAt = 0;
  }

  /**
   * Feed a perf snapshot; returns the new ladder index when a step down is
   * due, null otherwise. Callers apply the returned level to the renderer.
   */
  update(snapshot: { p95: number; frames: number }, nowMs: number): number | null {
    if (this.locked) return null;
    if (this.level >= QUALITY_LADDER.length - 1) return null;
    if (nowMs < WARMUP_MS) return null;
    if (nowMs - this.lastChangeAt < COOLDOWN_MS) return null;
    if (snapshot.frames < MIN_FRAMES) return null;
    if (snapshot.p95 <= P95_BUDGET_MS) return null;
    this.level++;
    this.lastChangeAt = nowMs;
    return this.level;
  }
}
