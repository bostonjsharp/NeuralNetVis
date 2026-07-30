import { smoothstep } from "./fire";

/**
 * Ambient motion for interactive mode — the fire.ts sibling. Pure math, no
 * WebGL: camera micro-drift, a coherent breathing wave, the duck envelope
 * that silences ambience while a fire cinematic or morph owns the stage,
 * and (added by later tasks) draw-excitement state and spark scheduling.
 *
 * Every tuning constant lives here, named, so the whole feel is retuned in
 * one place on the actual wall.
 */

// ── Camera micro-drift ──────────────────────────────────────────────────
export const DRIFT_POS_AMP = 0.35;
export const DRIFT_TARGET_AMP = 0.15;

// ── Breathing wave ──────────────────────────────────────────────────────
export const BREATHE_PERIOD_S = 8;
export const BREATHE_BRIGHTNESS = 0.03;
export const BREATHE_SCALE = 0.02;
/** Phase lag per world unit — swells roll across the network like a tide. */
const BREATHE_WAVE_X = 0.28;
const BREATHE_WAVE_Y = 0.16;

// ── Duck envelope ───────────────────────────────────────────────────────
export const DUCK_FLOOR = 0.15;
export const DUCK_ATTACK_S = 0.3;
export const DUCK_RELEASE_S = 1.0;

const TAU = Math.PI * 2;

export interface CameraDriftOffset {
  px: number;
  py: number;
  pz: number;
  tx: number;
  ty: number;
}

/** Incommensurate periods (seconds) so the drift path never visibly repeats. */
const DRIFT_PERIOD = { px: 11, py: 17, pz: 23, tx: 13, ty: 19 } as const;

/** Lissajous-style offset for the interactive camera framing. Slow enough
 *  to be felt as parallax, never seen as movement. */
export function cameraDrift(elapsed: number): CameraDriftOffset {
  return {
    px: DRIFT_POS_AMP * Math.sin((elapsed / DRIFT_PERIOD.px) * TAU),
    py: DRIFT_POS_AMP * 0.6 * Math.sin((elapsed / DRIFT_PERIOD.py) * TAU),
    pz: DRIFT_POS_AMP * 0.5 * Math.sin((elapsed / DRIFT_PERIOD.pz) * TAU),
    tx: DRIFT_TARGET_AMP * Math.sin((elapsed / DRIFT_PERIOD.tx) * TAU),
    ty: DRIFT_TARGET_AMP * 0.7 * Math.sin((elapsed / DRIFT_PERIOD.ty) * TAU),
  };
}

/** 0..1 traveling brightness wave; phase depends on world position so the
 *  swell moves across the network instead of twinkling per neuron. */
export function breathe(elapsed: number, x: number, y: number): number {
  const phase =
    (elapsed / BREATHE_PERIOD_S) * TAU - x * BREATHE_WAVE_X - y * BREATHE_WAVE_Y;
  return 0.5 + 0.5 * Math.sin(phase);
}

/**
 * Master ambient volume, 0..1. Idle (fireStart = -1e9 makes tSinceFire huge)
 * saturates the release ramp and returns 1 with no special case. A fresh
 * fire ramps down to the floor over the attack; ambience recovers over the
 * release once the cinematic's total has elapsed. Morphs silence everything.
 */
export function duckEnvelope(
  tSinceFire: number,
  fireTotal: number,
  morphing: boolean
): number {
  if (morphing) return 0;
  if (tSinceFire < fireTotal) {
    return 1 - (1 - DUCK_FLOOR) * smoothstep(0, DUCK_ATTACK_S, tSinceFire);
  }
  return (
    DUCK_FLOOR + (1 - DUCK_FLOOR) * smoothstep(0, DUCK_RELEASE_S, tSinceFire - fireTotal)
  );
}

// ── Draw excitement ─────────────────────────────────────────────────────
export const EXCITEMENT_TAU_S = 1.5;
/** A pixel's ink gain ≥ this marks it "recently inked" for spark bias. */
export const INK_THRESHOLD = 0.15;
/** Scales a stroke event's summed fresh ink into an excitement bump. */
export const INK_TO_EXCITEMENT = 0.6;
/** Stage-0 connection glow lift at full excitement (the wiring warms under the ink). */
export const DRAW_GLOW = 0.15;

/** The scene's one piece of ambient mutable state: how recently/heavily the
 *  visitor has been inking. Bumped by setInputPixels diffs, decays fast. */
export class AmbientState {
  excitement = 0;

  bump(amount: number): void {
    this.excitement = Math.min(1, this.excitement + Math.max(0, amount));
  }

  decay(dt: number): void {
    this.excitement *= Math.exp(-dt / EXCITEMENT_TAU_S);
  }
}

export interface InkDelta {
  /** Sum of positive per-pixel deltas. */
  total: number;
  /** Pixels whose gain crossed INK_THRESHOLD, for spark spawn bias. */
  indices: number[];
}

/** Positive-only pixel diff. Clearing the pad (deltas ≤ 0) yields zero, so
 *  clearing never excites the network. */
export function inkDelta(prev: Float32Array, next: Float32Array): InkDelta {
  let total = 0;
  const indices: number[] = [];
  for (let i = 0; i < next.length; i++) {
    const d = next[i] - prev[i];
    if (d <= 0) continue;
    total += d;
    if (d >= INK_THRESHOLD) indices.push(i);
  }
  return { total, indices };
}

// ── Ambient sparks ──────────────────────────────────────────────────────
export const SPARK_POOL_SIZE = 48;
export const SPARK_MIN_DUR_S = 0.45;
export const SPARK_MAX_DUR_S = 0.8;
/** Target concurrent sparks: aimless idle static → stirred while drawing. */
export const IDLE_CONCURRENCY = 2.5;
export const EXCITED_CONCURRENCY = 9;
/** At full excitement, this fraction of spawns hugs recently-inked stage-0 edges. */
export const INK_BIAS = 0.85;
/** How many recently-inked pixels are remembered for the bias. */
export const INK_MEMORY = 64;

export interface SparkSpawn {
  slot: number;
  stage: number;
  edge: number;
  duration: number;
  /** Display strength 0..1 — always below pulse-comet brightness. */
  magnitude: number;
}

/**
 * Pure spawn logic for AmbientSparks. Behavioral randomness (where sparks
 * go) flows through the injected rng so tests are deterministic; a
 * Poisson-style accumulator (rate × dt) converts target concurrency into
 * spawn events. The duck envelope multiplies the rate, so ambience thins
 * out the moment a fire starts.
 */
export class SparkScheduler {
  private readonly busyUntil: Float64Array;
  /** input pixel → stage-0 edge indices fanning out of it */
  private readonly pixelEdges = new Map<number, number[]>();
  private readonly inked: number[] = [];
  private acc = 0;

  constructor(
    private readonly edgeCounts: number[],
    stage0From: Uint16Array,
    private readonly rng: () => number = Math.random,
    private readonly poolSize: number = SPARK_POOL_SIZE
  ) {
    this.busyUntil = new Float64Array(poolSize).fill(-1e9);
    for (let e = 0; e < stage0From.length; e++) {
      const pixel = stage0From[e];
      let list = this.pixelEdges.get(pixel);
      if (!list) this.pixelEdges.set(pixel, (list = []));
      list.push(e);
    }
  }

  noteInk(pixelIndices: number[]): void {
    for (const p of pixelIndices) {
      this.inked.push(p);
      if (this.inked.length > INK_MEMORY) this.inked.shift();
    }
  }

  clearInk(): void {
    this.inked.length = 0;
  }

  update(now: number, dt: number, excitement: number, duck: number): SparkSpawn[] {
    const concurrency =
      IDLE_CONCURRENCY + (EXCITED_CONCURRENCY - IDLE_CONCURRENCY) * excitement;
    const avgDur = (SPARK_MIN_DUR_S + SPARK_MAX_DUR_S) / 2;
    this.acc += (concurrency / avgDur) * duck * dt;
    const spawns: SparkSpawn[] = [];
    while (this.acc >= 1) {
      this.acc -= 1;
      const slot = this.freeSlot(now);
      if (slot === -1) break;
      const choice = this.choose(excitement);
      if (!choice) break;
      const duration =
        SPARK_MIN_DUR_S + (SPARK_MAX_DUR_S - SPARK_MIN_DUR_S) * this.rng();
      this.busyUntil[slot] = now + duration;
      spawns.push({
        slot,
        ...choice,
        duration,
        magnitude: 0.35 + 0.45 * this.rng(),
      });
    }
    return spawns;
  }

  private freeSlot(now: number): number {
    for (let s = 0; s < this.poolSize; s++) if (this.busyUntil[s] <= now) return s;
    return -1;
  }

  private choose(excitement: number): { stage: number; edge: number } | null {
    // While the visitor inks, sparks concentrate where the ink is landing
    if (this.inked.length > 0 && this.rng() < INK_BIAS * excitement) {
      const pixel = this.inked[Math.floor(this.rng() * this.inked.length)];
      const edges = this.pixelEdges.get(pixel);
      if (edges && edges.length > 0) {
        return { stage: 0, edge: edges[Math.floor(this.rng() * edges.length)] };
      }
    }
    const total = this.edgeCounts.reduce((a, b) => a + b, 0);
    if (total === 0) return null;
    let pick = Math.floor(this.rng() * total);
    for (let stage = 0; stage < this.edgeCounts.length; stage++) {
      if (pick < this.edgeCounts[stage]) return { stage, edge: pick };
      pick -= this.edgeCounts[stage];
    }
    return null;
  }
}
