/**
 * Timeline for one forward-pass cinematic. Inference itself is instant —
 * these curves replay the precomputed activations slowly enough to read
 * from across a room. All times are seconds since the fire started.
 * Pure math: consumed by the render loop and mirrored into pulse-shader
 * uniforms, and unit-testable without WebGL.
 *
 * Timelines are built per network shape (1..3 pulse stages); the 3-stage
 * result is exactly the hand-tuned classic rhythm.
 */

export interface FireTimeline {
  /** Pulse waves leaving each source layer (input first). */
  stageStart: number[];
  stageDur: number[];
  /** When each destination layer's activations reveal (last = output). */
  layerPop: number[];
  winnerFlare: number;
  total: number;
}

/** Hand-tuned wave durations: the input wave is the long establishing shot,
 *  deeper (smaller) stages tighten up. */
const STAGE_DUR = [1.1, 0.85, 0.7];
/** A layer pops slightly before its wave fully lands… */
const POP_LEAD = 0.1;
/** …and the next wave leaves a beat after the layer lights. */
const NEXT_WAVE_PAUSE = 0.25;

export function makeFireTimeline(stageCount: number): FireTimeline {
  if (stageCount < 1 || stageCount > STAGE_DUR.length) {
    throw new Error(`fire timeline supports 1..${STAGE_DUR.length} stages, got ${stageCount}`);
  }
  const stageStart: number[] = [];
  const stageDur: number[] = [];
  const layerPop: number[] = [];
  let t = 0.3;
  for (let s = 0; s < stageCount; s++) {
    stageStart.push(t);
    stageDur.push(STAGE_DUR[s]);
    layerPop.push(t + STAGE_DUR[s] - POP_LEAD);
    t = layerPop[s] + NEXT_WAVE_PAUSE;
  }
  const winnerFlare = layerPop[stageCount - 1] + 0.35;
  return { stageStart, stageDur, layerPop, winnerFlare, total: winnerFlare + 0.9 };
}

/** The classic 3-stage timeline — the app's default until a brain swaps. */
export const FIRE = makeFireTimeline(3);

export const FIRE_TOTAL_MS = FIRE.total * 1000;

export function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

/** Input plane brightness: idle glow → quick bloom-up with a little overshoot. */
export function inputRamp(t: number): number {
  if (t < 0) return 0.9;
  const rise = smoothstep(0, 0.3, t);
  const overshoot = 0.12 * Math.sin(Math.PI * clamp01(t / 0.6)) * rise;
  return 0.55 + 0.35 * rise + overshoot;
}

/** Per-neuron reveal stagger cascades down the column. */
export function neuronRevealStart(
  fire: FireTimeline,
  layer: number,
  index: number,
  count: number
): number {
  return fire.layerPop[layer] + (index / Math.max(1, count - 1)) * 0.18;
}

/** 0..1 how much of a neuron's activation is shown at time t. */
export function neuronReveal(
  fire: FireTimeline,
  t: number,
  layer: number,
  index: number,
  count: number
): number {
  return smoothstep(0, 0.25, t - neuronRevealStart(fire, layer, index, count));
}

/** Scale punch when a neuron pops on: brief bump above 1, settling back. */
export function neuronPop(
  fire: FireTimeline,
  t: number,
  layer: number,
  index: number,
  count: number
): number {
  const u = (t - neuronRevealStart(fire, layer, index, count)) / 0.55;
  if (u <= 0 || u >= 1) return 1;
  return 1 + 0.45 * Math.sin(Math.PI * u) * (1 - u);
}

/** Winner celebration envelope: fast rise, slow fall. */
export function winnerFlare(fire: FireTimeline, t: number): number {
  const u = (t - fire.winnerFlare) / 0.95;
  if (u <= 0 || u >= 1) return 0;
  return smoothstep(0, 0.16, u) * (1 - smoothstep(0.16, 1, u));
}

/** Connection glow lift per stage while its pulse wave is in flight. */
export function stageGlow(fire: FireTimeline, t: number, stage: number): number {
  const start = fire.stageStart[stage];
  const end = start + fire.stageDur[stage];
  return smoothstep(start - 0.15, start + 0.15, t) * (1 - smoothstep(end, end + 0.4, t));
}
