/**
 * Timeline for one forward-pass cinematic. Inference itself is instant —
 * these curves replay the precomputed activations slowly enough to read
 * from across a room. All times are seconds since the fire started.
 * Pure math: consumed by the render loop and mirrored into pulse-shader
 * uniforms, and unit-testable without WebGL.
 */

export const FIRE = {
  /** Pulse waves leaving input, h1, h2. */
  stageStart: [0.3, 1.55, 2.55] as const,
  stageDur: [1.1, 0.85, 0.7] as const,
  /** When each destination layer's activations reveal (h1, h2, output). */
  layerPop: [1.3, 2.3, 3.15] as const,
  winnerFlare: 3.5,
  total: 4.4,
};

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
export function neuronRevealStart(layer: number, index: number, count: number): number {
  return FIRE.layerPop[layer] + (index / Math.max(1, count - 1)) * 0.18;
}

/** 0..1 how much of a neuron's activation is shown at time t. */
export function neuronReveal(t: number, layer: number, index: number, count: number): number {
  return smoothstep(0, 0.25, t - neuronRevealStart(layer, index, count));
}

/** Scale punch when a neuron pops on: brief bump above 1, settling back. */
export function neuronPop(t: number, layer: number, index: number, count: number): number {
  const u = (t - neuronRevealStart(layer, index, count)) / 0.55;
  if (u <= 0 || u >= 1) return 1;
  return 1 + 0.45 * Math.sin(Math.PI * u) * (1 - u);
}

/** Winner celebration envelope: fast rise, slow fall. */
export function winnerFlare(t: number): number {
  const u = (t - FIRE.winnerFlare) / 0.95;
  if (u <= 0 || u >= 1) return 0;
  return smoothstep(0, 0.16, u) * (1 - smoothstep(0.16, 1, u));
}

/** Connection glow lift per stage while its pulse wave is in flight. */
export function stageGlow(t: number, stage: number): number {
  const start = FIRE.stageStart[stage];
  const end = start + FIRE.stageDur[stage];
  return smoothstep(start - 0.15, start + 0.15, t) * (1 - smoothstep(end, end + 0.4, t));
}
