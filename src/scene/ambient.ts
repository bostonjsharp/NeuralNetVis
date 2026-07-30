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
