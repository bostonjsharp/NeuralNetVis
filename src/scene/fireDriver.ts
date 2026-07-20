import type { ForwardResult } from "../nn/inference";
import type { StagedPass } from "../nn/stagedPass";
import type { FireTimeline } from "./fire";
import { stepsDue } from "./fire";

export interface FireDriverHooks {
  /** Called once per computed destination layer, in order. `layerIndex` is
   *  0-based over destination layers (hidden…, output — the input has no
   *  slot here, it's already on screen at t=0). `sourceActivations` are the
   *  layer's raw values, handed back so the caller can seed the next pulse
   *  stage; `normalizedValues` are the same values scaled for display. */
  onLayer(layerIndex: number, normalizedValues: Float32Array, sourceActivations: Float32Array): void;
  /** Called exactly once, at the final layer. */
  onComplete(result: ForwardResult): void;
}

export interface FireDriver {
  advance(tSinceFire: number): void;
}

/**
 * Steps a StagedPass forward as a FireTimeline's waves land, one layer per
 * landing, and normalizes each layer for display brightness. No three.js —
 * this is the pure advance/delivery core SceneManager wraps with its own
 * WebGL-side bookkeeping (pulse loading, flare target, callback delivery).
 */
export function createFireDriver(
  fire: FireTimeline,
  pass: StagedPass,
  layerCount: number,
  hooks: FireDriverHooks
): FireDriver {
  let stepsTaken = 0;
  let done = false;

  return {
    // A stalled tab (rAF gap) catches up in order here rather than skipping
    // layers past a due count.
    advance(tSinceFire: number): void {
      if (done) return;
      const due = stepsDue(fire, tSinceFire);
      while (!done && stepsTaken < due) {
        pass.step();
        stepsTaken++;
        const layerIdx = stepsTaken; // activations[layerIdx] was just computed
        const isLast = layerIdx === layerCount;
        // Normalize for display brightness: output follows probabilities,
        // hidden layers follow activations (each scaled by its layer max).
        const values = isLast ? pass.result!.probs : pass.activations[layerIdx];
        let max = 0;
        for (let i = 0; i < values.length; i++) max = Math.max(max, values[i]);
        const norm = max > 0 ? 1 / max : 0;
        const normalized = Float32Array.from(values, (v) => v * norm);
        hooks.onLayer(layerIdx - 1, normalized, pass.activations[layerIdx]);
        if (isLast) {
          // Mark done before the hook runs: onComplete may synchronously
          // re-enter advance() (directly, or via a callback that starts a
          // new fire) — this driver must be inert by then, permanently,
          // even against a stale larger `due` on that call.
          done = true;
          hooks.onComplete(pass.result!);
          return;
        }
      }
    },
  };
}
