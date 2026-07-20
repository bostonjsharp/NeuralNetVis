# Staged Forward Pass — Design

**Date:** 2026-07-20
**Status:** Approved

## Problem

Inference currently runs instantly at fire time (`forwardPass` in `src/nn/inference.ts`), and the
scene replays the precomputed activations as a cinematic. Two consequences:

1. The output digit glyphs are updated with a hardcoded `Math.max(outputReveal, 0.35)` reveal floor
   (`SceneManager.ts`), so the winning digit visibly glows the moment `fire()` is called — seconds
   before the pulse wave reaches the output layer. The answer leaks.
2. The app "knows" the answer (probs, argmax, HUD summary) before the animation has earned it, which
   requires wall-clock timers in `App.tsx` that duplicate timeline knowledge to fake synchronization.

## Goal

The network genuinely computes layer by layer, in sync with the animation: each layer's matmul
executes at the moment its incoming pulse wave lands. Nothing anywhere in the app — scene, HUD,
state machine — knows a value before the wavefront reaches it. Numerical results are identical to
the current one-shot pass.

Granularity is **per-layer** (one wave landing = one computation step). Per-neuron staging was
considered and rejected: visually indistinguishable, meaningfully more complex.

## Design

### 1. New pure module: `src/nn/stagedPass.ts`

```ts
interface StagedPass {
  readonly net: Net;
  /** Input first; grows by one layer per step(). */
  readonly activations: Float32Array[];
  readonly done: boolean;
  /** Non-null only after the final step: logits, probs, argmax. */
  readonly result: ForwardResult | null;
  /** Compute the next layer (matmul + ReLU; logits + softmax/argmax on the last). */
  step(): void;
}

function createStagedPass(net: Net, input: Float32Array): StagedPass;
```

- Calling `step()` when `done` throws.
- `forwardPass(net, input)` is reimplemented as `createStagedPass` + `while (!done) step()`, so the
  existing `inference.test.ts` suite guarantees staged ≡ one-shot with a single source of truth for
  the math.

### 2. Scene drives the computation clock (`SceneManager.ts`)

`SceneApi.fire(result: ForwardResult)` becomes:

```ts
fire(pass: StagedPass, onResult: (result: ForwardResult) => void): void;
```

Per-frame, **before** `updateNeurons`, the scene advances the pass against its own rAF clock:

- **At fire (t = 0):** input activations are known → load stage 0's pulse signals, set the fire
  timestamp. Zero every other stage's signal range first — GPU attribute buffers persist across
  fires, and stale signals from the previous digit must not fly on stages whose waves haven't
  departed.
- **At `layerPop[l]` (wave l lands):** `step()` computes layer l+1; that layer's `normalized`
  display-brightness array is built then (output layer normalizes probs, hidden layers normalize
  activations, as today). If it is the last layer: set the winner-flare target and invoke
  `onResult(pass.result)`.
- **At `stageStart[l+1]` (next wave departs):** load stage l+1's pulse signals from the
  just-computed source activations. Ordering is safe by construction: `layerPop[l] <
  stageStart[l+1]` in every timeline `makeFireTimeline` can produce.

A pure helper in `fire.ts`:

```ts
/** How many destination layers should have been computed by time t. */
function stepsDue(fire: FireTimeline, tSinceFire: number): number;
```

The scene loops `while (stepsTaken < stepsDue(...)) step-and-apply`, so a stalled tab (rAF gap)
catches up in order on the next frame rather than skipping layers.

`updateNeurons` treats a not-yet-computed layer as unlit (its `normalized` entry doesn't exist yet;
`neuronReveal` is already 0 before `layerPop[l]`, so this is belt-and-suspenders).

The glyph reveal floor `Math.max(outputReveal, fireResult ? 0.35 : 0.2)` is **deleted**. Glyphs
receive `null` probs until the output layer computes; after completion the result persists exactly
as today.

### 3. PulseSystem API (`PulseSystem.ts`)

`fire(activations, nowSeconds)` splits into:

```ts
/** Zero all signals and stamp the fire time. */
beginFire(nowSeconds: number): void;
/** Write one stage's per-edge signals (source activation × weight, normalized). */
loadStage(stage: number, sourceActivations: Float32Array): void;
```

Signal math is unchanged — per-edge `source[from] × weight`, normalized by layer max and clamped.

### 4. App and state machine (`App.tsx`, `state.ts`)

- `runInference` no longer calls `forwardPass`. It creates the staged pass, calls
  `scene.fire(pass, onResult)`, and dispatches `{ type: "fire" }`.
- The `"fire"` event **loses its summary payload**: it only performs the mode transition
  (draw/result → infer, attract stays attract) and clears the previous verdict (`current: null`).
- New event `{ type: "resultReady", summary: InferenceSummary }`, dispatched from `onResult`, sets
  `current`. It applies in `infer` and `attract` modes and is ignored elsewhere (a morph or reset
  that raced the callback wins).
- `showResultLater` and `barsTimer` are **deleted**. `onResult` does what they did: sets
  `displayed`, and consumes `prevSummaryRef` for the cross-brain comparison verdict. The HUD bars
  land exactly when the output layer computes — no wall-clock mirror of the timeline.
- `prevSummaryRef` capture moves to result time; the "new pixels invalidate comparison" check stays
  at fire time in `runInference` (that fact is known immediately).
- The `cinematicDone` unlock timer (`cinematicTimer`, `timelineFor(...).total`) is unchanged — it
  only unlocks input, and small drift there is harmless.
- `lastInputRef` / attract-loop re-fire behavior is unchanged; `runInference`'s return value (the
  `ForwardResult`) no longer exists — no caller uses it today beyond ignoring it.

### 5. Cancellation

- **New fire mid-flight** (attract interval, quick re-draw): `fire()` replaces the active pass;
  the superseded pass's `onResult` is never invoked.
- **Brain swap mid-flight** (`setNet`): the active pass is cancelled together with `fireResult`,
  as the current code already clears a stale result. The existing shape guard (a pass built for a
  different topology than the built scene must never animate) is kept, checking the pass's net
  shape against the built layout.
- **Dispose:** nothing is scheduled outside the rAF loop, so disposal needs no new cleanup.
- **Persistence:** a completed result stays lit indefinitely (unchanged). An in-flight pass shows
  nothing downstream of the wavefront; output digits idle dim until the answer is earned.

## Testing

- `src/nn/stagedPass.test.ts`: stepping to completion matches `forwardPass` on the real bundled
  weights (property: identical activations, probs, argmax); `done`/`result` semantics; stepping
  past `done` throws.
- `fire.test.ts`: `stepsDue` — 0 before the first landing, increments exactly at each `layerPop`,
  equals stage count after the last, for 1/2/3-stage timelines.
- `state.test.ts`: `"fire"` without summary clears `current`; `"resultReady"` sets it in
  infer/attract and is ignored in morph/draw.
- `smoke.test.tsx`: `SceneApi` mock updated to the new `fire(pass, onResult)` signature.

## Out of scope

- Per-neuron staged computation (rejected: invisible benefit).
- Time-slicing matmuls across frames during wave flight (rejected: invisible benefit, frame-budget
  bookkeeping).
- Live inference while drawing (different feature; not requested here).
