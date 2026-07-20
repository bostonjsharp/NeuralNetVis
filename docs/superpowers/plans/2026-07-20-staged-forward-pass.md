# Staged Forward Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The neural net computes each layer's matmul at the moment its incoming pulse wave lands in the cinematic, so no part of the app (scene, HUD, state machine) knows a value before the wavefront reaches it.

**Architecture:** A new pure `StagedPass` stepper in `src/nn/` computes one layer per `step()`. `SceneManager` — owner of the animation clock — calls `step()` when the timeline says a wave has landed (`stepsDue` helper in `fire.ts`), loads each pulse stage's signals only once its source layer exists, and invokes a completion callback when the output layer computes. `App.tsx` replaces its wall-clock HUD timer with that callback; the state machine's `"fire"` event loses its summary and a new `"resultReady"` event carries it.

**Tech Stack:** TypeScript, React 19, Three.js 0.185, Vitest 4. Spec: `docs/superpowers/specs/2026-07-20-staged-forward-pass-design.md`.

## Global Constraints

- NEVER run `scripts/train-mnist.mjs` (or `npm run train`) — it executes on import and overwrites the committed weights.
- All commands run from repo root `D:\dev\bost\footronIdeas\NeuralNetVis`.
- Test: `npm test` (vitest run). Typecheck: `npm run typecheck`. Lint: `npm run lint`. All three must pass at the end of every task.
- Numerical results must be identical to the current one-shot `forwardPass` — the golden-vector tests in `src/nn/inference.test.ts` must keep passing unmodified.
- Commit messages: end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` (project convention).

---

### Task 1: StagedPass stepper (`src/nn/stagedPass.ts`)

**Files:**
- Create: `src/nn/stagedPass.ts`
- Create: `src/nn/stagedPass.test.ts`
- Modify: `src/nn/inference.ts` (reimplement `forwardPass` via the stepper; `softmax` moves to `stagedPass.ts` and is re-exported)

**Interfaces:**
- Consumes: `Net` from `src/nn/weights.ts` (`{ shape: number[]; layers: { W: Float32Array; b: Float32Array }[] }`), `ForwardResult` from `src/nn/inference.ts` (type-only import — no runtime cycle).
- Produces (used by Tasks 4 and 6):
  ```ts
  interface StagedPass {
    readonly net: Net;
    readonly activations: Float32Array[]; // input first; grows by one per step()
    readonly done: boolean;
    readonly result: ForwardResult | null; // non-null only after the final step
    step(): void; // throws if done
  }
  function createStagedPass(net: Net, input: Float32Array): StagedPass;
  ```

- [ ] **Step 1: Write the failing test**

Create `src/nn/stagedPass.test.ts`. The equivalence test compares against `forwardPass` **while it is still the independent implementation** — after Step 5's refactor it becomes a regression guard, and the hand-computed goldens in `inference.test.ts` remain the independent anchor.

```ts
import { describe, expect, it } from "vitest";
import { forwardPass } from "./inference";
import { createStagedPass } from "./stagedPass";
import type { Net } from "./weights";

/** Same hand-computed 2→3→2 golden network as inference.test.ts. */
const toyNet: Net = {
  shape: [2, 3, 2],
  layers: [
    {
      W: Float32Array.from([1, -1, 0.5, 0.5, -2, 1]),
      b: Float32Array.from([0, 0.5, 1]),
    },
    {
      W: Float32Array.from([1, 0, -1, 0, 1, 1]),
      b: Float32Array.from([0.5, -0.5]),
    },
  ],
};
const input = Float32Array.from([1, 2]);

describe("createStagedPass", () => {
  it("starts with only the input, not done, no result", () => {
    const pass = createStagedPass(toyNet, input);
    expect(pass.activations).toHaveLength(1);
    expect(Array.from(pass.activations[0])).toEqual([1, 2]);
    expect(pass.done).toBe(false);
    expect(pass.result).toBeNull();
  });

  it("computes exactly one layer per step", () => {
    const pass = createStagedPass(toyNet, input);
    pass.step();
    expect(pass.activations).toHaveLength(2);
    expect(Array.from(pass.activations[1])).toEqual([0, 2, 1]); // post-ReLU golden
    expect(pass.done).toBe(false);
    expect(pass.result).toBeNull();
    pass.step();
    expect(pass.activations).toHaveLength(3);
    expect(pass.done).toBe(true);
  });

  it("produces the full result exactly at the final step", () => {
    const pass = createStagedPass(toyNet, input);
    pass.step();
    pass.step();
    const result = pass.result!;
    expect(result.logits[0]).toBeCloseTo(-0.5, 5);
    expect(result.logits[1]).toBeCloseTo(2.5, 5);
    expect(result.probs[1]).toBeCloseTo(0.95257, 4);
    expect(result.argmax).toBe(1);
    expect(result.activations).toBe(pass.activations);
  });

  it("matches forwardPass exactly when stepped to completion", () => {
    const pass = createStagedPass(toyNet, input);
    while (!pass.done) pass.step();
    const oneShot = forwardPass(toyNet, input);
    expect(pass.activations.map((a) => Array.from(a))).toEqual(
      oneShot.activations.map((a) => Array.from(a))
    );
    expect(Array.from(pass.result!.probs)).toEqual(Array.from(oneShot.probs));
    expect(pass.result!.argmax).toBe(oneShot.argmax);
  });

  it("throws when stepped past done", () => {
    const pass = createStagedPass(toyNet, input);
    pass.step();
    pass.step();
    expect(() => pass.step()).toThrow(/complete/);
  });

  it("rejects wrongly sized input", () => {
    expect(() => createStagedPass(toyNet, new Float32Array(3))).toThrow(/expected 2/);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/nn/stagedPass.test.ts`
Expected: FAIL — `Cannot find module './stagedPass'` (or equivalent resolve error).

- [ ] **Step 3: Write the implementation**

Create `src/nn/stagedPass.ts`. `softmax` lives here now (moved verbatim from `inference.ts`) so `inference.ts` can depend on this module with only a type flowing the other way.

```ts
import type { ForwardResult } from "./inference";
import type { Net } from "./weights";

/**
 * A forward pass that computes one layer per step() — the scene calls step()
 * when a pulse wave lands, so the app genuinely does not know deeper layers'
 * values (or the answer) until the animation reaches them. Stepping to
 * completion is exactly forwardPass; inference.ts is implemented on top.
 */
export interface StagedPass {
  readonly net: Net;
  /** Input first; grows by one layer per step(). */
  readonly activations: Float32Array[];
  readonly done: boolean;
  /** Non-null only after the final step: logits, probs, argmax. */
  readonly result: ForwardResult | null;
  /** Compute the next layer (matmul + ReLU; logits + softmax/argmax on the last). */
  step(): void;
}

export function createStagedPass(net: Net, input: Float32Array): StagedPass {
  const { shape, layers } = net;
  if (input.length !== shape[0]) {
    throw new Error(`input has ${input.length} values, expected ${shape[0]}`);
  }
  const activations: Float32Array[] = [Float32Array.from(input)];
  const last = layers.length - 1;
  let result: ForwardResult | null = null;
  return {
    net,
    activations,
    get done() {
      return result !== null;
    },
    get result() {
      return result;
    },
    step() {
      if (result !== null) throw new Error("staged pass already complete");
      const l = activations.length - 1;
      const { W, b } = layers[l];
      const aIn = activations[l];
      const nIn = shape[l];
      const nOut = shape[l + 1];
      const aOut = new Float32Array(nOut);
      for (let o = 0; o < nOut; o++) {
        let sum = b[o];
        const row = o * nIn;
        for (let i = 0; i < nIn; i++) sum += W[row + i] * aIn[i];
        aOut[o] = l < last ? Math.max(0, sum) : sum;
      }
      activations.push(aOut);
      if (l === last) {
        const logits = aOut;
        const probs = softmax(logits);
        let argmax = 0;
        for (let o = 1; o < probs.length; o++) if (probs[o] > probs[argmax]) argmax = o;
        result = { activations, logits, probs, argmax };
      }
    },
  };
}

export function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/nn/stagedPass.test.ts`
Expected: PASS (6 tests). `forwardPass` is still the old independent implementation at this point — the equivalence test is meaningful.

- [ ] **Step 5: Reimplement forwardPass on the stepper**

Replace the **entire contents** of `src/nn/inference.ts` with:

```ts
import { createStagedPass, softmax } from "./stagedPass";
import type { Net } from "./weights";

export interface ForwardResult {
  /** One entry per network layer, input first. Hidden layers are post-ReLU. */
  activations: Float32Array[];
  /** Raw output-layer scores (pre-softmax). Same array as activations at the end. */
  logits: Float32Array;
  probs: Float32Array;
  argmax: number;
}

/**
 * Full forward pass in one call — a StagedPass stepped to completion, so the
 * math has a single source of truth (see stagedPass.ts). The scene uses the
 * stepper directly; this convenience wrapper serves tests and any caller
 * that wants the answer immediately.
 */
export function forwardPass(net: Net, input: Float32Array): ForwardResult {
  const pass = createStagedPass(net, input);
  while (!pass.done) pass.step();
  return pass.result!;
}

export { softmax };
```

- [ ] **Step 6: Run the full nn suite to verify the goldens still pass**

Run: `npx vitest run src/nn`
Expected: PASS — including all pre-existing `inference.test.ts` golden-vector tests, unmodified.

- [ ] **Step 7: Commit**

```bash
git add src/nn/stagedPass.ts src/nn/stagedPass.test.ts src/nn/inference.ts
git commit -m "feat: StagedPass stepper computing one layer per step

forwardPass is now a StagedPass stepped to completion — one source of
truth for the math, guarded by the existing golden-vector tests.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `stepsDue` timeline helper (`src/scene/fire.ts`)

**Files:**
- Modify: `src/scene/fire.ts` (append one function)
- Modify: `src/scene/fire.test.ts` (append one describe block)

**Interfaces:**
- Consumes: `FireTimeline` (already in `fire.ts`).
- Produces (used by Task 4): `function stepsDue(fire: FireTimeline, tSinceFire: number): number` — how many destination layers should have been computed by `tSinceFire` seconds after fire.

- [ ] **Step 1: Write the failing test**

Append to `src/scene/fire.test.ts` (it already imports from `./fire`; extend the import list with `stepsDue` and `makeFireTimeline` if not present):

```ts
describe("stepsDue", () => {
  const fire = makeFireTimeline(3);

  it("is 0 before the first wave lands", () => {
    expect(stepsDue(fire, 0)).toBe(0);
    expect(stepsDue(fire, fire.layerPop[0] - 0.01)).toBe(0);
  });

  it("increments exactly at each layerPop", () => {
    expect(stepsDue(fire, fire.layerPop[0])).toBe(1);
    expect(stepsDue(fire, fire.layerPop[1] - 0.01)).toBe(1);
    expect(stepsDue(fire, fire.layerPop[1])).toBe(2);
    expect(stepsDue(fire, fire.layerPop[2])).toBe(3);
  });

  it("stays at the stage count forever after", () => {
    expect(stepsDue(fire, 1000)).toBe(3);
  });

  it("handles negative time (before fire) and single-stage timelines", () => {
    expect(stepsDue(fire, -5)).toBe(0);
    const single = makeFireTimeline(1);
    expect(stepsDue(single, single.layerPop[0])).toBe(1);
    expect(stepsDue(single, 1000)).toBe(1);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/scene/fire.test.ts`
Expected: FAIL — `stepsDue` is not exported.

- [ ] **Step 3: Write the implementation**

Append to `src/scene/fire.ts`:

```ts
/** How many destination layers should have been computed by time t —
 *  one per wave landing. The scene steps its StagedPass up to this. */
export function stepsDue(fire: FireTimeline, tSinceFire: number): number {
  let due = 0;
  for (const pop of fire.layerPop) if (tSinceFire >= pop) due++;
  return due;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/scene/fire.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scene/fire.ts src/scene/fire.test.ts
git commit -m "feat: stepsDue — how many layers the timeline has earned at time t

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: PulseSystem staged loading (`src/scene/PulseSystem.ts`)

**Files:**
- Modify: `src/scene/PulseSystem.ts` (replace `fire()` with `beginFire()` + `loadStage()`; keep `fire()` as a thin wrapper so `SceneManager` still compiles until Task 4 removes the call)
- Create: `src/scene/PulseSystem.test.ts`

**Interfaces:**
- Consumes: `NetworkLayout` / `maxAbsWeight` from `./NetworkLayout`, `FireTimeline` from `./fire`.
- Produces (used by Task 4):
  ```ts
  beginFire(nowSeconds: number): void;               // zero ALL signals, stamp fire time
  loadStage(stage: number, source: Float32Array): void; // write one stage's per-edge signals
  ```
  Signal math is unchanged: `source[from] × weight`, source normalized by its layer max, weight by the stage's max |weight|, clamped to [-1, 1].

- [ ] **Step 1: Write the failing test**

Create `src/scene/PulseSystem.test.ts`. Three.js `BufferGeometry`/`ShaderMaterial`/`Points` are pure JS until a renderer touches them, so this runs headless. The net helper mirrors `NetworkLayout.test.ts` (input must be 784 — the layout builds a 28×28 grid).

```ts
import { describe, expect, it } from "vitest";
import type { Net } from "../nn/weights";
import { makeFireTimeline } from "./fire";
import { buildLayout } from "./NetworkLayout";
import { PulseSystem } from "./PulseSystem";

/** Deterministic pseudo-random net (same recipe as NetworkLayout.test.ts). */
const makeNet = (shape: number[]): Net => ({
  shape,
  layers: Array.from({ length: shape.length - 1 }, (_, l) => {
    const [nIn, nOut] = [shape[l], shape[l + 1]];
    const W = new Float32Array(nIn * nOut);
    for (let i = 0; i < W.length; i++) W[i] = Math.sin(i * 12.9898 + l) * 0.5;
    return { W, b: new Float32Array(nOut) };
  }),
});

const PARTICLES_PER_EDGE = 2; // mirrors the constant in PulseSystem.ts

function build() {
  const layout = buildLayout(makeNet([784, 4, 10]));
  const pulses = new PulseSystem(layout, makeFireTimeline(layout.edges.length));
  const signals = () =>
    pulses.points.geometry.getAttribute("aSignal").array as Float32Array;
  const stage0End = layout.edges[0].count * PARTICLES_PER_EDGE;
  return { layout, pulses, signals, stage0End };
}

const ones = (n: number) => new Float32Array(n).fill(1);

describe("PulseSystem staged loading", () => {
  it("beginFire zeroes every stage's signals", () => {
    const { pulses, signals } = build();
    pulses.loadStage(0, ones(784));
    pulses.loadStage(1, ones(4));
    expect(signals().some((s) => s !== 0)).toBe(true);
    pulses.beginFire(1.5);
    expect(signals().every((s) => s === 0)).toBe(true);
  });

  it("loadStage(0) writes only stage 0's particle range", () => {
    const { pulses, signals, stage0End } = build();
    pulses.beginFire(0);
    pulses.loadStage(0, ones(784));
    const arr = signals();
    expect(arr.slice(0, stage0End).some((s) => s !== 0)).toBe(true);
    expect(arr.slice(stage0End).every((s) => s === 0)).toBe(true);
  });

  it("loadStage(1) writes only stage 1's particle range", () => {
    const { pulses, signals, stage0End } = build();
    pulses.beginFire(0);
    pulses.loadStage(1, ones(4));
    const arr = signals();
    expect(arr.slice(0, stage0End).every((s) => s === 0)).toBe(true);
    expect(arr.slice(stage0End).some((s) => s !== 0)).toBe(true);
  });

  it("writes both particle slots of an edge with the same clamped signal", () => {
    const { layout, pulses, signals, stage0End } = build();
    pulses.beginFire(0);
    pulses.loadStage(1, Float32Array.from([100, 0, 0, 0])); // huge source → clamps
    const arr = signals().slice(stage0End);
    expect(arr.length).toBe(layout.edges[1].count * PARTICLES_PER_EDGE);
    for (let e = 0; e < layout.edges[1].count; e++) {
      expect(arr[e * 2]).toBe(arr[e * 2 + 1]);
      expect(Math.abs(arr[e * 2])).toBeLessThanOrEqual(1);
    }
  });

  it("an all-zero source produces all-zero signals (norm guard)", () => {
    const { pulses, signals, stage0End } = build();
    pulses.beginFire(0);
    pulses.loadStage(1, new Float32Array(4));
    expect(signals().slice(stage0End).every((s) => s === 0)).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/scene/PulseSystem.test.ts`
Expected: FAIL — `pulses.loadStage is not a function`.

- [ ] **Step 3: Write the implementation**

In `src/scene/PulseSystem.ts`, replace the entire `fire(...)` method (currently lines 148–172) with:

```ts
  /**
   * Start a new pass: zero every stage's signals and stamp the fire time.
   * GPU attribute buffers persist across fires — without the wipe, the
   * previous digit's signals would fly on stages whose waves haven't been
   * earned yet. Stages light up via loadStage as their sources compute.
   */
  beginFire(nowSeconds: number): void {
    (this.signalAttr.array as Float32Array).fill(0);
    this.signalAttr.needsUpdate = true;
    this.material.uniforms.uFireTime.value = nowSeconds;
  }

  /**
   * Write one stage's per-edge signals from its source layer's activations
   * (activation × weight — the literal terms of the destination dot
   * products). Hidden-layer sources are normalized by their layer max so
   * brightness always uses the full range.
   */
  loadStage(stage: number, source: Float32Array): void {
    const signals = this.signalAttr.array as Float32Array;
    let p = 0;
    for (let s = 0; s < stage; s++) p += this.edgeCounts[s] * PARTICLES_PER_EDGE;
    const set = this.layout.edges[stage];
    let sourceMax = 0;
    for (let i = 0; i < source.length; i++) sourceMax = Math.max(sourceMax, source[i]);
    const norm = sourceMax > 0 ? 1 / sourceMax : 0;
    for (let e = 0; e < set.count; e++) {
      const signal =
        source[set.from[e]] * norm * (set.weight[e] / this.weightNorm[stage]);
      const clamped = Math.max(-1, Math.min(1, signal));
      for (let slot = 0; slot < PARTICLES_PER_EDGE; slot++, p++) signals[p] = clamped;
    }
    this.signalAttr.needsUpdate = true;
  }

  /** One-shot loading of a completed pass — SceneManager still calls this
   *  until the staged drive lands; removed in the SceneManager task. */
  fire(activations: Float32Array[], nowSeconds: number): void {
    this.beginFire(nowSeconds);
    for (let stage = 0; stage < this.layout.edges.length; stage++) {
      this.loadStage(stage, activations[stage]);
    }
  }
```

- [ ] **Step 4: Run tests and typecheck**

Run: `npx vitest run src/scene/PulseSystem.test.ts` — Expected: PASS (5 tests).
Run: `npm run typecheck` — Expected: clean (SceneManager still compiles against the wrapper).

- [ ] **Step 5: Commit**

```bash
git add src/scene/PulseSystem.ts src/scene/PulseSystem.test.ts
git commit -m "feat: PulseSystem beginFire/loadStage for per-stage signal loading

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: SceneManager drives the staged pass (`src/scene/SceneManager.ts`)

No unit test file — this module is the WebGL boundary (existing convention). Verification is typecheck + the full suite + Task 6's app wiring. Read the whole file before editing; line numbers refer to the current committed state.

**Files:**
- Modify: `src/scene/SceneManager.ts`
- Modify: `src/scene/PulseSystem.ts` (delete the transitional `fire()` wrapper)

**Interfaces:**
- Consumes: `createStagedPass` output type `StagedPass` (Task 1), `stepsDue` (Task 2), `beginFire`/`loadStage` (Task 3).
- Produces (used by Task 6): the changed `SceneApi` method
  ```ts
  fire(pass: StagedPass, onResult: (result: ForwardResult) => void): void;
  ```
  Contract: `onResult` is invoked exactly once, at the frame where the output layer computes (in sync with the output reveal) — or never, if the pass is superseded by a new `fire()`, cancelled by `setNet()`, or rejected by the topology guard.

- [ ] **Step 1: Update imports and the SceneApi interface**

In `src/scene/SceneManager.ts`:

Add `stepsDue` to the `./fire` import list. Add below the `ForwardResult` import:

```ts
import type { StagedPass } from "../nn/stagedPass";
```

Replace the `fire` member of `SceneApi` (currently `fire(result: ForwardResult): void;` with its doc comment) with:

```ts
  /**
   * Start the forward-pass cinematic. The pass is computed layer-by-layer as
   * the pulse waves land; onResult fires when the output layer computes —
   * or never, if a new fire/brain-swap supersedes this pass first.
   */
  fire(pass: StagedPass, onResult: (result: ForwardResult) => void): void;
```

- [ ] **Step 2: Replace the fire-state block**

Replace (currently lines 145–148):

```ts
  // Active cinematic (persists after completion so the last result stays lit)
  let fireStart = -1e9;
  let fireResult: ForwardResult | null = null;
  let normalized: Float32Array[] = [];
```

with:

```ts
  // Active cinematic. The pass computes layer-by-layer as waves land;
  // fireResult exists only once the final layer has computed (and persists
  // after completion so the last result stays lit). normalized[layer] exists
  // only for layers the pass has reached.
  let fireStart = -1e9;
  let firePass: StagedPass | null = null;
  let fireOnResult: ((result: ForwardResult) => void) | null = null;
  let fireResult: ForwardResult | null = null;
  let stepsTaken = 0;
  let normalized: Float32Array[] = [];
```

- [ ] **Step 3: Add the staged-advance function**

Insert directly above `function updateNeurons(...)`:

```ts
  /** Execute the matmuls whose incoming waves have landed. A stalled tab
   *  (rAF gap) catches up in order here rather than skipping layers. */
  function advanceStagedPass(tSinceFire: number): void {
    if (!firePass) return;
    const due = stepsDue(fire, tSinceFire);
    while (firePass && stepsTaken < due) {
      firePass.step();
      stepsTaken++;
      const layerIdx = stepsTaken; // activations[layerIdx] was just computed
      const isLast = layerIdx === layerCounts.length;
      // Normalize for display brightness: output follows probabilities,
      // hidden layers follow activations (each scaled by its layer max).
      const values = isLast ? firePass.result!.probs : firePass.activations[layerIdx];
      let max = 0;
      for (let i = 0; i < values.length; i++) max = Math.max(max, values[i]);
      const norm = max > 0 ? 1 / max : 0;
      normalized[layerIdx - 1] = Float32Array.from(values, (v) => v * norm);
      if (isLast) {
        fireResult = firePass.result;
        flare.setTarget(neurons.positionOf(layerCounts.length - 1, fireResult!.argmax));
        const deliver = fireOnResult;
        firePass = null;
        fireOnResult = null;
        deliver?.(fireResult!);
      } else {
        // The wave carrying these values departs at stageStart[layerIdx],
        // after this landing — its particles stay invisible until then, so
        // loading at compute time leaks nothing.
        pulses.loadStage(layerIdx, firePass.activations[layerIdx]);
      }
    }
  }
```

- [ ] **Step 4: Gate updateNeurons on computed layers**

Inside `updateNeurons`, replace the `if (fireResult) { ... }` block (currently lines 221–233) with — note the new `levels` guard and the added `fireResult &&` in the winner condition:

```ts
        const levels = normalized[layer]; // exists only once this layer computed
        if (levels) {
          const reveal = neuronReveal(fire, tSinceFire, layer, i, count);
          // Perceptual curve: keep weak activations visibly dimmer than
          // strong ones instead of letting bloom crush everything to white.
          const level = Math.pow(levels[i], 1.6);
          brightness += reveal * level * 1.05;
          scale = neuronPop(fire, tSinceFire, layer, i, count) * (1 + 0.22 * level * reveal);
          if (layer === lastLayer && fireResult && i === fireResult.argmax) {
            brightness += 0.4 * flareEnv;
            scale += 0.3 * flareEnv;
            warmth = Math.max(flareEnv, 0.4 * reveal * level);
          }
        }
```

- [ ] **Step 5: Drive the pass and fix the frame's reveal gates**

In `frame()`, insert directly after the morph block (after the closing `}` of `if (morphState) { ... }`, before `rig.update(elapsed, dt);`):

```ts
    advanceStagedPass(tSinceFire);
    const fireActive = firePass !== null || fireResult !== null;
```

Then three edits below it:

1. Input plane (currently `inputPlane.setBrightness(fireResult ? inputRamp(tSinceFire) : 1);`):

```ts
    inputPlane.setBrightness(fireActive ? inputRamp(tSinceFire) : 1);
```

2. Stage glow (inside `connections.setStageGlow(...)`, replace `fireResult ?` with `fireActive ?`):

```ts
    connections.setStageGlow(
      layout.edges.map((_, stage) =>
        fireActive ? (stage === 0 ? 0.5 : 1) * stageGlow(fire, tSinceFire, stage) : 0
      )
    );
```

3. Glyphs — delete the 0.35 leak floor. Replace (currently lines 289–292):

```ts
    const outputReveal = fireResult
      ? neuronReveal(fire, tSinceFire, layerCounts.length - 1, 0, 1)
      : 0;
    glyphs.update(fireResult ? fireResult.probs : null, Math.max(outputReveal, fireResult ? 0.35 : 0.2), fireResult?.argmax ?? 0, flareEnv);
```

with:

```ts
    // Glyphs know nothing until the output layer actually computes — the
    // answer must be earned by the wavefront, never leaked ahead of it.
    const outputReveal = fireResult
      ? neuronReveal(fire, tSinceFire, layerCounts.length - 1, 0, 1)
      : 0;
    glyphs.update(
      fireResult ? fireResult.probs : null,
      fireResult ? Math.max(outputReveal, 0.2) : 0.2,
      fireResult?.argmax ?? 0,
      flareEnv
    );
```

- [ ] **Step 6: Rewrite fire() and extend setNet() cancellation**

Replace the `fire(result) { ... }` method of the returned object with:

```ts
    fire(pass, onResult) {
      // The morph advances on rAF frames while callers schedule re-fires on
      // wall-clock timers — under load a fire can arrive mid-morph. Land the
      // pending swap first so the scene's topology matches the pass.
      if (morphState) {
        if (!morphState.swapped) swapSubsystems(morphState.net, morphState.options);
        const { onDone } = morphState;
        morphState = null;
        onDone();
      }
      // A pass built for a different topology than the built scene must
      // never animate — it would index out of every buffer.
      if (pass.net.shape.length - 1 !== layerCounts.length) return;
      firePass = pass;
      fireOnResult = onResult;
      fireResult = null; // the old verdict is about the old input
      stepsTaken = 0;
      normalized = [];
      fireStart = elapsed;
      inputPlane.setPixels(pass.activations[0]);
      pulses.beginFire(elapsed);
      pulses.loadStage(0, pass.activations[0]); // the input is known at t=0
    },
```

In `setNet(...)`, replace the two cancellation lines:

```ts
      // A lit result about the old brain must not survive the swap
      fireResult = null;
      fireStart = -1e9;
```

with:

```ts
      // A lit result — or in-flight pass — about the old brain must not
      // survive the swap; a superseded pass's onResult never fires.
      firePass = null;
      fireOnResult = null;
      fireResult = null;
      stepsTaken = 0;
      normalized = [];
      fireStart = -1e9;
```

- [ ] **Step 7: Delete the transitional PulseSystem.fire wrapper**

In `src/scene/PulseSystem.ts`, delete the entire `fire(activations, nowSeconds)` method added in Task 3 (the wrapper marked "removed in the SceneManager task").

- [ ] **Step 8: Typecheck and run the scene suite**

Run: `npm run typecheck`
Expected: clean, EXCEPT errors in `src/App.tsx` and `src/smoke.test.tsx` are **expected at this point** (they still call `fire(result)` — Task 6 fixes them). If typecheck reports errors *only* in those two files, proceed. Any error in `src/scene/**` must be fixed now.

Run: `npx vitest run src/scene`
Expected: PASS (fire, PulseSystem, NetworkLayout, adaptiveQuality, contextLossRecovery tests).

- [ ] **Step 9: Commit**

```bash
git add src/scene/SceneManager.ts src/scene/PulseSystem.ts
git commit -m "feat: scene computes the pass layer-by-layer as waves land

fire() now takes a StagedPass + onResult callback; the glyph reveal
floor that leaked the answer at fire time is gone.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: State machine events (`src/app/state.ts`)

**Files:**
- Modify: `src/app/state.ts`
- Modify: `src/app/state.test.ts`

**Interfaces:**
- Produces (used by Task 6):
  ```ts
  | { type: "fire" }                                    // was { type: "fire"; summary }
  | { type: "resultReady"; summary: InferenceSummary }  // new
  ```
  `"fire"` locks input and clears the old verdict; `"resultReady"` records the summary when the output layer actually computes (accepted in `infer`, `result`, and `attract` — `result` covers a wall-clock `cinematicDone` racing ahead of a stalled rAF callback).

- [ ] **Step 1: Update the tests to the new event semantics**

In `src/app/state.test.ts`:

Replace the test `"attract fire stays in attract but records the inference"` with:

```ts
  it("attract fire stays in attract, clearing the old verdict", () => {
    const next = reduce(at("attract", summary), { type: "fire" });
    expect(next).toEqual(at("attract"));
  });
```

Replace `"draw → infer on fire"` with:

```ts
  it("draw → infer on fire, clearing the old verdict until resultReady", () => {
    const next = reduce(at("draw"), { type: "fire" });
    expect(next).toEqual(at("infer"));
  });
```

Replace `"result → infer on refire (amended drawing)"` with:

```ts
  it("result → infer on refire (amended drawing)", () => {
    expect(reduce(at("result", summary), { type: "fire" }).mode).toBe("infer");
  });
```

In `"locks input during infer"` and `"morph locks drawing input"`, change every `{ type: "fire", summary }` to `{ type: "fire" }`.

Add a new describe block at the end of the `"app state machine"` describe:

```ts
  it("resultReady records the verdict when the output layer computes", () => {
    expect(reduce(at("infer"), { type: "resultReady", summary })).toEqual(
      at("infer", summary)
    );
    expect(reduce(at("attract"), { type: "resultReady", summary })).toEqual(
      at("attract", summary)
    );
    // cinematicDone can race ahead of a stalled rAF — the verdict still lands
    expect(reduce(at("result"), { type: "resultReady", summary })).toEqual(
      at("result", summary)
    );
  });

  it("resultReady is ignored where no pass can be in flight", () => {
    const draw = at("draw");
    expect(reduce(draw, { type: "resultReady", summary })).toBe(draw);
    const morph = at("morph", null, "wide");
    expect(reduce(morph, { type: "resultReady", summary })).toBe(morph);
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/app/state.test.ts`
Expected: FAIL — type errors / unknown event type `resultReady`.

- [ ] **Step 3: Update the state machine**

In `src/app/state.ts`:

Replace the `"fire"` line of `AppEvent`:

```ts
  | { type: "fire" }
  | { type: "resultReady"; summary: InferenceSummary }
```

Replace the `case "fire":` block in `reduce`:

```ts
    case "fire":
      // The verdict clears at fire and returns via resultReady — the app
      // genuinely doesn't know the answer until the output layer computes.
      if (state.mode === "attract") {
        return { mode: "attract", current: null, brainId };
      }
      if (state.mode === "draw" || state.mode === "result") {
        return { mode: "infer", current: null, brainId };
      }
      return state; // already inferring or morphing — input locked

    case "resultReady":
      if (state.mode === "infer" || state.mode === "result" || state.mode === "attract") {
        return { mode: state.mode, current: event.summary, brainId };
      }
      return state; // superseded by a morph/reset while the wave was in flight
```

Also update the module doc diagram's `──fire──▶` line comment if desired (optional, no behavior).

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/app/state.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/app/state.ts src/app/state.test.ts
git commit -m "feat: fire locks input without a verdict; resultReady lands it

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: App wiring (`src/App.tsx`) + full verification

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/smoke.test.tsx` (mock gets the new `fire` signature — no behavior change needed, but verify)

**Interfaces:**
- Consumes: `createStagedPass` (Task 1), `SceneApi.fire(pass, onResult)` (Task 4), `"fire"`/`"resultReady"` events (Task 5).
- Produces: user-visible behavior — HUD bars/verdict/comparison land at the exact frame the output layer computes.

- [ ] **Step 1: Update imports**

In `src/App.tsx` replace:

```ts
import { forwardPass, type ForwardResult } from "./nn/inference";
```

with:

```ts
import type { ForwardResult } from "./nn/inference";
import { createStagedPass } from "./nn/stagedPass";
```

- [ ] **Step 2: Delete the wall-clock HUD timer, rewrite runInference**

Delete the `barsTimer` ref declaration (`const barsTimer = useRef(0);`) and the entire `showResultLater` function (currently lines 112–126).

Replace the entire `runInference` function with:

```ts
  const runInference = (
    pixels: Float32Array,
    source: InferenceSummary["source"],
    sampleLabel?: number
  ): void => {
    const brainId = stateRef.current.brainId;
    const pass = createStagedPass(getVariant(brainId).net, pixels);
    // A NEW input invalidates any cross-brain comparison; a re-fire of the
    // exact same pixels array (morph landing, attract repeat) keeps it.
    if (prevPixelsRef.current !== pixels) prevSummaryRef.current = null;
    prevPixelsRef.current = pixels;
    lastInputRef.current = { pixels, source, sampleLabel };
    setDisplayed(null);
    setComparison(null);
    sceneRef.current?.setInputPixels(pixels);
    // The callback fires at the exact frame the output layer computes — the
    // HUD lands in lockstep with the cinematic's output reveal, no timer.
    // A pass superseded by a re-fire or brain swap never calls back.
    sceneRef.current?.fire(pass, (result: ForwardResult) => {
      const mode = stateRef.current.mode;
      // An attract fire's late verdict must not pop into a visitor's fresh
      // draw panel (they woke the wall while the wave was in flight).
      if (mode !== "infer" && mode !== "result" && mode !== "attract") return;
      const summary: InferenceSummary = {
        probs: Array.from(result.probs),
        argmax: result.argmax,
        source,
        sampleLabel,
        brainId,
      };
      // A comparison only makes sense against the same input through a
      // different brain — consumed once, when the verdict lands.
      const against = prevSummaryRef.current;
      prevSummaryRef.current = null;
      setDisplayed(summary);
      setComparison(against && against.brainId !== summary.brainId ? against : null);
      dispatch({ type: "resultReady", summary });
    });
    dispatch({ type: "fire" });
  };
```

Note `runInference` no longer returns a `ForwardResult` — no caller uses the return value.

- [ ] **Step 3: Remove the orphaned barsTimer references**

Two spots still reference `barsTimer`:

1. Brain-swap effect (currently line 97): delete the line `window.clearTimeout(barsTimer.current);` — keep the other two clears and the `prevSummaryRef.current = displayedRef.current;` capture.
2. Interactive-entry effect (currently lines 213–215 and 225): delete the comment sentence about "A bars timer from an attract-mode fire may still be pending…" and the `window.clearTimeout(barsTimer.current);` lines in both the effect body and its cleanup. The stale-verdict protection now lives in the fire callback's mode guard. Keep `setDisplayed(null)` and everything else.

- [ ] **Step 4: Verify the smoke-test mock still matches**

`src/smoke.test.tsx` stubs `fire: vi.fn()` — signature-compatible with `fire(pass, onResult)` (the stub simply never invokes the callback, which is a legal outcome per the SceneApi contract). No change required; confirm the file needs no edit and move on.

- [ ] **Step 5: Full verification**

Run: `npm test`
Expected: ALL tests pass (previously 108 + the new stagedPass/stepsDue/PulseSystem/state tests).

Run: `npm run typecheck`
Expected: clean — the Task 4 carve-out for App.tsx/smoke.test.tsx no longer applies.

Run: `npm run lint`
Expected: clean (deleting `showResultLater`/`barsTimer` leaves no unused vars; `timelineFor` is still used by `cinematicTimer` and stays).

- [ ] **Step 6: Manual sanity check (visual)**

Run: `npm run dev`, open the printed URL. Confirm:
- Draw a digit: the output digits stay **dim** until the last pulse wave lands, then the answer lights with the flare. No digit glows at fire start.
- The HUD probability bars appear at the same moment the output column lights.
- Swap brains mid-cinematic (B key): no crash, verdict clears, morph runs, held digit re-fires.
- Attract mode (wait or reload): sample digits fire on the timer; verdicts land only at wave arrival.

Stop the dev server when done.

- [ ] **Step 7: Commit**

```bash
git add src/App.tsx
git commit -m "feat: HUD verdict lands via the scene's resultReady callback

The wall-clock bars timer duplicating timeline knowledge is gone; the
app learns the answer when the output layer computes, not before.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
