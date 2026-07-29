# Ambient Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Subtle life in interactive mode — camera micro-drift, coherent neuron breathing, and draw-reactive ambient sparks, all ducked whenever the fire cinematic or a morph is playing.

**Architecture:** All tunable math lives in a new pure module `src/scene/ambient.ts` (the sibling of `fire.ts` — no WebGL, fully unit-tested). One new scene object `AmbientSparks` (a 48-particle `THREE.Points` pool, miniature sibling of `PulseSystem`). `CameraRig` and `SceneManager` consume the pure functions; `SceneApi` does not change.

**Tech Stack:** TypeScript, three.js 0.185.1, Vitest 4 (jsdom — constructing three geometries/materials works without a GL context; only rendering needs one).

**Spec:** `docs/superpowers/specs/2026-07-29-ambient-motion-design.md`

## Global Constraints

- `SceneApi` must not change — the scene observes drawing through the `setInputPixels` calls it already receives.
- Ambient motion must never read as real signal propagation: sparks stay smaller and dimmer than `PulseSystem` comets (point size ≤ ~14 px vs pulses' 20–66; peak alpha ≤ 0.4).
- All behavioral randomness goes through an injected RNG (`rng: () => number`) so tests are deterministic. Cosmetic build-time jitter may use `Math.random` (matching `PulseSystem`).
- Every tuning constant is a named export at the top of `ambient.ts`.
- NEVER run `scripts/train-mnist.mjs` (it runs on import and overwrites the committed weights).
- Verify with `npm test` (vitest run) and `npm run typecheck` (tsc -b --noEmit). Both must pass before every commit.

---

### Task 1: Pure curves — `cameraDrift`, `breathe`, `duckEnvelope`

**Files:**
- Create: `src/scene/ambient.ts`
- Test: `src/scene/ambient.test.ts`

**Interfaces:**
- Consumes: `smoothstep` from `src/scene/fire.ts` (already exported).
- Produces (later tasks rely on these exact names):
  - `cameraDrift(elapsed: number): { px, py, pz, tx, ty }` (all `number`, world units)
  - `breathe(elapsed: number, x: number, y: number): number` (0..1)
  - `duckEnvelope(tSinceFire: number, fireTotal: number, morphing: boolean): number` (0..1)
  - Constants: `DRIFT_POS_AMP`, `DRIFT_TARGET_AMP`, `BREATHE_PERIOD_S`, `BREATHE_BRIGHTNESS`, `BREATHE_SCALE`, `DUCK_FLOOR`, `DUCK_ATTACK_S`, `DUCK_RELEASE_S`

- [ ] **Step 1: Write the failing tests**

Create `src/scene/ambient.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import {
  breathe,
  cameraDrift,
  DRIFT_POS_AMP,
  DRIFT_TARGET_AMP,
  DUCK_ATTACK_S,
  DUCK_FLOOR,
  DUCK_RELEASE_S,
  duckEnvelope,
} from "./ambient";

describe("cameraDrift", () => {
  it("stays within its amplitudes", () => {
    for (let t = 0; t < 120; t += 0.37) {
      const d = cameraDrift(t);
      expect(Math.abs(d.px)).toBeLessThanOrEqual(DRIFT_POS_AMP);
      expect(Math.abs(d.py)).toBeLessThanOrEqual(DRIFT_POS_AMP);
      expect(Math.abs(d.pz)).toBeLessThanOrEqual(DRIFT_POS_AMP);
      expect(Math.abs(d.tx)).toBeLessThanOrEqual(DRIFT_TARGET_AMP);
      expect(Math.abs(d.ty)).toBeLessThanOrEqual(DRIFT_TARGET_AMP);
    }
  });

  it("actually moves", () => {
    expect(cameraDrift(0).px).not.toBe(cameraDrift(3).px);
  });

  it("does not repeat after one axis period (incommensurate periods)", () => {
    // 11s is the px period; the other axes must be elsewhere in their cycles
    const a = cameraDrift(10);
    const b = cameraDrift(10 + 11);
    expect(Math.abs(a.px - b.px)).toBeLessThan(1e-6);
    expect(Math.abs(a.py - b.py)).toBeGreaterThan(0.01);
    expect(Math.abs(a.pz - b.pz)).toBeGreaterThan(0.01);
  });
});

describe("breathe", () => {
  it("is bounded to [0, 1]", () => {
    for (let t = 0; t < 30; t += 0.31) {
      const v = breathe(t, -16.5, 3);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("gives distant neurons different phase (a traveling wave, not a blink)", () => {
    // Input plane (x ≈ -16.5) vs output column (x = 18) at the same instant
    expect(Math.abs(breathe(2, -16.5, 0) - breathe(2, 18, 0))).toBeGreaterThan(0.05);
  });

  it("is deterministic", () => {
    expect(breathe(5.5, 2, -3)).toBe(breathe(5.5, 2, -3));
  });
});

describe("duckEnvelope", () => {
  const TOTAL = 4; // a plausible fire.total

  it("is full at idle (fireStart is -1e9, so tSinceFire is huge)", () => {
    expect(duckEnvelope(1e9, TOTAL, false)).toBe(1);
  });

  it("reaches the floor by the end of the attack ramp", () => {
    expect(duckEnvelope(DUCK_ATTACK_S, TOTAL, false)).toBeCloseTo(DUCK_FLOOR, 5);
  });

  it("holds the floor through the cinematic", () => {
    expect(duckEnvelope(TOTAL / 2, TOTAL, false)).toBeCloseTo(DUCK_FLOOR, 5);
  });

  it("recovers to full after the cinematic ends", () => {
    expect(duckEnvelope(TOTAL + DUCK_RELEASE_S, TOTAL, false)).toBeCloseTo(1, 5);
  });

  it("is hard 0 while morphing, regardless of fire time", () => {
    expect(duckEnvelope(1e9, TOTAL, true)).toBe(0);
    expect(duckEnvelope(1, TOTAL, true)).toBe(0);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/scene/ambient.test.ts`
Expected: FAIL — `Cannot find module './ambient'` (or equivalent resolve error).

- [ ] **Step 3: Write the implementation**

Create `src/scene/ambient.ts`:

```ts
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
  return DUCK_FLOOR + (1 - DUCK_FLOOR) * smoothstep(0, DUCK_RELEASE_S, tSinceFire - fireTotal);
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/scene/ambient.test.ts`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add src/scene/ambient.ts src/scene/ambient.test.ts
git commit -m "feat: ambient curves — camera drift, breathing wave, duck envelope"
```

---

### Task 2: Excitement state and ink diff

**Files:**
- Modify: `src/scene/ambient.ts` (append)
- Test: `src/scene/ambient.test.ts` (append)

**Interfaces:**
- Produces:
  - `class AmbientState { excitement: number; bump(amount: number): void; decay(dt: number): void }`
  - `inkDelta(prev: Float32Array, next: Float32Array): { total: number; indices: number[] }`
  - Constants: `EXCITEMENT_TAU_S`, `INK_THRESHOLD`, `INK_TO_EXCITEMENT`, `DRAW_GLOW`

- [ ] **Step 1: Write the failing tests**

Append to `src/scene/ambient.test.ts` (add `AmbientState`, `inkDelta`, `INK_THRESHOLD` to the import):

```ts
describe("AmbientState excitement", () => {
  it("bump clamps to 1", () => {
    const s = new AmbientState();
    s.bump(0.7);
    s.bump(0.7);
    expect(s.excitement).toBe(1);
  });

  it("ignores negative bumps", () => {
    const s = new AmbientState();
    s.bump(0.5);
    s.bump(-2);
    expect(s.excitement).toBe(0.5);
  });

  it("decays below 0.05 within three time constants", () => {
    const s = new AmbientState();
    s.bump(1);
    for (let i = 0; i < 60 * 4.6; i++) s.decay(1 / 60);
    expect(s.excitement).toBeLessThan(0.05);
  });
});

describe("inkDelta", () => {
  it("reports total fresh ink and which pixels gained it", () => {
    const prev = new Float32Array(4);
    const next = Float32Array.from([0.5, INK_THRESHOLD - 0.01, 0, 0.9]);
    const d = inkDelta(prev, next);
    expect(d.total).toBeCloseTo(0.5 + (INK_THRESHOLD - 0.01) + 0.9, 5);
    // Only gains ≥ INK_THRESHOLD are worth a spark bias
    expect(d.indices).toEqual([0, 3]);
  });

  it("clearing the pad (all deltas negative) yields zero — never excites", () => {
    const prev = Float32Array.from([0.8, 0.3, 0.9, 0]);
    const next = new Float32Array(4);
    const d = inkDelta(prev, next);
    expect(d.total).toBe(0);
    expect(d.indices).toEqual([]);
  });

  it("identical frames yield zero", () => {
    const px = Float32Array.from([0.4, 0.4]);
    const d = inkDelta(px, px);
    expect(d.total).toBe(0);
    expect(d.indices).toEqual([]);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/scene/ambient.test.ts`
Expected: FAIL — `AmbientState` / `inkDelta` are not exported.

- [ ] **Step 3: Write the implementation**

Append to `src/scene/ambient.ts`:

```ts
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/scene/ambient.test.ts`
Expected: PASS (17 tests).

- [ ] **Step 5: Commit**

```bash
git add src/scene/ambient.ts src/scene/ambient.test.ts
git commit -m "feat: draw excitement state and positive-only ink diff"
```

---

### Task 3: SparkScheduler

**Files:**
- Modify: `src/scene/ambient.ts` (append)
- Test: `src/scene/ambient.test.ts` (append)

**Interfaces:**
- Produces:
  - `interface SparkSpawn { slot: number; stage: number; edge: number; duration: number; magnitude: number }`
  - `class SparkScheduler { constructor(edgeCounts: number[], stage0From: Uint16Array, rng?: () => number, poolSize?: number); noteInk(pixelIndices: number[]): void; clearInk(): void; update(now: number, dt: number, excitement: number, duck: number): SparkSpawn[] }`
  - Constants: `SPARK_POOL_SIZE`, `SPARK_MIN_DUR_S`, `SPARK_MAX_DUR_S`, `IDLE_CONCURRENCY`, `EXCITED_CONCURRENCY`, `INK_BIAS`, `INK_MEMORY`
- Consumes: nothing new (self-contained pure logic).

- [ ] **Step 1: Write the failing tests**

Append to `src/scene/ambient.test.ts` (extend the import accordingly):

```ts
/** Deterministic LCG so scheduler tests never flake. */
const lcg = (seed: number) => () =>
  (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;

/** Two stages: 8 stage-0 edges (from pixels 100..103), 4 stage-1 edges. */
const makeScheduler = (seed = 42, poolSize?: number) =>
  new SparkScheduler(
    [8, 4],
    Uint16Array.from([100, 100, 101, 101, 102, 102, 103, 103]),
    lcg(seed),
    poolSize
  );

describe("SparkScheduler", () => {
  it("is deterministic with the same seed", () => {
    const a = makeScheduler(7);
    const b = makeScheduler(7);
    const runsA: unknown[] = [];
    const runsB: unknown[] = [];
    for (let i = 0; i < 600; i++) {
      runsA.push(a.update(i / 60, 1 / 60, 0.5, 1));
      runsB.push(b.update(i / 60, 1 / 60, 0.5, 1));
    }
    expect(runsA).toEqual(runsB);
  });

  it("idle spawn rate sustains roughly IDLE_CONCURRENCY live sparks", () => {
    const s = makeScheduler();
    let spawned = 0;
    for (let i = 0; i < 60 * 60; i++) spawned += s.update(i / 60, 1 / 60, 0, 1).length;
    const avgDur = (SPARK_MIN_DUR_S + SPARK_MAX_DUR_S) / 2;
    const expected = (IDLE_CONCURRENCY / avgDur) * 60; // rate × seconds
    expect(spawned).toBeGreaterThan(expected * 0.7);
    expect(spawned).toBeLessThan(expected * 1.3);
  });

  it("full duck (0) stops all spawning", () => {
    const s = makeScheduler();
    let spawned = 0;
    for (let i = 0; i < 600; i++) spawned += s.update(i / 60, 1 / 60, 1, 0).length;
    expect(spawned).toBe(0);
  });

  it("at high excitement, spawns bias hard toward inked stage-0 edges", () => {
    const s = makeScheduler();
    s.noteInk([100, 102]);
    const inkedEdges = new Set([0, 1, 4, 5]); // edges whose from-pixel is 100 or 102
    let inked = 0;
    let total = 0;
    for (let i = 0; i < 60 * 30; i++) {
      for (const spawn of s.update(i / 60, 1 / 60, 1, 1)) {
        total++;
        if (spawn.stage === 0 && inkedEdges.has(spawn.edge)) inked++;
      }
    }
    expect(total).toBeGreaterThan(50);
    expect(inked / total).toBeGreaterThan(0.7);
  });

  it("never exceeds the pool size", () => {
    const s = makeScheduler(42, 4);
    const busyUntil: number[] = [];
    for (let i = 0; i < 600; i++) {
      const now = i / 60;
      for (const spawn of s.update(now, 1 / 60, 1, 1)) {
        expect(spawn.slot).toBeLessThan(4);
        busyUntil[spawn.slot] = now + spawn.duration;
        const live = busyUntil.filter((t) => t !== undefined && t > now).length;
        expect(live).toBeLessThanOrEqual(4);
      }
    }
  });

  it("recycles slots after a spark's lifetime", () => {
    const s = makeScheduler(42, 1);
    // Burn enough time at high drive to force at least two spawns through one slot
    const slots: number[] = [];
    for (let i = 0; i < 60 * 5; i++) {
      for (const spawn of s.update(i / 60, 1 / 60, 1, 1)) slots.push(spawn.slot);
    }
    expect(slots.length).toBeGreaterThan(1);
    expect(slots.every((slot) => slot === 0)).toBe(true);
  });

  it("spawn durations and magnitudes stay in range", () => {
    const s = makeScheduler();
    for (let i = 0; i < 600; i++) {
      for (const spawn of s.update(i / 60, 1 / 60, 0.5, 1)) {
        expect(spawn.duration).toBeGreaterThanOrEqual(SPARK_MIN_DUR_S);
        expect(spawn.duration).toBeLessThanOrEqual(SPARK_MAX_DUR_S);
        expect(spawn.magnitude).toBeGreaterThan(0);
        expect(spawn.magnitude).toBeLessThanOrEqual(1);
      }
    }
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/scene/ambient.test.ts`
Expected: FAIL — `SparkScheduler` is not exported.

- [ ] **Step 3: Write the implementation**

Append to `src/scene/ambient.ts`:

```ts
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/scene/ambient.test.ts`
Expected: PASS (24 tests).

- [ ] **Step 5: Commit**

```bash
git add src/scene/ambient.ts src/scene/ambient.test.ts
git commit -m "feat: SparkScheduler — deterministic ambient spark spawning with ink bias"
```

---

### Task 4: AmbientSparks scene object

**Files:**
- Create: `src/scene/AmbientSparks.ts`
- Test: `src/scene/AmbientSparks.test.ts`

**Interfaces:**
- Consumes: `SPARK_POOL_SIZE` from `./ambient`; `NetworkLayout`, `maxAbsWeight` from `./NetworkLayout`.
- Produces (SceneManager relies on):
  - `class AmbientSparks { readonly points: THREE.Points; constructor(layout: NetworkLayout, poolSize?: number); spawn(slot: number, stage: number, edge: number, birth: number, duration: number, magnitude: number): void; update(now: number, alpha: number): void; dispose(): void }`

- [ ] **Step 1: Write the failing tests**

Create `src/scene/AmbientSparks.test.ts` (net recipe mirrors `PulseSystem.test.ts`):

```ts
import { describe, expect, it } from "vitest";
import type * as THREE from "three";
import type { Net } from "../nn/weights";
import { AmbientSparks } from "./AmbientSparks";
import { buildLayout } from "./NetworkLayout";

/** Deterministic pseudo-random net (same recipe as PulseSystem.test.ts). */
const makeNet = (shape: number[]): Net => ({
  shape,
  layers: Array.from({ length: shape.length - 1 }, (_, l) => {
    const [nIn, nOut] = [shape[l], shape[l + 1]];
    const W = new Float32Array(nIn * nOut);
    for (let i = 0; i < W.length; i++) W[i] = Math.sin(i * 12.9898 + l) * 0.5;
    return { W, b: new Float32Array(nOut) };
  }),
});

function build() {
  const layout = buildLayout(makeNet([784, 4, 10]));
  const sparks = new AmbientSparks(layout, 8);
  const attr = (name: string) =>
    sparks.points.geometry.getAttribute(name).array as Float32Array;
  return { layout, sparks, attr };
}

describe("AmbientSparks", () => {
  it("every slot starts dead (birth in the far past)", () => {
    const { attr } = build();
    expect(Array.from(attr("aBirth")).every((b) => b <= -1e8)).toBe(true);
  });

  it("spawn writes the chosen edge's endpoints into the slot", () => {
    const { layout, sparks, attr } = build();
    const set = layout.edges[1]; // hidden → output stage
    sparks.spawn(3, 1, 5, 2.0, 0.6, 0.5);
    const from = layout.layerPositions[0];
    const to = layout.layerPositions[1];
    for (let axis = 0; axis < 3; axis++) {
      expect(attr("aStart")[3 * 3 + axis]).toBeCloseTo(from[set.from[5] * 3 + axis], 5);
      expect(attr("aEnd")[3 * 3 + axis]).toBeCloseTo(to[set.to[5] * 3 + axis], 5);
    }
    expect(attr("aBirth")[3]).toBe(2.0);
    expect(attr("aDur")[3]).toBeCloseTo(0.6, 5);
  });

  it("stage 0 spawns start on the input plane", () => {
    const { layout, sparks, attr } = build();
    const set = layout.edges[0];
    sparks.spawn(0, 0, 7, 1.0, 0.5, 0.4);
    for (let axis = 0; axis < 3; axis++) {
      expect(attr("aStart")[axis]).toBeCloseTo(
        layout.inputPositions[set.from[7] * 3 + axis],
        5
      );
    }
  });

  it("magnitude is signed by the edge's weight (warm/cool tint)", () => {
    const { layout, sparks, attr } = build();
    const set = layout.edges[1];
    const positive = Array.from(set.weight).findIndex((w) => w > 0);
    const negative = Array.from(set.weight).findIndex((w) => w < 0);
    sparks.spawn(0, 1, positive, 0, 0.5, 0.6);
    sparks.spawn(1, 1, negative, 0, 0.5, 0.6);
    expect(attr("aMag")[0]).toBeCloseTo(0.6, 5);
    expect(attr("aMag")[1]).toBeCloseTo(-0.6, 5);
  });

  it("update drives the shader clock and ambient alpha", () => {
    const { sparks } = build();
    sparks.update(12.5, 0.15);
    const material = sparks.points.material as THREE.ShaderMaterial;
    expect(material.uniforms.uTime.value).toBe(12.5);
    expect(material.uniforms.uAlpha.value).toBe(0.15);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/scene/AmbientSparks.test.ts`
Expected: FAIL — `Cannot find module './AmbientSparks'`.

- [ ] **Step 3: Write the implementation**

Create `src/scene/AmbientSparks.ts`:

```ts
import * as THREE from "three";
import { SPARK_POOL_SIZE } from "./ambient";
import { maxAbsWeight, type NetworkLayout } from "./NetworkLayout";

/**
 * Idle static: a small pool of dim sparks drifting along connection edges.
 * A miniature sibling of PulseSystem — same GPU edge interpolation, one
 * Points draw call — but deliberately smaller and dimmer than the pulse
 * comets. Ambient motion must read as static electricity, never as a real
 * signal wave: the answer is earned by the wavefront.
 */
export class AmbientSparks {
  readonly points: THREE.Points;
  private readonly material: THREE.ShaderMaterial;
  private readonly startAttr: THREE.BufferAttribute;
  private readonly endAttr: THREE.BufferAttribute;
  private readonly birthAttr: THREE.BufferAttribute;
  private readonly durAttr: THREE.BufferAttribute;
  private readonly magAttr: THREE.BufferAttribute;
  private readonly layout: NetworkLayout;
  private readonly weightNorm: number[];

  constructor(layout: NetworkLayout, poolSize: number = SPARK_POOL_SIZE) {
    this.layout = layout;
    this.weightNorm = layout.edges.map((set) => maxAbsWeight(set));

    const dyn = (itemSize: number, fill = 0) => {
      const attr = new THREE.BufferAttribute(
        new Float32Array(poolSize * itemSize).fill(fill),
        itemSize
      );
      attr.setUsage(THREE.DynamicDrawUsage);
      return attr;
    };

    const geometry = new THREE.BufferGeometry();
    // `position` must exist for draw range; travel is aStart→aEnd in the shader.
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(poolSize * 3), 3));
    geometry.setAttribute("aStart", (this.startAttr = dyn(3)));
    geometry.setAttribute("aEnd", (this.endAttr = dyn(3)));
    // Born in the far past with a nonzero duration: every slot starts dead.
    geometry.setAttribute("aBirth", (this.birthAttr = dyn(1, -1e9)));
    geometry.setAttribute("aDur", (this.durAttr = dyn(1, 1)));
    geometry.setAttribute("aMag", (this.magAttr = dyn(1)));

    this.material = new THREE.ShaderMaterial({
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      uniforms: {
        uTime: { value: 0 },
        uAlpha: { value: 1 },
      },
      vertexShader: /* glsl */ `
        attribute vec3 aStart;
        attribute vec3 aEnd;
        attribute float aBirth;
        attribute float aDur;
        attribute float aMag;
        uniform float uTime;
        uniform float uAlpha;
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          float p = (uTime - aBirth) / aDur;
          float visible = step(0.0, p) * (1.0 - step(1.0, p));
          float eased = p * p * (3.0 - 2.0 * p);
          vec3 pos = mix(aStart, aEnd, clamp(eased, 0.0, 1.0));
          vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
          gl_Position = projectionMatrix * mvPosition;
          float mag = abs(aMag);
          float fade = smoothstep(0.0, 0.25, p) * (1.0 - smoothstep(0.7, 1.0, p));
          vAlpha = visible * fade * (0.10 + 0.30 * mag) * uAlpha;
          vec3 warm = vec3(1.0, 0.62, 0.22);
          vec3 cool = vec3(0.30, 0.75, 1.0);
          vColor = (aMag >= 0.0 ? warm : cool) * (0.4 + 0.9 * mag);
          gl_PointSize = visible * (5.0 + 9.0 * mag) * (24.0 / max(1.0, -mvPosition.z));
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          if (vAlpha < 0.003) discard;
          float d = length(gl_PointCoord - 0.5);
          float core = smoothstep(0.5, 0.06, d);
          gl_FragColor = vec4(vColor, vAlpha * core);
        }
      `,
    });

    this.points = new THREE.Points(geometry, this.material);
    this.points.frustumCulled = false;
  }

  /** Launch one spark along an edge. Tint follows the weight's sign, so a
   *  spark tells the same warm/cool story as the wiring it rides. */
  spawn(
    slot: number,
    stage: number,
    edge: number,
    birth: number,
    duration: number,
    magnitude: number
  ): void {
    const set = this.layout.edges[stage];
    const from =
      stage === 0 ? this.layout.inputPositions : this.layout.layerPositions[stage - 1];
    const to = this.layout.layerPositions[stage];
    for (let axis = 0; axis < 3; axis++) {
      this.startAttr.setComponent(slot, axis, from[set.from[edge] * 3 + axis]);
      this.endAttr.setComponent(slot, axis, to[set.to[edge] * 3 + axis]);
    }
    this.birthAttr.setX(slot, birth);
    this.durAttr.setX(slot, duration);
    const sign = set.weight[edge] / this.weightNorm[stage] >= 0 ? 1 : -1;
    this.magAttr.setX(slot, sign * magnitude);
    this.startAttr.needsUpdate = true;
    this.endAttr.needsUpdate = true;
    this.birthAttr.needsUpdate = true;
    this.durAttr.needsUpdate = true;
    this.magAttr.needsUpdate = true;
  }

  /** Per-frame: advance the shader clock; alpha is the duck envelope, so
   *  live sparks dim (not vanish) when a fire starts. */
  update(now: number, alpha: number): void {
    this.material.uniforms.uTime.value = now;
    this.material.uniforms.uAlpha.value = alpha;
  }

  dispose(): void {
    this.points.geometry.dispose();
    this.material.dispose();
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/scene/AmbientSparks.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/scene/AmbientSparks.ts src/scene/AmbientSparks.test.ts
git commit -m "feat: AmbientSparks — pooled dim sparks riding connection edges"
```

---

### Task 5: CameraRig micro-drift

**Files:**
- Modify: `src/scene/CameraRig.ts`

**Interfaces:**
- Consumes: `cameraDrift` from `./ambient` (Task 1).
- Produces: `update(elapsed: number, dt: number, driftAmp?: number)` — third param defaults to 1, so the existing `rig.update(elapsed, dt)` call keeps compiling until Task 6 passes the duck envelope.

The drift math itself is unit-tested in Task 1; this task is thin consumption (matching the house pattern: pure math tested, WebGL glue thin). No new test file.

- [ ] **Step 1: Apply the change**

In `src/scene/CameraRig.ts`, add the import:

```ts
import { cameraDrift } from "./ambient";
```

Replace the `update` method's interactive branch (currently `this.desiredPosition.copy(this.interactivePosition); this.desiredTarget.copy(this.interactiveTarget);`) so the whole method reads:

```ts
  update(elapsed: number, dt: number, driftAmp = 1): void {
    if (this.mode === "attract") {
      this.curve.getPoint((elapsed / ATTRACT_PERIOD) % 1, this.desiredPosition);
      // Gaze drifts slowly around the network's heart
      this.desiredTarget.set(
        1.5 + 4.5 * Math.sin(elapsed * 0.05),
        1.1 * Math.sin(elapsed * 0.073),
        0
      );
    } else {
      // Micro-drift keeps the locked framing breathing; the duck envelope
      // stills the camera during the fire cinematic (stillness = attention).
      const drift = cameraDrift(elapsed);
      this.desiredPosition.set(
        this.interactivePosition.x + drift.px * driftAmp,
        this.interactivePosition.y + drift.py * driftAmp,
        this.interactivePosition.z + drift.pz * driftAmp
      );
      this.desiredTarget.set(
        this.interactiveTarget.x + drift.tx * driftAmp,
        this.interactiveTarget.y + drift.ty * driftAmp,
        this.interactiveTarget.z
      );
    }
    // Critically-damped exponential smoothing (frame-rate independent)
    const k = 1 - Math.exp(-1.8 * dt);
    this.camera.position.lerp(this.desiredPosition, k);
    this.smoothedTarget.lerp(this.desiredTarget, k);
    this.camera.lookAt(this.smoothedTarget);
  }
```

- [ ] **Step 2: Verify**

Run: `npm run typecheck` — expected: clean.
Run: `npm test` — expected: all tests pass (no behavior under test changed).

- [ ] **Step 3: Commit**

```bash
git add src/scene/CameraRig.ts
git commit -m "feat: interactive camera micro-drift, stilled by the duck envelope"
```

---

### Task 6: SceneManager wiring + manual verification

**Files:**
- Modify: `src/scene/SceneManager.ts`

**Interfaces:**
- Consumes (exact names from Tasks 1–5):
  - `AmbientState`, `SparkScheduler`, `inkDelta`, `breathe`, `duckEnvelope`, `BREATHE_BRIGHTNESS`, `BREATHE_SCALE`, `INK_TO_EXCITEMENT`, `DRAW_GLOW` from `./ambient`
  - `AmbientSparks` from `./AmbientSparks`
  - `rig.update(elapsed, dt, driftAmp)` from Task 5
- Produces: no API change — `SceneApi` is untouched.

- [ ] **Step 1: Add imports and ambient state**

Add to the imports in `src/scene/SceneManager.ts`:

```ts
import {
  AmbientState,
  BREATHE_BRIGHTNESS,
  BREATHE_SCALE,
  breathe,
  DRAW_GLOW,
  duckEnvelope,
  INK_TO_EXCITEMENT,
  inkDelta,
  SparkScheduler,
} from "./ambient";
import { AmbientSparks } from "./AmbientSparks";
```

After the `let pulses = new PulseSystem(layout, fire);` line, add the ambient subsystems and include the spark points in the same `scene.add` group as the pulses:

```ts
let sparks = new AmbientSparks(layout);
let scheduler = new SparkScheduler(
  layout.edges.map((set) => set.count),
  layout.edges[0].from
);
const ambient = new AmbientState();
// Baseline for the ink diff: what the input plane currently shows.
const prevPixels = new Float32Array(layout.inputPositions.length / 3);
```

Add `sparks.points` to the scene alongside the other rebuilt subsystems:

```ts
scene.add(pulses.points, glyphs.group, flare.mesh, sparks.points);
```

- [ ] **Step 2: Rebuild ambient subsystems on brain swap**

In `swapSubsystems`, dispose/rebuild sparks and scheduler with the other topology-shaped subsystems, and reset excitement (the pad clears on swap):

```ts
  function swapSubsystems(nextNet: Net, options: LayoutOptions): void {
    scene.remove(neurons.mesh, connections.lines, pulses.points, sparks.points);
    neurons.dispose();
    connections.dispose();
    pulses.dispose();
    sparks.dispose();
    layout = buildLayout(nextNet, options);
    fire = makeFireTimeline(layout.edges.length);
    neurons = new NeuronField(layout);
    connections = new ConnectionMesh(layout);
    pulses = new PulseSystem(layout, fire);
    sparks = new AmbientSparks(layout);
    scheduler = new SparkScheduler(
      layout.edges.map((set) => set.count),
      layout.edges[0].from
    );
    ambient.excitement = 0;
    prevPixels.fill(0);
    scene.add(connections.lines, neurons.mesh, pulses.points, sparks.points);
    layerCounts = layout.layerPositions.map((p) => p.length / 3);
  }
```

- [ ] **Step 3: Breathe instead of shimmer**

Change `updateNeurons` to take the duck level and replace the shimmer with the coherent wave. The signature becomes `updateNeurons(tSinceFire: number, morph: ..., duck: number)` and the top of the inner loop changes from:

```ts
        const instance = neurons.indexOf(layer, i);
        const shimmer = 0.02 * Math.sin(elapsed * 1.3 + instance * 0.71);
        let brightness = 0.07 + shimmer;
        let scale = 1;
```

to:

```ts
        const instance = neurons.indexOf(layer, i);
        const pos = neurons.positionOf(layer, i);
        const wave = breathe(elapsed, pos.x, pos.y);
        let brightness = 0.07 + BREATHE_BRIGHTNESS * duck * wave;
        let scale = 1 + BREATHE_SCALE * duck * (2 * wave - 1);
```

(`positionOf` returns a stored `Vector3` — no per-neuron allocation.)

- [ ] **Step 4: Wire the frame loop**

In `frame()`, after the `fireDriver?.advance(tSinceFire);` line, compute the ambient scalars and thread them through (replacing the existing `rig.update(elapsed, dt);` call and the `updateNeurons(tSinceFire, morph)` call):

```ts
    ambient.decay(dt);
    const duck = duckEnvelope(tSinceFire, fire.total, morphState !== null);

    rig.update(elapsed, dt, duck);
    starfield.update(elapsed);
    pulses.update(elapsed);
    for (const s of scheduler.update(elapsed, dt, ambient.excitement, duck)) {
      sparks.spawn(s.slot, s.stage, s.edge, elapsed, s.duration, s.magnitude);
    }
    sparks.update(elapsed, duck);
```

Replace the `setStageGlow` call so the draw-glow lift adds to (never replaces) the cinematic glow:

```ts
    // Stage 0's many lines share one small screen region — halve its glow lift.
    // While the visitor inks, the input wiring warms under the fresh strokes.
    connections.setStageGlow(
      layout.edges.map((_, stage) => {
        const cinematic = fireActive
          ? (stage === 0 ? 0.5 : 1) * stageGlow(fire, tSinceFire, stage)
          : 0;
        const draw = stage === 0 ? DRAW_GLOW * ambient.excitement * duck : 0;
        return cinematic + draw;
      })
    );
```

And pass the duck level into the neuron update:

```ts
    const flareEnv = updateNeurons(tSinceFire, morph, duck);
```

- [ ] **Step 5: Observe ink in setInputPixels, keep the diff baseline honest in fire()**

Replace the `setInputPixels` API method:

```ts
    setInputPixels(pixels) {
      // The scene observes drawing through the calls it already receives:
      // fresh ink excites the ambience and marks pixels for spark bias.
      const delta = inkDelta(prevPixels, pixels);
      prevPixels.set(pixels);
      if (delta.total > 0) {
        ambient.bump(delta.total * INK_TO_EXCITEMENT);
        scheduler.noteInk(delta.indices);
      }
      inputPlane.setPixels(pixels);
    },
```

In `fire()`, right after `inputPlane.setPixels(pass.activations[0]);`, sync the baseline without exciting (a fire is not a stroke):

```ts
      prevPixels.set(pass.activations[0]);
```

- [ ] **Step 6: Dispose**

Add `sparks.dispose();` to the `dispose()` method alongside `pulses.dispose();`.

- [ ] **Step 7: Verify — typecheck, full suite**

Run: `npm run typecheck`
Expected: clean.
Run: `npm test`
Expected: all tests pass (159 existing + ~29 new).

- [ ] **Step 8: Manual verification on the dev server**

Run: `npm run dev`, open the app, and check each behavior:

1. **Idle interactive:** camera framing drifts almost imperceptibly (watch a neuron against a screen edge for ~15s); brightness swells roll across the columns; a couple of dim sparks wander the wiring.
2. **While drawing:** sparks concentrate on the input-plane fan-out; stage-0 lines warm faintly; the effect fades ~2s after the last stroke.
3. **Fire:** the instant a pass starts, ambience stills (camera freezes, sparks thin to near-nothing) and the cinematic reads exactly as before; ambience breathes back in ~1s after the verdict lands.
4. **Brain swap (thumbs-up):** no ambient motion during the morph; no crash on the rebuilt topology; sparks return after.
5. **Attract mode:** orbit unchanged; idle sparks + breathing present.
6. **Clear pad:** clearing produces no excitement burst.

- [ ] **Step 9: Commit**

```bash
git add src/scene/SceneManager.ts
git commit -m "feat: wire ambient motion — breathing, sparks, draw glow, ducked by fires"
```
