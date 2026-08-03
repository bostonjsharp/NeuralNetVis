# Pose-Driven Gestures at Distance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The full gesture vocabulary (wake, ✕ clear, draw, brain-cycle) works at 2–4m by promoting the already-loaded body-pose model to a full "far tier" in `FramePipeline`, while close-range hand behavior stays byte-identical.

**Architecture:** A two-tier `FramePipeline` — a close hand drives exactly today's path; otherwise a visible body pose synthesizes the same `GestureState` (wrist cursor through a shoulder-anchored pad box, reach-toward-screen as pen evidence into the existing `PenLatch`, arm-out-to-the-side as the brain-cycle verb). Inference strides flip by driving tier. `GestureController` and `App.tsx` are untouched.

**Tech Stack:** TypeScript, Vite, Vitest, `@mediapipe/tasks-vision` (GestureRecognizer + PoseLandmarker, both already shipped).

**Spec:** `docs/superpowers/specs/2026-08-03-pose-distance-gestures-design.md`

## Global Constraints

- Existing tests pass with **no behavioral edits** — only import-path updates from the Task 1 module move. This is the enforced "close tier unchanged" guarantee.
- `GestureController.ts`, `App.tsx`, and all `GestureState` field names are untouched (`thumbsUp` keeps its name; it means "the brain-cycle verb is latched").
- Every tuning threshold is an exported, named, documented constant in `poseGesture.ts` — they are the on-site calibration surface.
- Unreadable input (low visibility, missing `z`) yields `NO_EVIDENCE` / no movement, never a guess — the latch's decay does the rest.
- Strides: close tier `{hand: 1, pose: 6}`, far tier (or nobody) `{hand: 4, pose: 1}`.
- Verify with `npm test`, `npm run typecheck`, `npm run lint`. NEVER run `scripts/train-mnist.mjs` (it overwrites committed weights on import).
- Comment style: match the codebase — comments state constraints/why, never narrate the next line.

## File Structure

- `src/gesture/poseGesture.ts` — **new.** Pure body-pose math, the pose-side twin of `handGesture.ts`. Receives the existing pose section (`PoseLandmark`, `POSE`, `segmentsIntersect`, `forearmsCrossed`) plus all new far-tier primitives.
- `src/gesture/poseGesture.test.ts` — **new.** Tests for everything above.
- `src/gesture/handGesture.ts` — loses its body-pose section (moves out verbatim).
- `src/gesture/handGesture.test.ts` — loses the pose tests (recreated in `poseGesture.test.ts`).
- `src/gesture/framePipeline.ts` — gains the far tier + `tier` output + debug fields.
- `src/gesture/framePipeline.test.ts` — import fix (Task 1), far-tier tests (Task 3).
- `src/gesture/visionTasks.ts` — `POSE_FRAME_STRIDE` → `STRIDES` pair; init protocol drops `poseStride`; `gestureMs` becomes optional in the response.
- `src/gesture/visionWorker.ts`, `src/gesture/handTracker.ts` — tier-driven stride loops.
- `src/hud/DebugPanel.tsx` — pose skeleton, reach-charge bar, tier indicator.

---

### Task 1: Extract `poseGesture.ts` (pure move, no behavior change)

**Files:**
- Create: `src/gesture/poseGesture.ts`
- Modify: `src/gesture/handGesture.ts` (delete lines 138–185, the `---- body-pose ✕ detection ----` section)
- Modify: `src/gesture/framePipeline.ts:1-21` (split the import)
- Modify: `src/gesture/framePipeline.test.ts:7` (import `PoseLandmark` from the new module)
- Modify: `src/gesture/handGesture.test.ts` (remove the three pose tests + their imports)
- Create: `src/gesture/poseGesture.test.ts` (the moved tests)

**Interfaces:**
- Consumes: `Landmark` from `./handGesture`.
- Produces: `poseGesture.ts` exporting `PoseLandmark`, `POSE` (`{leftElbow: 13, rightElbow: 14, leftWrist: 15, rightWrist: 16}` for now), `POSE_VISIBILITY_MIN = 0.5`, `segmentsIntersect(a1, a2, b1, b2): boolean`, `forearmsCrossed(pose: PoseLandmark[]): boolean`.

- [ ] **Step 1: Create `src/gesture/poseGesture.ts`**

```typescript
import type { Landmark } from "./handGesture";

/**
 * Pure body-pose math — the pose-side twin of handGesture.ts, no camera,
 * no MediaPipe imports, unit-testable. Body pose reads at far greater
 * range than hands; past ~2m it is the only signal the pipeline has.
 */

export interface PoseLandmark extends Landmark {
  visibility?: number;
}

/** MediaPipe pose landmark indices. */
export const POSE = {
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
} as const;

export const POSE_VISIBILITY_MIN = 0.5;

/** Strict 2D segment intersection (shared endpoints/collinear don't count). */
export function segmentsIntersect(
  a1: Landmark,
  a2: Landmark,
  b1: Landmark,
  b2: Landmark
): boolean {
  const cross = (o: Landmark, p: Landmark, q: Landmark) =>
    (p.x - o.x) * (q.y - o.y) - (p.y - o.y) * (q.x - o.x);
  const d1 = cross(b1, b2, a1);
  const d2 = cross(b1, b2, a2);
  const d3 = cross(a1, a2, b1);
  const d4 = cross(a1, a2, b2);
  return d1 > 0 !== d2 > 0 && d3 > 0 !== d4 > 0;
}

/** The literal ✕ test: do the two forearm segments (elbow→wrist) cross? */
export function forearmsCrossed(pose: PoseLandmark[]): boolean {
  const points = [
    pose[POSE.leftElbow],
    pose[POSE.leftWrist],
    pose[POSE.rightElbow],
    pose[POSE.rightWrist],
  ];
  if (points.some((p) => !p || (p.visibility ?? 1) < POSE_VISIBILITY_MIN)) return false;
  return segmentsIntersect(
    pose[POSE.leftElbow],
    pose[POSE.leftWrist],
    pose[POSE.rightElbow],
    pose[POSE.rightWrist]
  );
}
```

Note: `POSE_VISIBILITY_MIN` was module-private in `handGesture.ts`; it becomes exported here (Tasks 2–5 need it).

- [ ] **Step 2: Delete the moved section from `handGesture.ts`**

Delete everything from the line `// ---- body-pose ✕ detection (arms readable at far greater range) ----` through the closing brace of `forearmsCrossed` (currently lines 138–185). `PadMapper`, `OneEuroFilter`, `PenLatch` and everything above the section stay.

- [ ] **Step 3: Fix imports in `framePipeline.ts`**

Replace the single import block at the top:

```typescript
import {
  handSpan,
  isCloseEnough,
  isRaised,
  NO_EVIDENCE,
  OneEuroFilter,
  PadMapper,
  PALM,
  penEvidence,
  PenLatch,
  pickPrimaryHand,
  thumbEvidence,
  WRIST,
  wristsClose,
  type GestureCategory,
  type Landmark,
  type PenPose,
} from "./handGesture";
import { forearmsCrossed, POSE, type PoseLandmark } from "./poseGesture";
```

- [ ] **Step 4: Fix imports in the test files**

In `framePipeline.test.ts` line 7, change to:

```typescript
import type { GestureCategory, Landmark } from "./handGesture";
import type { PoseLandmark } from "./poseGesture";
```

In `handGesture.test.ts`: remove `forearmsCrossed`, `POSE`, `segmentsIntersect` (and `PoseLandmark` if imported) from the import list, and delete the three tests: `"segmentsIntersect finds a true crossing"`, the forearms-crossed-true test (~line 215), the forearms-not-crossed test (~line 225), and the low-visibility test (~line 235). (Exact titles visible in the file; delete every test that calls `segmentsIntersect` or `forearmsCrossed`.)

- [ ] **Step 5: Create `src/gesture/poseGesture.test.ts` with the moved coverage**

```typescript
import { describe, expect, it } from "vitest";
import { forearmsCrossed, POSE, segmentsIntersect, type PoseLandmark } from "./poseGesture";

/** Sparse pose array with only the named indices populated. */
const poseWith = (points: Record<number, PoseLandmark>): PoseLandmark[] => {
  const pose: PoseLandmark[] = Array.from({ length: 33 }, () => ({
    x: 0,
    y: 0,
    visibility: 0,
  }));
  for (const [i, p] of Object.entries(points)) pose[Number(i)] = p;
  return pose;
};

describe("segmentsIntersect", () => {
  it("finds a true crossing", () => {
    expect(
      segmentsIntersect({ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 }, { x: 1, y: 0 })
    ).toBe(true);
    expect(
      segmentsIntersect({ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 1 })
    ).toBe(false);
  });
});

describe("forearmsCrossed", () => {
  it("detects forearm segments crossing in an ✕", () => {
    const pose = poseWith({
      [POSE.leftElbow]: { x: 0.6, y: 0.6, visibility: 1 },
      [POSE.leftWrist]: { x: 0.4, y: 0.4, visibility: 1 },
      [POSE.rightElbow]: { x: 0.4, y: 0.6, visibility: 1 },
      [POSE.rightWrist]: { x: 0.6, y: 0.4, visibility: 1 },
    });
    expect(forearmsCrossed(pose)).toBe(true);
  });

  it("stays false for parallel forearms", () => {
    const pose = poseWith({
      [POSE.leftElbow]: { x: 0.6, y: 0.6, visibility: 1 },
      [POSE.leftWrist]: { x: 0.6, y: 0.4, visibility: 1 },
      [POSE.rightElbow]: { x: 0.4, y: 0.6, visibility: 1 },
      [POSE.rightWrist]: { x: 0.4, y: 0.4, visibility: 1 },
    });
    expect(forearmsCrossed(pose)).toBe(false);
  });

  it("refuses to guess from a low-visibility landmark", () => {
    const pose = poseWith({
      [POSE.leftElbow]: { x: 0.6, y: 0.6, visibility: 0.2 },
      [POSE.leftWrist]: { x: 0.4, y: 0.4, visibility: 1 },
      [POSE.rightElbow]: { x: 0.4, y: 0.6, visibility: 1 },
      [POSE.rightWrist]: { x: 0.6, y: 0.4, visibility: 1 },
    });
    expect(forearmsCrossed(pose)).toBe(false);
  });
});
```

- [ ] **Step 6: Verify — full suite, typecheck**

Run: `npm test` → all tests pass (same count as before this task, 233 + the recreated ones netting zero behavior change).
Run: `npm run typecheck` → clean.

- [ ] **Step 7: Commit**

```bash
git add src/gesture/poseGesture.ts src/gesture/poseGesture.test.ts src/gesture/handGesture.ts src/gesture/handGesture.test.ts src/gesture/framePipeline.ts src/gesture/framePipeline.test.ts
git commit -m "refactor: extract body-pose math into poseGesture.ts"
```

---

### Task 2: Far-tier pose primitives (TDD)

**Files:**
- Modify: `src/gesture/poseGesture.ts`
- Test: `src/gesture/poseGesture.test.ts`

**Interfaces:**
- Consumes: `NO_EVIDENCE`, `type PenEvidence` from `./handGesture` (shape `{fist: number; open: number}` — for far-tier evidence, `fist` carries the pen-down/verb-on drive, `open` the counter-drive, exactly like `thumbEvidence` does for 👍).
- Produces (all exported from `poseGesture.ts`):
  - `POSE` extended with `leftShoulder: 11, rightShoulder: 12, leftHip: 23, rightHip: 24`
  - `type Side = "left" | "right"`
  - `visible(p: PoseLandmark | undefined): p is PoseLandmark`
  - `shoulderWidth(pose: PoseLandmark[]): number` (0 when unreadable)
  - `wristOf(pose: PoseLandmark[], side: Side): PoseLandmark`
  - `POSE_RAISE_MARGIN = 0.05`, `poseRaised(pose, side): boolean`
  - `ARM_OUT_LATERAL = 0.8`, `ARM_OUT_VERTICAL = 0.5`, `armOut(pose, side): boolean`
  - `REACH_REST = 0.5`, `REACH_FULL = 1.5`, `reachEvidence(pose, side): PenEvidence`
  - `POSE_PAD_REACH = 2.5`
  - `pickActiveArm(pose: PoseLandmark[], last: Side | null): Side | null`
  - `PoseLandmark` gains `z?: number`

- [ ] **Step 1: Write the failing tests (append to `poseGesture.test.ts`)**

```typescript
import {
  armOut,
  pickActiveArm,
  poseRaised,
  reachEvidence,
  shoulderWidth,
  // ...merge into the existing import from "./poseGesture"
} from "./poseGesture";

/** A visitor standing square to the camera: shoulders 0.2 apart at y 0.35,
 *  arms hanging (wrists just above the hip line at y 0.65). */
const standing = (): PoseLandmark[] => {
  const pose: PoseLandmark[] = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.5,
    z: 0,
    visibility: 0,
  }));
  const set = (i: number, x: number, y: number, z = 0) => {
    pose[i] = { x, y, z, visibility: 1 };
  };
  set(POSE.leftShoulder, 0.6, 0.35);
  set(POSE.rightShoulder, 0.4, 0.35);
  set(POSE.leftElbow, 0.65, 0.5);
  set(POSE.rightElbow, 0.35, 0.5);
  set(POSE.leftWrist, 0.65, 0.62);
  set(POSE.rightWrist, 0.35, 0.62);
  set(POSE.leftHip, 0.55, 0.65);
  set(POSE.rightHip, 0.45, 0.65);
  return pose;
};

describe("shoulderWidth", () => {
  it("measures shoulder-to-shoulder distance", () => {
    expect(shoulderWidth(standing())).toBeCloseTo(0.2);
  });

  it("returns 0 when a shoulder is unreadable", () => {
    const pose = standing();
    pose[POSE.leftShoulder] = { x: 0.6, y: 0.35, z: 0, visibility: 0.1 };
    expect(shoulderWidth(pose)).toBe(0);
  });
});

describe("poseRaised", () => {
  it("is true when the wrist is above the shoulder by the margin", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.62, y: 0.2, z: 0, visibility: 1 };
    expect(poseRaised(pose, "left")).toBe(true);
  });

  it("is false for a wrist at shoulder height (a forward reach, not a wake)", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.62, y: 0.34, z: 0, visibility: 1 };
    expect(poseRaised(pose, "left")).toBe(false);
  });
});

describe("armOut", () => {
  it("detects an arm held straight out to the side", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.85, y: 0.37, z: 0, visibility: 1 };
    expect(armOut(pose, "left")).toBe(true);
  });

  it("rejects a raised arm (vertical, not lateral)", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.62, y: 0.2, z: 0, visibility: 1 };
    expect(armOut(pose, "left")).toBe(false);
  });

  it("rejects a forward reach (wrist stays near the shoulder in 2D)", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.63, y: 0.38, z: -0.4, visibility: 1 };
    expect(armOut(pose, "left")).toBe(false);
  });
});

describe("reachEvidence", () => {
  it("drives pen-up when the arm hangs at rest", () => {
    // Wrist z at shoulder depth → reach 0, well below REACH_REST.
    expect(reachEvidence(standing(), "left")).toMatchObject({ fist: 0, open: 1 });
  });

  it("ramps the pen-down drive between rest and full reach", () => {
    const pose = standing();
    // reach = (0 − (−0.2)) / 0.2 = 1.0 → halfway between REACH_REST and REACH_FULL
    pose[POSE.leftWrist] = { x: 0.6, y: 0.4, z: -0.2, visibility: 1 };
    const e = reachEvidence(pose, "left");
    expect(e.fist).toBeCloseTo(0.5);
    expect(e.open).toBe(0);
  });

  it("saturates at full reach", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.6, y: 0.4, z: -0.4, visibility: 1 };
    expect(reachEvidence(pose, "left")).toMatchObject({ fist: 1, open: 0 });
  });

  it("treats a wrist below the hip as firmly pen-up regardless of z", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.6, y: 0.75, z: -0.4, visibility: 1 };
    expect(reachEvidence(pose, "left")).toMatchObject({ fist: 0, open: 1 });
  });

  it("yields no evidence when z is missing or the wrist is unreadable", () => {
    const noZ = standing();
    noZ[POSE.leftWrist] = { x: 0.6, y: 0.4, visibility: 1 };
    expect(reachEvidence(noZ, "left")).toMatchObject({ fist: 0, open: 0 });
    const dim = standing();
    dim[POSE.leftWrist] = { x: 0.6, y: 0.4, z: -0.4, visibility: 0.2 };
    expect(reachEvidence(dim, "left")).toMatchObject({ fist: 0, open: 0 });
  });
});

describe("pickActiveArm", () => {
  it("prefers the higher wrist on first acquisition", () => {
    const pose = standing();
    pose[POSE.rightWrist] = { x: 0.35, y: 0.3, z: 0, visibility: 1 };
    expect(pickActiveArm(pose, null)).toBe("right");
  });

  it("stays sticky to the current arm while its wrist reads", () => {
    const pose = standing();
    pose[POSE.rightWrist] = { x: 0.35, y: 0.3, z: 0, visibility: 1 };
    expect(pickActiveArm(pose, "left")).toBe("left");
  });

  it("falls back to the other visible wrist, and to null when neither reads", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.65, y: 0.62, z: 0, visibility: 0.1 };
    expect(pickActiveArm(pose, "left")).toBe("right");
    pose[POSE.rightWrist] = { x: 0.35, y: 0.62, z: 0, visibility: 0.1 };
    expect(pickActiveArm(pose, "left")).toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- poseGesture`
Expected: FAIL — `shoulderWidth`, `poseRaised`, etc. are not exported.

- [ ] **Step 3: Implement in `poseGesture.ts`**

Extend the `PoseLandmark` interface and `POSE` table, then append:

```typescript
export interface PoseLandmark extends Landmark {
  /** Depth relative to the hip midpoint, roughly x-scale units; more
   *  negative = toward the camera. The noisiest channel MediaPipe ships —
   *  only ever consumed through a PenLatch's hysteresis. */
  z?: number;
  visibility?: number;
}

/** MediaPipe pose landmark indices (the upper-body subset we read). */
export const POSE = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
} as const;

export type Side = "left" | "right";

export function visible(p: PoseLandmark | undefined): p is PoseLandmark {
  return !!p && (p.visibility ?? 1) >= POSE_VISIBILITY_MIN;
}

/** Shoulder-to-shoulder distance in normalized units — the far tier's size
 *  ruler, playing the role hand-span plays up close. 0 when unreadable. */
export function shoulderWidth(pose: PoseLandmark[]): number {
  const l = pose[POSE.leftShoulder];
  const r = pose[POSE.rightShoulder];
  if (!visible(l) || !visible(r)) return 0;
  return Math.hypot(l.x - r.x, l.y - r.y);
}

export function wristOf(pose: PoseLandmark[], side: Side): PoseLandmark {
  return pose[side === "left" ? POSE.leftWrist : POSE.rightWrist];
}

function shoulderOf(pose: PoseLandmark[], side: Side): PoseLandmark {
  return pose[side === "left" ? POSE.leftShoulder : POSE.rightShoulder];
}

function hipOf(pose: PoseLandmark[], side: Side): PoseLandmark {
  return pose[side === "left" ? POSE.leftHip : POSE.rightHip];
}

/** Wrist above the same-side shoulder by this margin = the wake pose. The
 *  margin keeps a forward reach (wrist near shoulder height) from reading
 *  as a wake. */
export const POSE_RAISE_MARGIN = 0.05;

export function poseRaised(pose: PoseLandmark[], side: Side): boolean {
  const wrist = wristOf(pose, side);
  const shoulder = shoulderOf(pose, side);
  if (!visible(wrist) || !visible(shoulder)) return false;
  return wrist.y < shoulder.y - POSE_RAISE_MARGIN;
}

/** Arm straight out to the side — the far-tier brain-cycle verb. Bounds in
 *  shoulder-widths so the pose reads the same at any distance. Geometrically
 *  orthogonal to wake (up) and reach (forward) by construction. */
export const ARM_OUT_LATERAL = 0.8;
export const ARM_OUT_VERTICAL = 0.5;

export function armOut(pose: PoseLandmark[], side: Side): boolean {
  const wrist = wristOf(pose, side);
  const shoulder = shoulderOf(pose, side);
  const sw = shoulderWidth(pose);
  if (sw === 0 || !visible(wrist) || !visible(shoulder)) return false;
  const lateral = Math.abs(wrist.x - shoulder.x) / sw;
  const vertical = Math.abs(wrist.y - shoulder.y) / sw;
  return lateral > ARM_OUT_LATERAL && vertical < ARM_OUT_VERTICAL;
}

/** Reach ratio ((shoulder.z − wrist.z) / shoulderWidth) where the pen-down
 *  drive begins, and where it saturates. THE on-site tuning pair: watch the
 *  reach bar in the debug overlay and move these until a relaxed arm sits
 *  at zero and a comfortable reach pins the bar. */
export const REACH_REST = 0.5;
export const REACH_FULL = 1.5;

/**
 * Reach-toward-the-screen as pen evidence, shaped for PenLatch (`fist`
 * = pen-down drive, `open` = pen-up drive, same convention as
 * thumbEvidence). Unreadable input yields NO_EVIDENCE so the latch's decay
 * lifts the pen on its own — uncertainty must stay silent.
 */
export function reachEvidence(pose: PoseLandmark[], side: Side): PenEvidence {
  const wrist = wristOf(pose, side);
  const shoulder = shoulderOf(pose, side);
  const sw = shoulderWidth(pose);
  if (sw === 0 || !visible(wrist) || !visible(shoulder)) return NO_EVIDENCE;
  if (wrist.z === undefined || shoulder.z === undefined) return NO_EVIDENCE;
  // An arm hanging past the hip can't be a reach, whatever z claims.
  const hip = hipOf(pose, side);
  if (visible(hip) && wrist.y > hip.y) return { fist: 0, open: 1 };
  const reach = (shoulder.z - wrist.z) / sw;
  const t = (reach - REACH_REST) / (REACH_FULL - REACH_REST);
  if (t >= 1) return { fist: 1, open: 0 };
  if (t > 0) return { fist: t, open: 0 };
  return { fist: 0, open: Math.min(1, (REACH_REST - reach) / REACH_REST) };
}

/** Pad box width in shoulder-widths — PadMapper's `reach` parameter when
 *  the far tier feeds it wrist + shoulderWidth instead of palm + hand-span,
 *  so drawing needs the same comfortable arm travel at any distance. */
export const POSE_PAD_REACH = 2.5;

/** Sticky active-arm choice, mirroring pickPrimaryHand: keep the current
 *  arm while its wrist reads; otherwise the higher visible wrist wins.
 *  (Pose sides are stable identities — no distance-matching needed.) */
export function pickActiveArm(pose: PoseLandmark[], last: Side | null): Side | null {
  const leftOk = visible(pose[POSE.leftWrist]);
  const rightOk = visible(pose[POSE.rightWrist]);
  if (last === "left" && leftOk) return "left";
  if (last === "right" && rightOk) return "right";
  if (leftOk && rightOk) {
    return pose[POSE.leftWrist].y <= pose[POSE.rightWrist].y ? "left" : "right";
  }
  if (leftOk) return "left";
  if (rightOk) return "right";
  return null;
}
```

Also update the module's import line:

```typescript
import { NO_EVIDENCE, type Landmark, type PenEvidence } from "./handGesture";
```

(The Task 1 `PoseLandmark`/`POSE` declarations are replaced by these extended ones — one declaration each, not duplicates.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test -- poseGesture`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gesture/poseGesture.ts src/gesture/poseGesture.test.ts
git commit -m "feat: far-tier pose primitives — reach, arm-out, raise, active arm"
```

---

### Task 3: The far tier in `FramePipeline` (TDD)

**Files:**
- Modify: `src/gesture/framePipeline.ts`
- Test: `src/gesture/framePipeline.test.ts`

**Interfaces:**
- Consumes (from `./poseGesture`): everything Task 2 produced.
- Produces:
  - `export type DrivingTier = "close" | "far" | null` (from `framePipeline.ts`)
  - `PipelineFrameOutput` gains `tier: DrivingTier` — Task 4's loops read it to pick strides.
  - `HandDebugFrame` gains `tier: DrivingTier`, `bodyPose: PoseLandmark[] | null`, `farReach: { side: Side; charge: number } | null` — Task 5 renders them.
- Tier semantics (load-bearing for Task 4): `"close"` whenever the close tier owns the state **including its grace window**; `"far"` while the far tier emits (including its grace); `null` otherwise. Hand stride only relaxes when tier ≠ `"close"`, so the close tier always releases at full hand rate.

- [ ] **Step 1: Write the failing tests (append to `framePipeline.test.ts`)**

Add pose fixtures near the existing helpers (note `crossedPose()` already exists for the close-tier ✕ test — these are full bodies for the far tier):

```typescript
import { POSE, type PoseLandmark } from "./poseGesture";

/** Full standing body for far-tier tests: shoulders 0.2 apart, arms down. */
const standingBody = (): PoseLandmark[] => {
  const pose: PoseLandmark[] = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.5,
    z: 0,
    visibility: 0,
  }));
  const set = (i: number, x: number, y: number, z = 0) => {
    pose[i] = { x, y, z, visibility: 1 };
  };
  set(POSE.leftShoulder, 0.6, 0.35);
  set(POSE.rightShoulder, 0.4, 0.35);
  set(POSE.leftElbow, 0.65, 0.5);
  set(POSE.rightElbow, 0.35, 0.5);
  set(POSE.leftWrist, 0.65, 0.62);
  set(POSE.rightWrist, 0.35, 0.62);
  set(POSE.leftHip, 0.55, 0.65);
  set(POSE.rightHip, 0.45, 0.65);
  return pose;
};

const reachingBody = (): PoseLandmark[] => {
  const pose = standingBody();
  pose[POSE.leftWrist] = { x: 0.6, y: 0.4, z: -0.4, visibility: 1 };
  return pose;
};

const raisedBody = (): PoseLandmark[] => {
  const pose = standingBody();
  pose[POSE.leftWrist] = { x: 0.62, y: 0.2, z: 0, visibility: 1 };
  return pose;
};

const armOutBody = (): PoseLandmark[] => {
  const pose = standingBody();
  pose[POSE.leftWrist] = { x: 0.85, y: 0.37, z: 0, visibility: 1 };
  return pose;
};

const crossedBody = (): PoseLandmark[] => {
  const pose = standingBody();
  pose[POSE.leftElbow] = { x: 0.6, y: 0.5, z: 0, visibility: 1 };
  pose[POSE.leftWrist] = { x: 0.4, y: 0.4, z: 0, visibility: 1 };
  pose[POSE.rightElbow] = { x: 0.4, y: 0.5, z: 0, visibility: 1 };
  pose[POSE.rightWrist] = { x: 0.6, y: 0.4, z: 0, visibility: 1 };
  return pose;
};
```

And the test block:

```typescript
describe("FramePipeline far tier", () => {
  it("reports presence with the pen up from body pose alone", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 3, { poseRan: true, bodyPose: standingBody() });
    expect(out.state).toMatchObject({ present: true, pose: "open" });
    expect(out.tier).toBe("far");
  });

  it("latches the pen down after sustained reach", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 8, { poseRan: true, bodyPose: reachingBody() });
    expect(out.state).toMatchObject({ present: true, pose: "fist" });
  });

  it("reports raised for a wrist held above the shoulder", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 3, { poseRan: true, bodyPose: raisedBody() });
    expect(out.state).toMatchObject({ raised: true });
  });

  it("latches the brain-cycle verb for an arm held out to the side", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 8, { poseRan: true, bodyPose: armOutBody() });
    expect(out.state).toMatchObject({ thumbsUp: true });
  });

  it("reports the ✕ from crossed forearms", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 3, { poseRan: true, bodyPose: crossedBody() });
    expect(out.state).toMatchObject({ crossed: true });
  });

  it("moves the cursor with the active wrist (mirrored)", () => {
    const pipeline = new FramePipeline();
    run(pipeline, 5, { poseRan: true, bodyPose: raisedBody() });
    const moved = raisedBody();
    moved[POSE.leftWrist] = { x: 0.72, y: 0.2, z: 0, visibility: 1 };
    const out = run(pipeline, 30, { poseRan: true, bodyPose: moved }, 5 * 33);
    // +0.1 wrist travel in a 0.5-wide pad box → −0.2 in mirrored pad space
    expect(out.state!.x).toBeLessThan(0.45);
  });

  it("a close hand takes the state over from the far tier", () => {
    const pipeline = new FramePipeline();
    run(pipeline, 5, { poseRan: true, bodyPose: reachingBody() });
    const out = run(
      pipeline,
      3,
      {
        allHands: [makeHand(0.5, 0.6)],
        allGestures: [OPEN],
        poseRan: true,
        bodyPose: reachingBody(),
      },
      5 * 33
    );
    expect(out.tier).toBe("close");
    expect(out.state).toMatchObject({ present: true, pose: "open" });
  });

  it("takes over only after the close tier fully releases", () => {
    const pipeline = new FramePipeline();
    run(pipeline, 3, { allHands: [makeHand(0.5, 0.6)], allGestures: [OPEN] });
    // Hand gone but body visible: the close tier's grace window holds.
    const during = run(
      pipeline,
      ABSENCE_GRACE_FRAMES,
      { poseRan: true, bodyPose: standingBody() },
      3 * 33
    );
    expect(during.tier).toBe("close");
    // One more frame releases close (present:false), then far engages.
    const released = pipeline.update(
      frame({ poseRan: true, bodyPose: standingBody(), nowMs: 999 })
    );
    expect(released.state).toMatchObject({ present: false });
    const far = run(pipeline, 3, { poseRan: true, bodyPose: standingBody() }, 1100);
    expect(far.tier).toBe("far");
    expect(far.state).toMatchObject({ present: true });
  });

  it("bleeds out and reports absence when the body disappears", () => {
    const pipeline = new FramePipeline();
    run(pipeline, 8, { poseRan: true, bodyPose: reachingBody() });
    const out = run(
      pipeline,
      ABSENCE_GRACE_FRAMES + 1,
      { poseRan: true, bodyPose: undefined },
      8 * 33
    );
    expect(out.state).toMatchObject({ present: false });
    expect(pipeline.update(frame({ poseRan: true, nowMs: 9999 })).state).toBeNull();
  });

  it("exposes tier, body pose, and reach charge to the debug overlay", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 3, {
      poseRan: true,
      bodyPose: reachingBody(),
      wantDebug: true,
    });
    expect(out.debug?.tier).toBe("far");
    expect(out.debug?.bodyPose).not.toBeNull();
    expect(out.debug?.farReach?.side).toBe("left");
    expect(out.debug?.farReach?.charge).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `npm test -- framePipeline`
Expected: FAIL — `out.tier` undefined, far-tier states null. **All pre-existing tests must still pass at this point** (the fixtures don't touch them).

- [ ] **Step 3: Implement the far tier in `framePipeline.ts`**

3a. Imports — extend the `poseGesture` import:

```typescript
import {
  armOut,
  forearmsCrossed,
  pickActiveArm,
  POSE,
  POSE_PAD_REACH,
  poseRaised,
  reachEvidence,
  shoulderWidth,
  wristOf,
  type PoseLandmark,
  type Side,
} from "./poseGesture";
```

3b. Types — add after `GestureState`:

```typescript
/** Which signal source produced the frame's state. "close" includes the
 *  close tier's absence-grace window — the far tier may only engage once
 *  the close tier has fully released. */
export type DrivingTier = "close" | "far" | null;
```

Extend `HandDebugFrame` (after `crossed: boolean;`):

```typescript
  tier: DrivingTier;
  /** Held body pose, for skeleton drawing. */
  bodyPose: PoseLandmark[] | null;
  /** Far-tier reach latch readout (the on-site tuning instrument). */
  farReach: { side: Side; charge: number } | null;
```

Extend `PipelineFrameOutput`:

```typescript
export interface PipelineFrameOutput {
  /** State to report to the app — null when nothing should be emitted. */
  state: GestureState | null;
  debug: HandDebugFrame | null;
  tier: DrivingTier;
}
```

3c. Class fields — add below the existing private fields:

```typescript
  // ---- far tier: separate instances so tier handoff can't smear state ----
  private readonly farLatch = new PenLatch();
  private readonly farArmOutLatch = new PenLatch({
    press: 0.3,
    release: 0.5,
    decay: 0.06,
    downAt: 0.6,
    upAt: 0.25,
  });
  private readonly farMapper = new PadMapper(POSE_PAD_REACH);
  private readonly farFilterX = new OneEuroFilter();
  private readonly farFilterY = new OneEuroFilter();
  private farSmoothSw = 0;
  private farActive = false;
  private farMissed = 0;
  private activeArm: Side | null = null;
  private tier: DrivingTier = null;
```

And the reset helper (next to `emit`):

```typescript
  private resetFar(): void {
    this.farActive = false;
    this.farMissed = 0;
    this.activeArm = null;
    this.farLatch.reset();
    this.farArmOutLatch.reset();
    this.farMapper.reset();
    this.farFilterX.reset();
    this.farFilterY.reset();
    this.farSmoothSw = 0;
  }
```

3d. Debug closure — inside `buildDebug`, add the three new fields to the object literal (after `crossed,`):

```typescript
            tier: this.tier,
            bodyPose: bodyPose ?? null,
            farReach: this.activeArm
              ? { side: this.activeArm, charge: this.farLatch.level() }
              : null,
```

3e. The far branch. Replace the current early-out at the top of the no-hands block:

```typescript
    if (hands.length === 0) {
      if (!this.hadHand) {
        buildDebug?.();
        return { state: null, debug };
      }
```

with a far-tier branch (the close tier's grace below it is untouched):

```typescript
    if (hands.length === 0) {
      if (!this.hadHand) {
        // Close tier idle → the far tier reads the whole vocabulary from
        // body pose. Same GestureState out, so nothing downstream can tell.
        const sw = bodyPose ? shoulderWidth(bodyPose) : 0;
        const arm = bodyPose && sw > 0 ? pickActiveArm(bodyPose, this.activeArm) : null;
        if (arm === null) {
          if (!this.farActive) {
            this.tier = null;
            buildDebug?.();
            return { state: null, debug, tier: null };
          }
          // Far dropout: same shape as a vanished hand — bleed the latches,
          // hold the cursor, report absence only after the grace window.
          const pose = this.farLatch.update(NO_EVIDENCE);
          const thumbsUp = this.farArmOutLatch.update(NO_EVIDENCE) === "fist";
          if (++this.farMissed > ABSENCE_GRACE_FRAMES) {
            this.resetFar();
            this.lastFrameTime = 0;
            this.tier = null;
            const state = this.emit({
              present: false,
              x: this.smoothX,
              y: this.smoothY,
              pose: "open",
              raised: false,
              crossed: false,
              thumbsUp: false,
            });
            buildDebug?.();
            return { state, debug, tier: null };
          }
          this.tier = "far";
          const state = this.emit({
            present: true,
            x: this.smoothX,
            y: this.smoothY,
            pose,
            raised: false,
            crossed,
            thumbsUp,
          });
          buildDebug?.();
          return { state, debug, tier: "far" };
        }
        this.farMissed = 0;
        this.activeArm = arm;
        this.tier = "far";
        const wrist = wristOf(bodyPose!, arm);
        // Smooth the ruler like the close tier smooths hand-span.
        this.farSmoothSw =
          this.farSmoothSw === 0 ? sw : this.farSmoothSw + (sw - this.farSmoothSw) * 0.1;
        const mapped = this.farMapper.update(wrist, this.farSmoothSw);
        if (!this.farActive) {
          this.farFilterX.reset();
          this.farFilterY.reset();
          this.farActive = true;
        }
        const dt = this.lastFrameTime ? (nowMs - this.lastFrameTime) / 1000 : NOMINAL_DT;
        this.lastFrameTime = nowMs;
        this.smoothX = this.farFilterX.filter(mapped.x, dt);
        this.smoothY = this.farFilterY.filter(mapped.y, dt);
        const pose = this.farLatch.update(reachEvidence(bodyPose!, arm));
        const thumbsUp =
          this.farArmOutLatch.update(
            armOut(bodyPose!, arm) ? { fist: 1, open: 0 } : { fist: 0, open: 1 }
          ) === "fist";
        const state = this.emit({
          present: true,
          x: this.smoothX,
          y: this.smoothY,
          pose,
          raised: poseRaised(bodyPose!, arm),
          crossed,
          thumbsUp,
        });
        buildDebug?.();
        return { state, debug, tier: "far" };
      }
```

3f. Tier bookkeeping in the existing paths:

- In the close-grace branch (`hands.length === 0 && this.hadHand`): set `this.tier = "close";` before `buildDebug` in the held-state path, and `this.tier = null;` in the `missedFrames > ABSENCE_GRACE_FRAMES` release path. Add `tier: this.tier` to both `return` statements.
- In the close-active path (a hand is present, after `this.missedFrames = 0;`): add

```typescript
    if (this.farActive || this.activeArm) this.resetFar();
    this.tier = "close";
```

and `tier: "close"` to its return.

Every `return` in `update()` now carries `tier` — the compiler enforces completeness once `PipelineFrameOutput` has the field.

- [ ] **Step 4: Run the full suite**

Run: `npm test`
Expected: PASS — all far-tier tests green AND every pre-existing test untouched and green. If any pre-existing test fails, the close tier changed: fix the pipeline, never the test.

- [ ] **Step 5: Commit**

```bash
git add src/gesture/framePipeline.ts src/gesture/framePipeline.test.ts
git commit -m "feat: far tier — full gesture vocabulary from body pose"
```

---

### Task 4: Tier-driven inference scheduling

**Files:**
- Modify: `src/gesture/visionTasks.ts` (STRIDES, init protocol, optional gestureMs)
- Modify: `src/gesture/visionWorker.ts` (stride countdowns)
- Modify: `src/gesture/handTracker.ts` (inline path countdowns, init message, perf mark guard)

**Interfaces:**
- Consumes: `PipelineFrameOutput.tier` from Task 3; `type DrivingTier` from `./framePipeline`.
- Produces:
  - `visionTasks.ts` exports `STRIDES = { close: { hand: 1, pose: 6 }, far: { hand: 4, pose: 1 } } as const` and **removes** `POSE_FRAME_STRIDE`.
  - `VisionWorkerInit` loses `poseStride`.
  - `VisionWorkerResponse` "state" variant: `gestureMs?: number` (optional — hand inference may be skipped on a strided frame).

- [ ] **Step 1: Update `visionTasks.ts`**

Replace the `POSE_FRAME_STRIDE` constant and its comment with:

```typescript
/** Per-tier inference strides. Close tier: hands per-frame (a stroking
 *  fist is fast and blur-sensitive), pose at ~10Hz — it only feeds the ✕
 *  there, which must be HELD 800ms. Far tier (or nobody in frame): the
 *  arrangement inverts — pose feeds the cursor so it runs per-frame, and
 *  the hand model's only job is noticing someone stepping close. */
export const STRIDES = {
  close: { hand: 1, pose: 6 },
  far: { hand: 4, pose: 1 },
} as const;
```

In `VisionWorkerInit`, delete the `poseStride: number;` line. In the "state" variant of `VisionWorkerResponse`, change `gestureMs: number;` to `gestureMs?: number;` with the comment `/** Absent when the hand model was strided out this frame. */`.

- [ ] **Step 2: Update `visionWorker.ts`**

Replace the module-level `let poseStride = 6; let poseFrame = 0;` with:

```typescript
let tier: DrivingTier = null;
let handCountdown = 0;
let poseCountdown = 0;
```

(import `type DrivingTier` from `./framePipeline`; drop `poseStride = msg.poseStride;` from the init handler). Replace the frame handler's inference section (from `const gestureStart = ...` through `bitmap.close();`) with countdown-gated runs — countdowns, not modulo, so a stride change takes effect cleanly mid-stream:

```typescript
      const strides = tier === "close" ? STRIDES.close : STRIDES.far;
      let allHands: PipelineFrameInput["allHands"] = [];
      let allGestures: PipelineFrameInput["allGestures"] = [];
      let gestureMs: number | undefined;
      if (--handCountdown <= 0) {
        handCountdown = strides.hand;
        const gestureStart = performance.now();
        const result = tasks.recognizer.recognizeForVideo(bitmap, t);
        gestureMs = performance.now() - gestureStart;
        allHands = result.landmarks ?? [];
        allGestures = result.gestures ?? [];
      }
      let poseRan = false;
      let bodyPose;
      let poseMs: number | undefined;
      if (tasks.poseLandmarker && --poseCountdown <= 0) {
        poseCountdown = strides.pose;
        poseRan = true;
        const poseStart = performance.now();
        bodyPose = tasks.poseLandmarker.detectForVideo(bitmap, t).landmarks?.[0];
        poseMs = performance.now() - poseStart;
      }
      bitmap.close();
```

then after `pipeline.update(...)`: `tier = out.tier;` before posting (the post itself is unchanged — `gestureMs` is already the local). Import `STRIDES` from `./visionTasks` and `type PipelineFrameInput` from `./framePipeline`.

- [ ] **Step 3: Update `handTracker.ts`**

- Remove `POSE_FRAME_STRIDE` from the `./visionTasks` import; the worker init message becomes `worker.postMessage({ type: "init", ...assets });`.
- Guard the perf mark in the worker response handler: `if (msg.gestureMs !== undefined) perf.recordMark("gesture", msg.gestureMs);`.
- In `startInlinePath`, replace `let poseFrame = 0;` with the same three locals as the worker (`tier`/`handCountdown`/`poseCountdown`) and mirror the countdown-gated inference block (using `video` instead of `bitmap`, `now` instead of `t`, and `perf.recordMark("gesture" | "pose", ...)` inline where measured), then `tier = out.tier;` after `pipeline.update(...)`. Import `STRIDES` from `./visionTasks` and `type DrivingTier, type PipelineFrameInput` from `./framePipeline`.

- [ ] **Step 4: Verify**

Run: `npm test` → PASS (pipeline behavior unchanged; this task is scheduling glue, which has no unit tests by existing convention).
Run: `npm run typecheck` → clean — this is the real check here: the removed `poseStride`/`POSE_FRAME_STRIDE` must leave zero dangling references.
Run: `npm run lint` → clean.

- [ ] **Step 5: Commit**

```bash
git add src/gesture/visionTasks.ts src/gesture/visionWorker.ts src/gesture/handTracker.ts
git commit -m "feat: tier-driven inference strides — pose per-frame when it drives"
```

---

### Task 5: Debug overlay — pose skeleton, reach bar, tier

**Files:**
- Modify: `src/hud/DebugPanel.tsx`

**Interfaces:**
- Consumes: `HandDebugFrame.tier / .bodyPose / .farReach` from Task 3; `POSE`, `POSE_VISIBILITY_MIN` from `./poseGesture`.
- Produces: nothing downstream — this is the on-site tuning instrument.

- [ ] **Step 1: Add the skeleton drawing**

Import: `import { POSE, POSE_VISIBILITY_MIN } from "../gesture/poseGesture";` and add beside `BONES`:

```typescript
/** Upper-body pose segments — the far tier's raw signal. */
const POSE_BONES: readonly [number, number][] = [
  [POSE.leftShoulder, POSE.rightShoulder],
  [POSE.leftShoulder, POSE.leftElbow],
  [POSE.leftElbow, POSE.leftWrist],
  [POSE.rightShoulder, POSE.rightElbow],
  [POSE.rightElbow, POSE.rightWrist],
];
```

Inside `draw`, immediately BEFORE the existing forearms block (so the ✕ coloring still overdraws the forearm segments):

```typescript
      // Upper-body skeleton — bright when the far tier is driving, faint
      // otherwise; segment alpha tracks landmark visibility so a flaky
      // reading is visibly flaky.
      if (frame.bodyPose) {
        ctx.lineWidth = 2.5;
        for (const [a, b] of POSE_BONES) {
          const pa = frame.bodyPose[a];
          const pb = frame.bodyPose[b];
          if (!pa || !pb) continue;
          const vis = Math.min(pa.visibility ?? 1, pb.visibility ?? 1);
          if (vis < POSE_VISIBILITY_MIN) continue;
          ctx.strokeStyle =
            frame.tier === "far"
              ? `rgba(125, 255, 154, ${vis.toFixed(2)})`
              : `rgba(141, 151, 180, ${(vis * 0.7).toFixed(2)})`;
          ctx.beginPath();
          ctx.moveTo(mx(pa.x), pa.y * VIEW_H);
          ctx.lineTo(mx(pb.x), pb.y * VIEW_H);
          ctx.stroke();
        }
      }
```

- [ ] **Step 2: Extend the status line**

Replace the `status.textContent = [...]` array with:

```typescript
      const filled = Math.round(frame.charge * 10);
      const reach = frame.farReach;
      const reachFilled = reach ? Math.round(reach.charge * 10) : 0;
      status.textContent = [
        `tier: ${frame.tier ?? "—"}`,
        `hands: ${frame.hands.length}`,
        `pen ${"█".repeat(filled)}${"░".repeat(10 - filled)} ${frame.charge.toFixed(2)}`,
        reach
          ? `reach(${reach.side}) ${"█".repeat(reachFilled)}${"░".repeat(10 - reachFilled)} ${reach.charge.toFixed(2)}`
          : "reach —",
        `✕: ${frame.crossed}${frame.forearms ? ` (arms ${frame.forearms.crossed ? "crossed" : "apart"})` : " (no body pose)"}`,
        s
          ? `emitted: ${s.present ? `${s.pose}${s.raised ? " · raised" : ""}${s.thumbsUp ? " · 👍" : ""} @ ${s.x.toFixed(2)},${s.y.toFixed(2)}` : "absent"}`
          : "emitted: —",
      ].join("   |   ");
```

(The `const filled` line already exists — keep one copy.)

- [ ] **Step 3: Verify**

Run: `npm run typecheck && npm run lint && npm test` → all clean/green.

- [ ] **Step 4: Commit**

```bash
git add src/hud/DebugPanel.tsx
git commit -m "feat: debug overlay — pose skeleton, reach bar, driving tier"
```

---

### Task 6: End-to-end verification in the running app

**Files:** none (verification only; tuning edits go to the named constants in `poseGesture.ts` if obviously needed).

- [ ] **Step 1: Full gate**

Run: `npm test && npm run typecheck && npm run lint`
Expected: all pass. Test count ≥ 233 + ~25 new.

- [ ] **Step 2: Manual far-tier check**

Run: `npm run dev`, open the app, press `G` for the overlay, then:

1. Sit at normal desk distance with hands visible → status shows `tier: close`; fist-draw and 👍 behave exactly as before this branch.
2. Hide both hands below the desk / step back so hands drop out but your torso is framed → after the grace window, `tier: far`, skeleton turns bright, cursor follows your wrist.
3. Reach toward the screen → reach bar fills, pen goes down; relax → bar drains, pen lifts.
4. Hold an arm straight out to the side ~1.5s → brain cycles.
5. Raise a hand overhead → `raised` appears in the emitted line (wake path).
6. Cross forearms → ✕ fires.
7. Watch the perf readout: `pose` mark should tick every frame in far tier without pushing frame p95 over budget.

Note observed reach-bar behavior in the commit message or a follow-up note — `REACH_REST`/`REACH_FULL` are expected to need on-site adjustment at the real footron distance; desk verification only needs to show the ramp responding in the right direction.

- [ ] **Step 3: Final commit (if any tuning edits were made)**

```bash
git add -A src/gesture
git commit -m "chore: desk-tuning pass on far-tier reach constants"
```

---

## Self-Review Notes

- Spec coverage: module layout → T1/T2; verb table + evidence + mapper + active arm → T2/T3; tier hysteresis → T3 (semantics pinned in the Interfaces block); scheduling flip + protocol change → T4; overlay → T5; error handling (missing pose model → far tier simply never engages; visibility → NO_EVIDENCE) → T2/T3; "existing tests unmodified" → enforced in T3 Step 4.
- Spec's `PosePadMapper` is realized as `PadMapper` + `POSE_PAD_REACH` (identical contract, zero new code — DRY); recorded here as a deliberate deviation in name only.
- Type consistency: `DrivingTier`/`tier` (T3→T4/T5), `farReach.{side,charge}` (T3→T5), `STRIDES.{close,far}.{hand,pose}` (T4 both loops), `gestureMs?` guard (T4).
