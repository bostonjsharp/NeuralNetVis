import { describe, expect, it } from "vitest";
import {
  AmbientState,
  breathe,
  cameraDrift,
  DRIFT_POS_AMP,
  DRIFT_TARGET_AMP,
  DUCK_ATTACK_S,
  DUCK_FLOOR,
  DUCK_RELEASE_S,
  duckEnvelope,
  INK_THRESHOLD,
  inkDelta,
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
