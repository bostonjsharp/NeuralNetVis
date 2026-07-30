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
