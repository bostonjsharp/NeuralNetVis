import { describe, expect, it } from "vitest";
import { forwardPass, softmax } from "./inference";
import type { Net } from "./weights";

/**
 * Hand-computed 2→3→2 golden network:
 *   z1 = W1·x + b1 = [-1, 2, 1]  →  ReLU  →  a1 = [0, 2, 1]
 *   logits = W2·a1 + b2 = [-0.5, 2.5]
 *   softmax → [0.04743, 0.95257]
 */
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

describe("forwardPass", () => {
  it("matches the hand-computed golden vectors", () => {
    const result = forwardPass(toyNet, Float32Array.from([1, 2]));
    expect(Array.from(result.activations[1])).toEqual([0, 2, 1]);
    expect(result.logits[0]).toBeCloseTo(-0.5, 5);
    expect(result.logits[1]).toBeCloseTo(2.5, 5);
    expect(result.probs[0]).toBeCloseTo(0.04743, 4);
    expect(result.probs[1]).toBeCloseTo(0.95257, 4);
    expect(result.argmax).toBe(1);
  });

  it("returns one activation array per network layer, input first", () => {
    const result = forwardPass(toyNet, Float32Array.from([1, 2]));
    expect(result.activations.map((a) => a.length)).toEqual([2, 3, 2]);
    expect(Array.from(result.activations[0])).toEqual([1, 2]);
  });

  it("rejects wrongly sized input", () => {
    expect(() => forwardPass(toyNet, new Float32Array(3))).toThrow(/expected 2/);
  });
});

describe("softmax", () => {
  it("sums to 1", () => {
    const probs = softmax(Float32Array.from([0.1, -2, 3, 0]));
    expect(probs.reduce((s, p) => s + p, 0)).toBeCloseTo(1, 6);
  });

  it("is stable for large logits", () => {
    const probs = softmax(Float32Array.from([1000, 1001, 999]));
    expect(Number.isFinite(probs[0])).toBe(true);
    expect(probs.reduce((s, p) => s + p, 0)).toBeCloseTo(1, 6);
    expect(probs[1]).toBeGreaterThan(probs[0]);
  });

  it("is uniform for equal logits", () => {
    const probs = softmax(Float32Array.from([2, 2, 2, 2]));
    for (const p of probs) expect(p).toBeCloseTo(0.25, 6);
  });
});
