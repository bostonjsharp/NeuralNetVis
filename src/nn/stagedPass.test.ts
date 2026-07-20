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
