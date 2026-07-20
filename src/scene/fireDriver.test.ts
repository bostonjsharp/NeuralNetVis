import { describe, expect, it } from "vitest";
import type { ForwardResult } from "../nn/inference";
import { createStagedPass } from "../nn/stagedPass";
import type { Net } from "../nn/weights";
import { makeFireTimeline } from "./fire";
import { createFireDriver } from "./fireDriver";

/** Same hand-computed 2→3→2 golden network as stagedPass.test.ts. */
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

/** fire.layerPop for makeFireTimeline(2): [1.3, 2.3] (see fire.ts's constants). */
const fire = makeFireTimeline(2);
const LAYER_COUNT = 2; // destination layers: hidden, output

function harness() {
  const pass = createStagedPass(toyNet, input);
  const layers: { layerIndex: number; normalized: number[]; source: number[] }[] = [];
  let completeCount = 0;
  let completedResult: ForwardResult | null = null;
  const driver = createFireDriver(fire, pass, LAYER_COUNT, {
    onLayer(layerIndex, normalizedValues, sourceActivations) {
      layers.push({
        layerIndex,
        normalized: Array.from(normalizedValues),
        source: Array.from(sourceActivations),
      });
    },
    onComplete(r) {
      completeCount++;
      completedResult = r;
    },
  });
  return { pass, driver, layers, getCompleteCount: () => completeCount, getResult: () => completedResult };
}

describe("createFireDriver", () => {
  it("computes nothing before the first layer pops", () => {
    const { pass, driver, layers, getCompleteCount } = harness();
    driver.advance(1.0); // before layerPop[0] = 1.3
    expect(pass.activations).toHaveLength(1);
    expect(layers).toHaveLength(0);
    expect(getCompleteCount()).toBe(0);
  });

  it("delivers onComplete exactly once, with the correct result", () => {
    const { driver, getCompleteCount, getResult } = harness();
    driver.advance(1.5); // past layerPop[0] only
    expect(getCompleteCount()).toBe(0);
    driver.advance(2.5); // past layerPop[1] too
    expect(getCompleteCount()).toBe(1);
    const result = getResult()!;
    expect(result.logits[0]).toBeCloseTo(-0.5, 5);
    expect(result.logits[1]).toBeCloseTo(2.5, 5);
    expect(result.probs[1]).toBeCloseTo(0.95257, 4);
    expect(result.argmax).toBe(1);
    driver.advance(10); // further advances must not re-deliver
    expect(getCompleteCount()).toBe(1);
  });

  it("catches up every due layer, in order, in one call (stall catch-up)", () => {
    const { pass, driver, layers, getCompleteCount } = harness();
    driver.advance(100); // far past the end — a stalled-tab rAF gap
    expect(pass.activations).toHaveLength(3); // input + 2 computed layers
    expect(layers.map((l) => l.layerIndex)).toEqual([0, 1]);
    expect(getCompleteCount()).toBe(1);
  });

  it("is a no-op if advance() re-enters from inside onComplete itself", () => {
    const pass = createStagedPass(toyNet, input);
    const layers: number[] = [];
    let completeCount = 0;
    const driver: ReturnType<typeof createFireDriver> = createFireDriver(fire, pass, LAYER_COUNT, {
      onLayer(layerIndex) {
        layers.push(layerIndex);
      },
      onComplete() {
        completeCount++;
        // Regression: a completed driver must be permanently inert, even
        // when re-entered synchronously from its own onComplete with a
        // stale, larger `due` — it must never step or deliver again.
        driver.advance(1000);
      },
    });
    driver.advance(1000);
    expect(completeCount).toBe(1);
    expect(layers).toEqual([0, 1]);
    expect(pass.activations).toHaveLength(3);
  });

  it("normalizes onLayer values by each layer's own max (output uses probs)", () => {
    const { driver, layers } = harness();
    driver.advance(100);
    // Hidden layer (index 0): post-ReLU activations [0, 2, 1], own max 2.
    expect(layers[0].layerIndex).toBe(0);
    expect(layers[0].source).toEqual([0, 2, 1]);
    expect(layers[0].normalized).toEqual([0, 1, 0.5]);
    // Output layer (index 1): source is the raw logits, but normalization
    // must follow probs (per the finding), not the logits themselves.
    expect(layers[1].layerIndex).toBe(1);
    const logits = layers[1].source;
    expect(logits[0]).toBeCloseTo(-0.5, 5);
    expect(logits[1]).toBeCloseTo(2.5, 5);
    // probs[1] ≈ 0.95257 is the layer's own max — normalizing by it maps
    // the winning slot to exactly 1, which a raw-logit normalization (max
    // 2.5) would not.
    expect(layers[1].normalized[1]).toBe(1);
    expect(layers[1].normalized[0]).toBeCloseTo((1 - 0.95257) / 0.95257, 4);
  });
});
