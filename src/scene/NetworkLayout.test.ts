import { describe, expect, it } from "vitest";
import weightsJson from "../assets/weights.json";
import { loadNet, type WeightsJson } from "../nn/weights";
import { buildLayout, INPUT_TOP_K, maxAbsWeight, topKIndices } from "./NetworkLayout";

describe("topKIndices", () => {
  it("picks the largest magnitudes, strongest first", () => {
    const values = Float32Array.from([0.1, -5, 2, -0.5, 3]);
    expect(Array.from(topKIndices(values, 3))).toEqual([1, 4, 2]);
  });
});

describe("buildLayout", () => {
  const net = loadNet(weightsJson as WeightsJson);
  const layout = buildLayout(net);

  it("positions every pixel and neuron", () => {
    expect(layout.inputPositions).toHaveLength(784 * 3);
    expect(layout.layerPositions[0]).toHaveLength(16 * 3);
    expect(layout.layerPositions[1]).toHaveLength(16 * 3);
    expect(layout.layerPositions[2]).toHaveLength(10 * 3);
  });

  it("flows left to right", () => {
    const meanInputX =
      Array.from({ length: 784 }, (_, i) => layout.inputPositions[i * 3]).reduce(
        (a, b) => a + b
      ) / 784;
    expect(meanInputX).toBeLessThan(layout.layerPositions[0][0]);
    expect(layout.layerPositions[0][0]).toBeLessThan(layout.layerPositions[1][0]);
    expect(layout.layerPositions[1][0]).toBeLessThan(layout.layerPositions[2][0]);
  });

  it("renders top-K input edges and all deeper edges", () => {
    expect(layout.edges[0].count).toBe(16 * INPUT_TOP_K);
    expect(layout.edges[1].count).toBe(16 * 16);
    expect(layout.edges[2].count).toBe(16 * 10);
  });

  it("keeps edge endpoints in range", () => {
    const [s0, s1, s2] = layout.edges;
    for (let i = 0; i < s0.count; i++) {
      expect(s0.from[i]).toBeLessThan(784);
      expect(s0.to[i]).toBeLessThan(16);
    }
    for (let i = 0; i < s1.count; i++) {
      expect(s1.from[i]).toBeLessThan(16);
      expect(s1.to[i]).toBeLessThan(16);
    }
    for (let i = 0; i < s2.count; i++) {
      expect(s2.from[i]).toBeLessThan(16);
      expect(s2.to[i]).toBeLessThan(10);
    }
  });

  it("stage-0 edges really are each neuron's strongest weights", () => {
    const { from, to, weight } = layout.edges[0];
    const W = net.layers[0].W;
    // For neuron 0, the weakest kept edge must be ≥ every dropped weight
    const kept = new Set<number>();
    let weakestKept = Infinity;
    for (let e = 0; e < layout.edges[0].count; e++) {
      if (to[e] !== 0) continue;
      kept.add(from[e]);
      weakestKept = Math.min(weakestKept, Math.abs(weight[e]));
    }
    expect(kept.size).toBe(INPUT_TOP_K);
    for (let i = 0; i < 784; i++) {
      if (!kept.has(i)) {
        expect(Math.abs(W[i])).toBeLessThanOrEqual(weakestKept + 1e-6);
      }
    }
  });

  it("computes a positive normalization scale", () => {
    for (const set of layout.edges) expect(maxAbsWeight(set)).toBeGreaterThan(0);
  });

  it("rejects nets that are not 784-h-h-out", () => {
    expect(() =>
      buildLayout({
        shape: [4, 3, 2],
        layers: [
          { W: new Float32Array(12), b: new Float32Array(3) },
          { W: new Float32Array(6), b: new Float32Array(2) },
        ],
      })
    ).toThrow(/784/);
  });
});
