import { describe, expect, it } from "vitest";
import weightsJson from "../assets/weights.json";
import { loadNet, type Net, type WeightsJson } from "../nn/weights";
import { buildLayout, INPUT_TOP_K, maxAbsWeight, topKIndices } from "./NetworkLayout";

/** Deterministic pseudo-random net for shapes we don't ship weights for. */
const makeNet = (shape: number[]): Net => ({
  shape,
  layers: Array.from({ length: shape.length - 1 }, (_, l) => {
    const [nIn, nOut] = [shape[l], shape[l + 1]];
    const W = new Float32Array(nIn * nOut);
    for (let i = 0; i < W.length; i++) W[i] = Math.sin(i * 12.9898 + l) * 0.5;
    return { W, b: new Float32Array(nOut) };
  }),
});

describe("topKIndices", () => {
  it("picks the largest magnitudes, strongest first", () => {
    const values = Float32Array.from([0.1, -5, 2, -0.5, 3]);
    expect(Array.from(topKIndices(values, 3))).toEqual([1, 4, 2]);
  });
});

describe("buildLayout (classic weights)", () => {
  const net = loadNet(weightsJson as WeightsJson);
  const layout = buildLayout(net);

  it("positions every pixel and neuron", () => {
    expect(layout.inputPositions).toHaveLength(784 * 3);
    expect(layout.layerPositions).toHaveLength(3);
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

  it("keeps the two hidden columns at their hand-tuned X positions", () => {
    // The N-hidden-layer generalization must reproduce the tuned composition
    expect(layout.layerPositions[0][0]).toBeCloseTo(-3.5, 6);
    expect(layout.layerPositions[1][0]).toBeCloseTo(7.5, 6);
    // ...and the tuned 0.92 vertical spacing for 16-neuron columns
    expect(layout.layerPositions[0][1] - layout.layerPositions[0][4]).toBeCloseTo(0.92, 6);
  });

  it("renders top-K input edges and all deeper edges", () => {
    expect(layout.edges).toHaveLength(3);
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
});

describe("buildLayout (variant shapes)", () => {
  it("lays out a linear 784→10 net with one pruned stage", () => {
    const layout = buildLayout(makeNet([784, 10]), { inputTopK: 64 });
    expect(layout.layerPositions).toHaveLength(1);
    expect(layout.layerPositions[0]).toHaveLength(10 * 3);
    expect(layout.edges).toHaveLength(1);
    expect(layout.edges[0].count).toBe(10 * 64);
    for (let i = 0; i < layout.edges[0].count; i++) {
      expect(layout.edges[0].from[i]).toBeLessThan(784);
      expect(layout.edges[0].to[i]).toBeLessThan(10);
    }
  });

  it("lays out a single-hidden-layer 784→8→10 net", () => {
    const layout = buildLayout(makeNet([784, 8, 10]));
    expect(layout.layerPositions).toHaveLength(2);
    expect(layout.layerPositions[0]).toHaveLength(8 * 3);
    expect(layout.edges).toHaveLength(2);
    expect(layout.edges[0].count).toBe(8 * INPUT_TOP_K);
    expect(layout.edges[1].count).toBe(8 * 10);
    // The lone hidden column sits between input and output
    const hiddenX = layout.layerPositions[0][0];
    expect(hiddenX).toBeGreaterThan(-16.5);
    expect(hiddenX).toBeLessThan(layout.layerPositions[1][0]);
  });

  it("tightens vertical spacing so a 32-neuron column still fits the frame", () => {
    const layout = buildLayout(makeNet([784, 32, 32, 10]));
    const ys = Array.from({ length: 32 }, (_, i) => layout.layerPositions[0][i * 3 + 1]);
    const height = Math.max(...ys) - Math.min(...ys);
    expect(height).toBeLessThanOrEqual(14.5);
    // ...but 16-neuron columns keep their original look
    const classic = buildLayout(makeNet([784, 16, 16, 10]));
    expect(classic.layerPositions[0][1] - classic.layerPositions[0][4]).toBeCloseTo(0.92, 6);
  });

  it("keeps output at the same X for every shape (camera framing invariant)", () => {
    const outputX = (net: Net) => {
      const layout = buildLayout(net);
      const last = layout.layerPositions[layout.layerPositions.length - 1];
      return last[0];
    };
    const xs = [
      outputX(makeNet([784, 10])),
      outputX(makeNet([784, 8, 10])),
      outputX(makeNet([784, 16, 16, 10])),
      outputX(makeNet([784, 32, 32, 10])),
    ];
    for (const x of xs) expect(x).toBeCloseTo(xs[0], 6);
  });

  it("rejects shapes outside the exhibit's invariants", () => {
    expect(() => buildLayout(makeNet([4, 3, 2]))).toThrow(/784/);
    expect(() => buildLayout(makeNet([784, 16, 16, 16, 10]))).toThrow();
    expect(() => buildLayout(makeNet([784, 16, 12]))).toThrow(/10/);
  });
});
