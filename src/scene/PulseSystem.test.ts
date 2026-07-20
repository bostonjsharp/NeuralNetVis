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
