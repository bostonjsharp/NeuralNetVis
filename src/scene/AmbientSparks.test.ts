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
