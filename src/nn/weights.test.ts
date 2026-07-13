import { describe, expect, it } from "vitest";
import { decodeF32, decodeSamplePixels, encodeF32, loadNet, type WeightsJson } from "./weights";

describe("base64 float codec", () => {
  it("round-trips a Float32Array exactly", () => {
    const arr = Float32Array.from([0, 1, -1, 3.14159, 1e-30, -1e30, 0.5]);
    expect(Array.from(decodeF32(encodeF32(arr)))).toEqual(Array.from(arr));
  });
});

function toyJson(overrides: Partial<WeightsJson> = {}): WeightsJson {
  return {
    shape: [2, 3, 2],
    testAccuracy: 1,
    trainedAt: "2026-01-01",
    layers: [
      { W: encodeF32(new Float32Array(6)), b: encodeF32(new Float32Array(3)) },
      { W: encodeF32(new Float32Array(6)), b: encodeF32(new Float32Array(2)) },
    ],
    ...overrides,
  };
}

describe("loadNet", () => {
  it("loads a valid net", () => {
    const net = loadNet(toyJson());
    expect(net.shape).toEqual([2, 3, 2]);
    expect(net.layers).toHaveLength(2);
    expect(net.layers[0].W).toHaveLength(6);
    expect(net.layers[1].b).toHaveLength(2);
  });

  it("rejects a layer-count mismatch", () => {
    const bad = toyJson({ shape: [2, 3, 3, 2] });
    expect(() => loadNet(bad)).toThrow(/layers/);
  });

  it("rejects a wrongly sized weight matrix", () => {
    const bad = toyJson();
    bad.layers[0].W = encodeF32(new Float32Array(5));
    expect(() => loadNet(bad)).toThrow(/layer 0/);
  });

  it("rejects a wrongly sized bias vector", () => {
    const bad = toyJson();
    bad.layers[1].b = encodeF32(new Float32Array(3));
    expect(() => loadNet(bad)).toThrow(/layer 1/);
  });
});

describe("decodeSamplePixels", () => {
  it("scales bytes to [0,1]", () => {
    const bytes = new Uint8Array(784);
    bytes[0] = 255;
    bytes[1] = 51;
    const b64 = btoa(String.fromCharCode(...bytes));
    const pixels = decodeSamplePixels(b64);
    expect(pixels[0]).toBeCloseTo(1);
    expect(pixels[1]).toBeCloseTo(0.2);
    expect(pixels[2]).toBe(0);
  });

  it("rejects non-784-pixel payloads", () => {
    expect(() => decodeSamplePixels(btoa("abc"))).toThrow(/784/);
  });
});
