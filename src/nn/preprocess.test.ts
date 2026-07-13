import { describe, expect, it } from "vitest";
import { preprocessDrawing } from "./preprocess";

function centerOfMass(img: Float32Array): { x: number; y: number; mass: number } {
  let mass = 0;
  let x = 0;
  let y = 0;
  for (let i = 0; i < 784; i++) {
    const v = img[i];
    mass += v;
    x += v * (i % 28);
    y += v * Math.floor(i / 28);
  }
  return { x: x / mass, y: y / mass, mass };
}

/** Paints a filled rectangle of ink on a blank w×h canvas. */
function rect(w: number, h: number, x0: number, y0: number, rw: number, rh: number): Float32Array {
  const img = new Float32Array(w * h);
  for (let y = y0; y < y0 + rh; y++) {
    for (let x = x0; x < x0 + rw; x++) img[y * w + x] = 1;
  }
  return img;
}

describe("preprocessDrawing", () => {
  it("returns zeros for an empty drawing", () => {
    const out = preprocessDrawing(new Float32Array(280 * 280), 280, 280);
    expect(out.every((v) => v === 0)).toBe(true);
  });

  it("centers an off-corner blob's mass on the field center", () => {
    // Small square tucked in the top-left of a big canvas
    const out = preprocessDrawing(rect(280, 280, 10, 10, 40, 40), 280, 280);
    const { x, y, mass } = centerOfMass(out);
    expect(mass).toBeGreaterThan(0);
    expect(x).toBeGreaterThan(12.5);
    expect(x).toBeLessThan(14.5);
    expect(y).toBeGreaterThan(12.5);
    expect(y).toBeLessThan(14.5);
  });

  it("scales the longest side to ~20px preserving aspect", () => {
    // Tall thin stroke (like a drawn "1"): 20×160 on a 280×280 canvas
    const out = preprocessDrawing(rect(280, 280, 130, 60, 20, 160), 280, 280);
    let minX = 28;
    let maxX = -1;
    let minY = 28;
    let maxY = -1;
    for (let i = 0; i < 784; i++) {
      if (out[i] > 0.01) {
        const x = i % 28;
        const y = Math.floor(i / 28);
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
    const inkH = maxY - minY + 1;
    const inkW = maxX - minX + 1;
    expect(inkH).toBeGreaterThanOrEqual(19);
    expect(inkH).toBeLessThanOrEqual(21);
    // Aspect preserved: width stays proportionally narrow (20/160 → ~2-3px)
    expect(inkW).toBeLessThanOrEqual(4);
  });

  it("keeps values in [0,1]", () => {
    const out = preprocessDrawing(rect(100, 100, 20, 20, 50, 50), 100, 100);
    for (const v of out) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});
