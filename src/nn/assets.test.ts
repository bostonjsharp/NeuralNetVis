import { describe, expect, it } from "vitest";
import weightsJson from "../assets/weights.json";
import samplesJson from "../assets/samples.json";
import { forwardPass } from "./inference";
import { decodeSamplePixels, loadNet, type SamplesJson, type WeightsJson } from "./weights";
import { DEFAULT_VARIANT_ID, nextVariantId, VARIANTS } from "./variants";

/**
 * Locks the training script and the runtime inference math to each other:
 * the committed weights, run through the runtime forward pass, must agree
 * with the committed sample labels. (Not 100% — samples deliberately include
 * one low-confidence digit per class.)
 */
describe("committed assets", () => {
  const net = loadNet(weightsJson as WeightsJson);
  const { samples } = samplesJson as SamplesJson;

  it("has the visualized 784-16-16-10 shape and ≥92% recorded accuracy", () => {
    expect(net.shape).toEqual([784, 16, 16, 10]);
    expect((weightsJson as WeightsJson).testAccuracy).toBeGreaterThanOrEqual(0.92);
  });

  it("ships 60 sample digits, 6 per class", () => {
    expect(samples).toHaveLength(60);
    for (let digit = 0; digit < 10; digit++) {
      expect(samples.filter((s) => s.label === digit)).toHaveLength(6);
    }
  });

  it("runtime inference agrees with ≥90% of sample labels", () => {
    let correct = 0;
    for (const sample of samples) {
      const { argmax } = forwardPass(net, decodeSamplePixels(sample.pixels));
      if (argmax === sample.label) correct++;
    }
    expect(correct / samples.length).toBeGreaterThanOrEqual(0.9);
  });
});

describe("brain variants", () => {
  const { samples } = samplesJson as SamplesJson;

  const EXPECTED: Record<
    string,
    { shape: number[]; minAcc: number; minSampleAgreement: number }
  > = {
    // Samples were confidence-picked FOR the classic net — dumber brains
    // legitimately missing more of them is the exhibit's whole point.
    linear: { shape: [784, 10], minAcc: 0.9, minSampleAgreement: 0.8 },
    tiny: { shape: [784, 8, 10], minAcc: 0.9, minSampleAgreement: 0.8 },
    classic: { shape: [784, 16, 16, 10], minAcc: 0.92, minSampleAgreement: 0.9 },
    wide: { shape: [784, 32, 32, 10], minAcc: 0.95, minSampleAgreement: 0.9 },
  };

  it("ships all four brains", () => {
    expect(VARIANTS.map((v) => v.id).sort()).toEqual(Object.keys(EXPECTED).sort());
  });

  for (const variant of VARIANTS) {
    describe(variant.id, () => {
      const expected = EXPECTED[variant.id];

      it("matches its expected shape and honest accuracy floor", () => {
        expect(variant.net.shape).toEqual(expected.shape);
        expect(variant.testAccuracy).toBeGreaterThanOrEqual(expected.minAcc);
      });

      it("agrees with enough of the shared sample deck", () => {
        let correct = 0;
        for (const sample of samples) {
          const { argmax } = forwardPass(variant.net, decodeSamplePixels(sample.pixels));
          if (argmax === sample.label) correct++;
        }
        expect(correct / samples.length).toBeGreaterThanOrEqual(expected.minSampleAgreement);
      });

      it("reports a true parameter count", () => {
        const expectedParams = expected.shape
          .slice(0, -1)
          .reduce((sum, nIn, l) => sum + nIn * expected.shape[l + 1] + expected.shape[l + 1], 0);
        expect(variant.paramCount).toBe(expectedParams);
      });
    });
  }

  it("defaults to classic — the same net as weights.json", () => {
    const classic = VARIANTS.find((v) => v.id === DEFAULT_VARIANT_ID)!;
    expect(classic.net.shape).toEqual([784, 16, 16, 10]);
    expect(classic.testAccuracy).toBe((weightsJson as WeightsJson).testAccuracy);
  });

  it("cycles through every brain and wraps around", () => {
    const seen = new Set<string>([DEFAULT_VARIANT_ID]);
    let id = DEFAULT_VARIANT_ID;
    for (let i = 0; i < VARIANTS.length - 1; i++) {
      id = nextVariantId(id);
      seen.add(id);
    }
    expect(seen.size).toBe(VARIANTS.length);
    expect(nextVariantId(id)).toBe(DEFAULT_VARIANT_ID);
  });
});
