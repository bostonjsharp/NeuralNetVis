import { describe, expect, it } from "vitest";
import weightsJson from "../assets/weights.json";
import samplesJson from "../assets/samples.json";
import { forwardPass } from "./inference";
import { decodeSamplePixels, loadNet, type SamplesJson, type WeightsJson } from "./weights";

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
