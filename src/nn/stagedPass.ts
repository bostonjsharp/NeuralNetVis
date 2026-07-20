import type { ForwardResult } from "./inference";
import type { Net } from "./weights";

/**
 * A forward pass that computes one layer per step() — the scene calls step()
 * when a pulse wave lands, so the app genuinely does not know deeper layers'
 * values (or the answer) until the animation reaches them. Stepping to
 * completion is exactly forwardPass; inference.ts is implemented on top.
 */
export interface StagedPass {
  readonly net: Net;
  /** Input first; grows by one layer per step(). */
  readonly activations: Float32Array[];
  readonly done: boolean;
  /** Non-null only after the final step: logits, probs, argmax. */
  readonly result: ForwardResult | null;
  /** Compute the next layer (matmul + ReLU; logits + softmax/argmax on the last). */
  step(): void;
}

export function createStagedPass(net: Net, input: Float32Array): StagedPass {
  const { shape, layers } = net;
  if (input.length !== shape[0]) {
    throw new Error(`input has ${input.length} values, expected ${shape[0]}`);
  }
  const activations: Float32Array[] = [Float32Array.from(input)];
  const last = layers.length - 1;
  let result: ForwardResult | null = null;
  return {
    net,
    activations,
    get done() {
      return result !== null;
    },
    get result() {
      return result;
    },
    step() {
      if (result !== null) throw new Error("staged pass already complete");
      const l = activations.length - 1;
      const { W, b } = layers[l];
      const aIn = activations[l];
      const nIn = shape[l];
      const nOut = shape[l + 1];
      const aOut = new Float32Array(nOut);
      for (let o = 0; o < nOut; o++) {
        let sum = b[o];
        const row = o * nIn;
        for (let i = 0; i < nIn; i++) sum += W[row + i] * aIn[i];
        aOut[o] = l < last ? Math.max(0, sum) : sum;
      }
      activations.push(aOut);
      if (l === last) {
        const logits = aOut;
        const probs = softmax(logits);
        let argmax = 0;
        for (let o = 1; o < probs.length; o++) if (probs[o] > probs[argmax]) argmax = o;
        result = { activations, logits, probs, argmax };
      }
    },
  };
}

export function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}
