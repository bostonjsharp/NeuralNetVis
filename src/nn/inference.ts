import type { Net } from "./weights";

export interface ForwardResult {
  /** One entry per network layer, input first. Hidden layers are post-ReLU. */
  activations: Float32Array[];
  /** Raw output-layer scores (pre-softmax). Same array as activations at the end. */
  logits: Float32Array;
  probs: Float32Array;
  argmax: number;
}

/**
 * Full forward pass, keeping every layer's activations so the scene can
 * replay the signal flow. Hidden layers use ReLU; the output layer returns
 * raw logits plus a numerically stable softmax.
 */
export function forwardPass(net: Net, input: Float32Array): ForwardResult {
  const { shape, layers } = net;
  if (input.length !== shape[0]) {
    throw new Error(`input has ${input.length} values, expected ${shape[0]}`);
  }
  const activations: Float32Array[] = [Float32Array.from(input)];
  const last = layers.length - 1;
  for (let l = 0; l < layers.length; l++) {
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
  }
  const logits = activations[activations.length - 1];
  const probs = softmax(logits);
  let argmax = 0;
  for (let o = 1; o < probs.length; o++) if (probs[o] > probs[argmax]) argmax = o;
  return { activations, logits, probs, argmax };
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
