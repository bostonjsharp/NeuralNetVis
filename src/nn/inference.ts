import { createStagedPass, softmax } from "./stagedPass";
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
 * Full forward pass in one call — a StagedPass stepped to completion, so the
 * math has a single source of truth (see stagedPass.ts). The scene uses the
 * stepper directly; this convenience wrapper serves tests and any caller
 * that wants the answer immediately.
 */
export function forwardPass(net: Net, input: Float32Array): ForwardResult {
  const pass = createStagedPass(net, input);
  while (!pass.done) pass.step();
  return pass.result!;
}

export { softmax };
