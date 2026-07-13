import type { Net } from "../nn/weights";

/**
 * Pure spatial layout of the network — no three.js. World units flow along
 * X (input left, output right), Y is up, the camera lives out on +Z.
 * Everything the scene renders (neuron instances, connection lines, pulse
 * endpoints) reads positions from here, so the whole composition is tuned
 * in one file and unit-testable.
 */

export const GRID = 28;
export const PIXEL_PITCH = 0.4;
/** Input plane tilt: normal tips toward the hidden layers so pulses leave its face. */
export const INPUT_TILT_Y = 0.45;
export const INPUT_CENTER: readonly [number, number, number] = [-16.5, 0, 0];

const HIDDEN_X = [-3.5, 7.5];
const HIDDEN_SPACING = 0.92;
const OUTPUT_X = 18;
const OUTPUT_SPACING = 1.24;

export const NEURON_RADIUS_HIDDEN = 0.34;
export const NEURON_RADIUS_OUTPUT = 0.46;

/** Strongest input→h1 edges kept per hidden neuron (full 12,544 is a hairball). */
export const INPUT_TOP_K = 32;

export interface EdgeSet {
  /** Index into the source layer (pixels for stage 0). */
  from: Uint16Array;
  /** Index into the destination layer. */
  to: Uint16Array;
  weight: Float32Array;
  count: number;
}

export interface NetworkLayout {
  /** World position of each input pixel center, xyz-interleaved (784×3). */
  inputPositions: Float32Array;
  /** h1, h2, output neuron world positions, xyz-interleaved. */
  layerPositions: [Float32Array, Float32Array, Float32Array];
  /** Rendered edges per stage: input→h1 (top-K), h1→h2 (all), h2→out (all). */
  edges: [EdgeSet, EdgeSet, EdgeSet];
}

export function buildLayout(net: Net): NetworkLayout {
  const { shape } = net;
  if (shape.length !== 4 || shape[0] !== GRID * GRID) {
    throw new Error(`layout expects a 784-h-h-out net, got [${shape}]`);
  }

  const inputPositions = new Float32Array(shape[0] * 3);
  const cos = Math.cos(INPUT_TILT_Y);
  const sin = Math.sin(INPUT_TILT_Y);
  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID; col++) {
      const i = row * GRID + col;
      const lx = (col - (GRID - 1) / 2) * PIXEL_PITCH;
      const ly = ((GRID - 1) / 2 - row) * PIXEL_PITCH;
      inputPositions[i * 3] = INPUT_CENTER[0] + cos * lx;
      inputPositions[i * 3 + 1] = INPUT_CENTER[1] + ly;
      inputPositions[i * 3 + 2] = INPUT_CENTER[2] - sin * lx;
    }
  }

  const column = (n: number, x: number, spacing: number): Float32Array => {
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      positions[i * 3] = x;
      positions[i * 3 + 1] = ((n - 1) / 2 - i) * spacing;
      positions[i * 3 + 2] = 0;
    }
    return positions;
  };
  const layerPositions: [Float32Array, Float32Array, Float32Array] = [
    column(shape[1], HIDDEN_X[0], HIDDEN_SPACING),
    column(shape[2], HIDDEN_X[1], HIDDEN_SPACING),
    column(shape[3], OUTPUT_X, OUTPUT_SPACING),
  ];

  // Stage 0: only each h1 neuron's strongest input weights
  const k = Math.min(INPUT_TOP_K, shape[0]);
  const s0 = {
    from: new Uint16Array(shape[1] * k),
    to: new Uint16Array(shape[1] * k),
    weight: new Float32Array(shape[1] * k),
    count: shape[1] * k,
  };
  const W0 = net.layers[0].W;
  for (let o = 0; o < shape[1]; o++) {
    const row = W0.subarray(o * shape[0], (o + 1) * shape[0]);
    const top = topKIndices(row, k);
    for (let j = 0; j < k; j++) {
      const e = o * k + j;
      s0.from[e] = top[j];
      s0.to[e] = o;
      s0.weight[e] = row[top[j]];
    }
  }

  const dense = (layer: number): EdgeSet => {
    const [nIn, nOut] = [shape[layer], shape[layer + 1]];
    const W = net.layers[layer].W;
    const set = {
      from: new Uint16Array(nIn * nOut),
      to: new Uint16Array(nIn * nOut),
      weight: new Float32Array(nIn * nOut),
      count: nIn * nOut,
    };
    let e = 0;
    for (let o = 0; o < nOut; o++) {
      for (let i = 0; i < nIn; i++, e++) {
        set.from[e] = i;
        set.to[e] = o;
        set.weight[e] = W[o * nIn + i];
      }
    }
    return set;
  };

  return { inputPositions, layerPositions, edges: [s0, dense(1), dense(2)] };
}

/** Indices of the k largest values by magnitude, strongest first. */
export function topKIndices(values: Float32Array, k: number): Uint16Array {
  const indices = Array.from(values.keys());
  indices.sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]));
  return Uint16Array.from(indices.slice(0, k));
}

/** Normalization scale so edge/pulse brightness uses the full range: max |weight| per stage. */
export function maxAbsWeight(edges: EdgeSet): number {
  let max = 0;
  for (let i = 0; i < edges.count; i++) max = Math.max(max, Math.abs(edges.weight[i]));
  return max || 1;
}
