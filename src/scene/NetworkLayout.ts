import type { Net } from "../nn/weights";

/**
 * Pure spatial layout of the network — no three.js. World units flow along
 * X (input left, output right), Y is up, the camera lives out on +Z.
 * Everything the scene renders (neuron instances, connection lines, pulse
 * endpoints) reads positions from here, so the whole composition is tuned
 * in one file and unit-testable.
 *
 * Shapes are 784 → (0..2 hidden layers) → 10. The input plane and output
 * column are exhibit invariants — they never move, so camera framing and
 * the digit glyphs survive every brain swap; only the middle reshapes.
 */

export const GRID = 28;
export const PIXEL_PITCH = 0.4;
/** Input plane tilt: normal tips toward the hidden layers so pulses leave its face. */
export const INPUT_TILT_Y = 0.45;
export const INPUT_CENTER: readonly [number, number, number] = [-16.5, 0, 0];

/** Hidden columns spread evenly across this span — [-3.5, 7.5] reproduces
 *  the original hand-tuned two-column composition exactly. */
const HIDDEN_X_MIN = -3.5;
const HIDDEN_X_MAX = 7.5;
const HIDDEN_SPACING = 0.92;
/** Tallest a column may grow before its spacing tightens to fit the frame. */
const MAX_COLUMN_HEIGHT = 14;
const OUTPUT_X = 18;
const OUTPUT_SPACING = 1.24;

export const NEURON_RADIUS_HIDDEN = 0.34;
export const NEURON_RADIUS_OUTPUT = 0.46;

/** How far past an output neuron its digit sprite reaches (see OutputGlyphs:
 *  the label sits at radius + 1.05 and swells to ~2× its 1.05 base scale when
 *  it wins). Camera framing has to allow for it or winners clip at the edge. */
const OUTPUT_LABEL_REACH = NEURON_RADIUS_OUTPUT + 1.05 + 1.05;

/** Strongest input edges kept per destination neuron (full 12,544 is a hairball). */
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
  /** Downstream layer positions (hidden…, output), xyz-interleaved. */
  layerPositions: Float32Array[];
  /** Rendered edges per stage: input→first (top-K), then dense. */
  edges: EdgeSet[];
  /** A handful of extreme world points (xyz-interleaved) that enclose
   *  everything drawn — the input quad's corners and each column's padded
   *  ends. CameraRig frames against these, so a taller brain automatically
   *  buys itself more camera distance instead of clipping. */
  bounds: Float32Array;
}

export interface LayoutOptions {
  /** Override the stage-0 pruning — sparse nets can afford richer fans. */
  inputTopK?: number;
}

export function buildLayout(net: Net, options: LayoutOptions = {}): NetworkLayout {
  const { shape } = net;
  if (shape[0] !== GRID * GRID) {
    throw new Error(`layout expects 784 inputs, got [${shape}]`);
  }
  if (shape[shape.length - 1] !== 10) {
    throw new Error(`layout expects 10 outputs, got [${shape}]`);
  }
  if (shape.length < 2 || shape.length > 4) {
    throw new Error(`layout supports 0..2 hidden layers, got [${shape}]`);
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

  const hiddenCount = shape.length - 2;
  const hiddenX = (index: number): number =>
    hiddenCount === 1
      ? (HIDDEN_X_MIN + HIDDEN_X_MAX) / 2
      : HIDDEN_X_MIN + (index * (HIDDEN_X_MAX - HIDDEN_X_MIN)) / (hiddenCount - 1);
  const hiddenSpacing = (n: number): number =>
    Math.min(HIDDEN_SPACING, MAX_COLUMN_HEIGHT / Math.max(1, n - 1));

  const layerPositions: Float32Array[] = [];
  for (let h = 0; h < hiddenCount; h++) {
    const n = shape[h + 1];
    layerPositions.push(column(n, hiddenX(h), hiddenSpacing(n)));
  }
  layerPositions.push(column(shape[shape.length - 1], OUTPUT_X, OUTPUT_SPACING));

  // Stage 0: only each first-layer neuron's strongest input weights
  const k = Math.min(options.inputTopK ?? INPUT_TOP_K, shape[0]);
  const nFirst = shape[1];
  const s0 = {
    from: new Uint16Array(nFirst * k),
    to: new Uint16Array(nFirst * k),
    weight: new Float32Array(nFirst * k),
    count: nFirst * k,
  };
  const W0 = net.layers[0].W;
  for (let o = 0; o < nFirst; o++) {
    const row = W0.subarray(o * shape[0], (o + 1) * shape[0]);
    const top = topKIndices(row, k);
    for (let j = 0; j < k; j++) {
      const e = o * k + j;
      s0.from[e] = top[j];
      s0.to[e] = o;
      s0.weight[e] = row[top[j]];
    }
  }

  const bounds = buildBounds(layerPositions);

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

  const edges: EdgeSet[] = [s0];
  for (let l = 1; l < shape.length - 1; l++) edges.push(dense(l));

  return { inputPositions, layerPositions, edges, bounds };
}

/**
 * The corner points the camera has to keep on screen. The input plane is a
 * tilted quad (four corners) and every other layer is a vertical segment at
 * one x (four padded corners each) — a dozen-odd points rather than the ~850
 * neurons, since only the extremes can ever be the first thing to clip.
 */
function buildBounds(layerPositions: Float32Array[]): Float32Array {
  const points: number[] = [];

  // Input quad, grown by half a pixel so the outer row's quads are covered.
  const half = ((GRID - 1) / 2) * PIXEL_PITCH + PIXEL_PITCH / 2;
  const cos = Math.cos(INPUT_TILT_Y);
  const sin = Math.sin(INPUT_TILT_Y);
  for (const lx of [-half, half]) {
    for (const ly of [-half, half]) {
      points.push(INPUT_CENTER[0] + cos * lx, INPUT_CENTER[1] + ly, INPUT_CENTER[2] - sin * lx);
    }
  }

  const lastLayer = layerPositions.length - 1;
  layerPositions.forEach((positions, layer) => {
    const output = layer === lastLayer;
    const radius = output ? NEURON_RADIUS_OUTPUT : NEURON_RADIUS_HIDDEN;
    // Digit labels hang off the +x side of the output column only.
    const reach = output ? OUTPUT_LABEL_REACH : radius;
    const x = positions[0];
    const top = positions[1];
    const bottom = positions[positions.length - 2];
    for (const px of [x - radius, x + reach]) {
      for (const py of [top + reach, bottom - reach]) points.push(px, py, 0);
    }
  });

  return Float32Array.from(points);
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
