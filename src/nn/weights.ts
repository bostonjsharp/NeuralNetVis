export interface NetLayer {
  /** Row-major [out][in]: y[o] = Σ W[o*nIn + i] * x[i] + b[o] */
  W: Float32Array;
  b: Float32Array;
}

export interface Net {
  shape: number[];
  layers: NetLayer[];
}

export interface WeightsJson {
  /** Variant identity — absent in pre-variant exports. */
  id?: string;
  label?: string;
  shape: number[];
  testAccuracy: number;
  trainedAt: string;
  layers: { W: string; b: string }[];
}

export interface SamplesJson {
  samples: { label: number; pixels: string }[];
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

export function decodeF32(b64: string): Float32Array {
  const bytes = base64ToBytes(b64);
  // Copy into a fresh buffer to guarantee 4-byte alignment for the view.
  return new Float32Array(bytes.slice().buffer);
}

export function encodeF32(arr: Float32Array): string {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

export function loadNet(json: WeightsJson): Net {
  const { shape } = json;
  if (json.layers.length !== shape.length - 1) {
    throw new Error(`weights: ${json.layers.length} layers for shape [${shape}]`);
  }
  const layers = json.layers.map((layer, l) => {
    const W = decodeF32(layer.W);
    const b = decodeF32(layer.b);
    const [nIn, nOut] = [shape[l], shape[l + 1]];
    if (W.length !== nOut * nIn) {
      throw new Error(`weights: layer ${l} W has ${W.length} values, expected ${nOut * nIn}`);
    }
    if (b.length !== nOut) {
      throw new Error(`weights: layer ${l} b has ${b.length} values, expected ${nOut}`);
    }
    return { W, b };
  });
  return { shape, layers };
}

/** Decode one sample's base64 pixels to a [0,1] Float32Array(784). */
export function decodeSamplePixels(b64: string): Float32Array {
  const bytes = base64ToBytes(b64);
  if (bytes.length !== 784) {
    throw new Error(`sample: ${bytes.length} pixels, expected 784`);
  }
  const out = new Float32Array(784);
  for (let i = 0; i < 784; i++) out[i] = bytes[i] / 255;
  return out;
}
