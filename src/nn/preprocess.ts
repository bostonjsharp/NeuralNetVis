/**
 * Normalizes a freehand drawing into MNIST's input distribution. The classic
 * dataset was built by scaling each digit to fit 20×20 (aspect preserved) and
 * shifting it so its center of mass sits at the middle of the 28×28 field —
 * the network only knows digits presented that way, so raw off-center
 * drawings classify terribly without this step.
 */

const INK_THRESHOLD = 0.01;
const DIGIT_SIZE = 20;
const FIELD = 28;
const FIELD_CENTER = 13.5; // continuous center of pixel indices 0..27

/**
 * `src` is a grayscale [0,1] image, row-major, ink = bright on black.
 * Returns a Float32Array(784) ready for `forwardPass`, or all zeros when the
 * drawing is empty.
 */
export function preprocessDrawing(src: Float32Array, width: number, height: number): Float32Array {
  const out = new Float32Array(FIELD * FIELD);

  // Ink bounding box
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (src[y * width + x] > INK_THRESHOLD) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return out;

  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;
  const scale = DIGIT_SIZE / Math.max(boxW, boxH);
  const destW = Math.max(1, Math.round(boxW * scale));
  const destH = Math.max(1, Math.round(boxH * scale));

  // Box-filter resample of the cropped ink into destW×destH
  const scaled = new Float32Array(destW * destH);
  for (let dy = 0; dy < destH; dy++) {
    const sy0 = minY + (dy / destH) * boxH;
    const sy1 = minY + ((dy + 1) / destH) * boxH;
    for (let dx = 0; dx < destW; dx++) {
      const sx0 = minX + (dx / destW) * boxW;
      const sx1 = minX + ((dx + 1) / destW) * boxW;
      let sum = 0;
      let count = 0;
      for (let sy = Math.floor(sy0); sy < Math.ceil(sy1); sy++) {
        for (let sx = Math.floor(sx0); sx < Math.ceil(sx1); sx++) {
          if (sy >= 0 && sy < height && sx >= 0 && sx < width) {
            sum += src[sy * width + sx];
            count++;
          }
        }
      }
      scaled[dy * destW + dx] = count > 0 ? sum / count : 0;
    }
  }

  // Center of mass of the scaled digit
  let mass = 0;
  let comX = 0;
  let comY = 0;
  for (let y = 0; y < destH; y++) {
    for (let x = 0; x < destW; x++) {
      const v = scaled[y * destW + x];
      mass += v;
      comX += v * x;
      comY += v * y;
    }
  }
  if (mass === 0) return out;
  comX /= mass;
  comY /= mass;

  // Paste so the center of mass lands on the field center
  const offX = Math.round(FIELD_CENTER - comX);
  const offY = Math.round(FIELD_CENTER - comY);
  for (let y = 0; y < destH; y++) {
    const fy = y + offY;
    if (fy < 0 || fy >= FIELD) continue;
    for (let x = 0; x < destW; x++) {
      const fx = x + offX;
      if (fx < 0 || fx >= FIELD) continue;
      out[fy * FIELD + fx] = Math.min(1, scaled[y * destW + x]);
    }
  }
  return out;
}
