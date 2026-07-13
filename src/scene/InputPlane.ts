import * as THREE from "three";
import { GRID, INPUT_CENTER, INPUT_TILT_Y, PIXEL_PITCH } from "./NetworkLayout";

/**
 * The 28×28 input image as a glowing pixel grid: one plane, one DataTexture,
 * nearest filtering for crisp pixels. `setBrightness` scales the material
 * color (HDR, feeds bloom) so the fire cinematic can bloom the plane up
 * without rewriting texture bytes.
 */
/** Texels per input pixel: a 3×3 cell whose last row/column darkens into a
 *  seam, so the plane reads as a literal grid of 784 pixels even mid-bloom. */
const CELL = 3;
const TEX = GRID * CELL;

export class InputPlane {
  readonly mesh: THREE.Mesh;
  private readonly texture: THREE.DataTexture;
  private readonly data: Uint8Array;
  private readonly material: THREE.MeshBasicMaterial;

  constructor() {
    this.data = new Uint8Array(TEX * TEX * 4);
    this.texture = new THREE.DataTexture(this.data, TEX, TEX, THREE.RGBAFormat);
    this.texture.magFilter = THREE.NearestFilter;
    this.texture.minFilter = THREE.NearestFilter;
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.material = new THREE.MeshBasicMaterial({
      map: this.texture,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    const size = GRID * PIXEL_PITCH;
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size), this.material);
    this.mesh.position.set(...INPUT_CENTER);
    this.mesh.rotation.y = INPUT_TILT_Y;
    this.setPixels(new Float32Array(GRID * GRID));
  }

  /** `pixels` is the MNIST-ordered [0,1] image (row 0 at top). */
  setPixels(pixels: Float32Array): void {
    for (let row = 0; row < GRID; row++) {
      for (let col = 0; col < GRID; col++) {
        const v = 0.06 + 0.94 * Math.pow(pixels[row * GRID + col], 0.8);
        // Peak stays below bloom saturation so the digit's shape survives
        // the glow instead of dissolving into a white blob.
        const r = 8 + 140 * v;
        const g = 12 + 162 * v;
        const b = 35 + 172 * v;
        for (let cy = 0; cy < CELL; cy++) {
          for (let cx = 0; cx < CELL; cx++) {
            const seam = cy === CELL - 1 || cx === CELL - 1 ? 0.3 : 1;
            // Texture row 0 renders at the plane's bottom; MNIST row 0 is the top.
            const ty = (GRID - 1 - row) * CELL + cy;
            const tx = col * CELL + cx;
            const t = (ty * TEX + tx) * 4;
            this.data[t] = r * seam;
            this.data[t + 1] = g * seam;
            this.data[t + 2] = b * seam;
            this.data[t + 3] = 255;
          }
        }
      }
    }
    this.texture.needsUpdate = true;
  }

  /** HDR brightness multiplier (1 = neutral; >1 blooms). */
  setBrightness(value: number): void {
    this.material.color.setScalar(value);
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.dispose();
    this.texture.dispose();
  }
}
