import * as THREE from "three";
import { NEURON_RADIUS_OUTPUT, type NetworkLayout } from "./NetworkLayout";

/**
 * The digits 0–9 floating beside their output neurons, so the answer is
 * readable inside the 3D scene from any camera angle (the HUD bars carry
 * the precise percentages).
 */
export class OutputGlyphs {
  readonly group = new THREE.Group();
  private readonly sprites: THREE.Sprite[] = [];

  constructor(layout: NetworkLayout) {
    const positions = layout.layerPositions[layout.layerPositions.length - 1];
    for (let digit = 0; digit < 10; digit++) {
      const sprite = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: makeDigitTexture(digit),
          transparent: true,
          depthWrite: false,
          toneMapped: false,
        })
      );
      sprite.position.set(
        positions[digit * 3] + NEURON_RADIUS_OUTPUT + 1.05,
        positions[digit * 3 + 1],
        positions[digit * 3 + 2]
      );
      sprite.scale.setScalar(1.05);
      this.sprites.push(sprite);
      this.group.add(sprite);
    }
  }

  /**
   * Brightness follows each digit's probability; the winner scales up and
   * whitens with the flare envelope.
   */
  update(probs: Float32Array | null, reveal: number, winner: number, flare: number): void {
    for (let digit = 0; digit < 10; digit++) {
      const sprite = this.sprites[digit];
      const material = sprite.material as THREE.SpriteMaterial;
      const p = probs ? probs[digit] : 0;
      const lit = reveal * p;
      const isWinner = probs !== null && digit === winner;
      const glow = isWinner ? flare : 0;
      material.color.setRGB(
        0.28 + 1.0 * lit + 1.0 * glow,
        0.32 + 0.95 * lit + 0.95 * glow,
        0.42 + 0.75 * lit + 0.75 * glow
      );
      material.opacity = 0.5 + 0.5 * reveal;
      sprite.scale.setScalar(1.05 * (1 + 0.55 * glow + 0.35 * lit * reveal));
    }
  }

  dispose(): void {
    for (const sprite of this.sprites) {
      (sprite.material as THREE.SpriteMaterial).map?.dispose();
      sprite.material.dispose();
    }
  }
}

function makeDigitTexture(digit: number): THREE.CanvasTexture {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#ffffff";
  ctx.font = `700 ${size * 0.78}px Outfit, "Segoe UI", sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(digit), size / 2, size / 2 + size * 0.04);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
