import * as THREE from "three";
import { maxAbsWeight, type NetworkLayout } from "./NetworkLayout";

/**
 * All rendered connections as one LineSegments draw call. Each vertex
 * carries its edge's normalized signed weight (warm = positive/excites,
 * cool = negative/inhibits) and a one-hot stage mask so the cinematic can
 * lift each stage's glow while its pulse wave is in flight.
 */
export class ConnectionMesh {
  readonly lines: THREE.LineSegments;
  private readonly material: THREE.ShaderMaterial;

  constructor(layout: NetworkLayout) {
    const totalEdges = layout.edges.reduce((sum, set) => sum + set.count, 0);
    const positions = new Float32Array(totalEdges * 2 * 3);
    const weights = new Float32Array(totalEdges * 2);
    const stageMasks = new Float32Array(totalEdges * 2 * 3);

    let v = 0;
    for (let stage = 0; stage < layout.edges.length; stage++) {
      const set = layout.edges[stage];
      const fromPositions = stage === 0 ? layout.inputPositions : layout.layerPositions[stage - 1];
      const toPositions = layout.layerPositions[stage];
      const wMax = maxAbsWeight(set);
      for (let e = 0; e < set.count; e++) {
        const w = set.weight[e] / wMax;
        for (const [source, idx] of [
          [fromPositions, set.from[e]],
          [toPositions, set.to[e]],
        ] as const) {
          positions[v * 3] = source[idx * 3];
          positions[v * 3 + 1] = source[idx * 3 + 1];
          positions[v * 3 + 2] = source[idx * 3 + 2];
          weights[v] = w;
          stageMasks[v * 3 + stage] = 1;
          v++;
        }
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("aWeight", new THREE.BufferAttribute(weights, 1));
    geometry.setAttribute("aStageMask", new THREE.BufferAttribute(stageMasks, 3));

    this.material = new THREE.ShaderMaterial({
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      uniforms: { uStageGlow: { value: new THREE.Vector3(0, 0, 0) } },
      vertexShader: /* glsl */ `
        attribute float aWeight;
        attribute vec3 aStageMask;
        uniform vec3 uStageGlow;
        varying float vWeight;
        varying float vGlow;
        void main() {
          vWeight = aWeight;
          vGlow = dot(aStageMask, uStageGlow);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        varying float vWeight;
        varying float vGlow;
        void main() {
          vec3 warm = vec3(1.0, 0.58, 0.20);
          vec3 cool = vec3(0.25, 0.72, 1.0);
          vec3 tint = vWeight >= 0.0 ? warm : cool;
          float mag = abs(vWeight);
          float alpha = (0.07 + 0.33 * mag) * (1.0 + 1.1 * vGlow);
          gl_FragColor = vec4(tint * (0.5 + 0.8 * vGlow), alpha * mag);
        }
      `,
    });

    this.lines = new THREE.LineSegments(geometry, this.material);
    this.lines.frustumCulled = false;
  }

  /** Per-stage glow lift, 0..1 each, driven by the fire timeline.
   *  Unused vec3 lanes (nets with <3 stages) stay zero. */
  setStageGlow(glows: number[]): void {
    (this.material.uniforms.uStageGlow.value as THREE.Vector3).set(
      glows[0] ?? 0,
      glows[1] ?? 0,
      glows[2] ?? 0
    );
  }

  dispose(): void {
    this.lines.geometry.dispose();
    this.material.dispose();
  }
}
