import * as THREE from "three";
import { FIRE } from "./fire";
import { maxAbsWeight, type NetworkLayout } from "./NetworkLayout";

/**
 * The hero effect: signal pulses flying along connections during a forward
 * pass. One THREE.Points draw call; the GPU interpolates every particle
 * along its edge from per-particle attributes, so the CPU only writes the
 * per-fire signal strengths (activation × weight) and a fire timestamp.
 * Additive blending + bloom turns strong signals into comets.
 */

const PARTICLES_PER_EDGE = 2;
const SIGNAL_VISIBLE_THRESHOLD = 0.05;

export class PulseSystem {
  readonly points: THREE.Points;
  private readonly material: THREE.ShaderMaterial;
  private readonly signalAttr: THREE.BufferAttribute;
  private readonly edgeCounts: number[];
  private readonly layout: NetworkLayout;
  private readonly weightNorm: [number, number, number];

  constructor(layout: NetworkLayout) {
    this.layout = layout;
    this.edgeCounts = layout.edges.map((set) => set.count);
    this.weightNorm = layout.edges.map((set) => maxAbsWeight(set)) as [number, number, number];
    const totalParticles =
      this.edgeCounts.reduce((a, b) => a + b, 0) * PARTICLES_PER_EDGE;

    const starts = new Float32Array(totalParticles * 3);
    const ends = new Float32Array(totalParticles * 3);
    const stageMasks = new Float32Array(totalParticles * 3);
    const jitters = new Float32Array(totalParticles);
    const slots = new Float32Array(totalParticles);
    const signals = new Float32Array(totalParticles);

    let p = 0;
    for (let stage = 0; stage < 3; stage++) {
      const set = layout.edges[stage];
      const fromPositions = stage === 0 ? layout.inputPositions : layout.layerPositions[stage - 1];
      const toPositions = layout.layerPositions[stage];
      for (let e = 0; e < set.count; e++) {
        for (let slot = 0; slot < PARTICLES_PER_EDGE; slot++, p++) {
          for (let axis = 0; axis < 3; axis++) {
            starts[p * 3 + axis] = fromPositions[set.from[e] * 3 + axis];
            ends[p * 3 + axis] = toPositions[set.to[e] * 3 + axis];
          }
          stageMasks[p * 3 + stage] = 1;
          jitters[p] = Math.random();
          slots[p] = slot;
        }
      }
    }

    const geometry = new THREE.BufferGeometry();
    // `position` must exist for draw range; actual travel is aStart→aEnd in the shader.
    geometry.setAttribute("position", new THREE.BufferAttribute(starts.slice(), 3));
    geometry.setAttribute("aStart", new THREE.BufferAttribute(starts, 3));
    geometry.setAttribute("aEnd", new THREE.BufferAttribute(ends, 3));
    geometry.setAttribute("aStageMask", new THREE.BufferAttribute(stageMasks, 3));
    geometry.setAttribute("aJitter", new THREE.BufferAttribute(jitters, 1));
    geometry.setAttribute("aSlot", new THREE.BufferAttribute(slots, 1));
    this.signalAttr = new THREE.BufferAttribute(signals, 1);
    this.signalAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("aSignal", this.signalAttr);

    this.material = new THREE.ShaderMaterial({
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      uniforms: {
        uTime: { value: 0 },
        uFireTime: { value: -1e9 },
        uStageStart: { value: new THREE.Vector3(...FIRE.stageStart) },
        uStageDur: { value: new THREE.Vector3(...FIRE.stageDur) },
      },
      vertexShader: /* glsl */ `
        attribute vec3 aStart;
        attribute vec3 aEnd;
        attribute vec3 aStageMask;
        attribute float aJitter;
        attribute float aSlot;
        attribute float aSignal;
        uniform float uTime;
        uniform float uFireTime;
        uniform vec3 uStageStart;
        uniform vec3 uStageDur;
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          float delay = dot(aStageMask, uStageStart) + aJitter * 0.28 + aSlot * 0.1;
          float dur = dot(aStageMask, uStageDur) * (0.8 + 0.4 * aJitter);
          float p = (uTime - uFireTime - delay) / dur;
          float mag = abs(aSignal);
          float visible = step(0.0, p) * (1.0 - step(1.0, p)) * step(${SIGNAL_VISIBLE_THRESHOLD}, mag);
          float eased = p * p * (3.0 - 2.0 * p);
          vec3 pos = mix(aStart, aEnd, clamp(eased, 0.0, 1.0));
          vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
          gl_Position = projectionMatrix * mvPosition;
          float fade = smoothstep(0.0, 0.12, p) * (1.0 - smoothstep(0.82, 1.0, p));
          vAlpha = visible * fade * (0.25 + 0.75 * mag);
          vec3 warm = vec3(1.0, 0.62, 0.22);
          vec3 cool = vec3(0.30, 0.75, 1.0);
          vColor = (aSignal >= 0.0 ? warm : cool) * (0.6 + 2.6 * mag);
          gl_PointSize = visible * (20.0 + 46.0 * mag) * (24.0 / max(1.0, -mvPosition.z));
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          if (vAlpha < 0.003) discard;
          float d = length(gl_PointCoord - 0.5);
          float core = smoothstep(0.5, 0.06, d);
          gl_FragColor = vec4(vColor, vAlpha * core);
        }
      `,
    });

    this.points = new THREE.Points(geometry, this.material);
    this.points.frustumCulled = false;
  }

  /**
   * Load per-edge signals for a forward pass and start the pulse waves.
   * `activations` is ForwardResult.activations (input first). Hidden-layer
   * sources are normalized by their layer max so brightness always uses the
   * full range.
   */
  fire(activations: Float32Array[], nowSeconds: number): void {
    const signals = this.signalAttr.array as Float32Array;
    let p = 0;
    for (let stage = 0; stage < 3; stage++) {
      const set = this.layout.edges[stage];
      const source = activations[stage];
      let sourceMax = 0;
      for (let i = 0; i < source.length; i++) sourceMax = Math.max(sourceMax, source[i]);
      const norm = sourceMax > 0 ? 1 / sourceMax : 0;
      for (let e = 0; e < set.count; e++) {
        const signal =
          source[set.from[e]] * norm * (set.weight[e] / this.weightNorm[stage]);
        const clamped = Math.max(-1, Math.min(1, signal));
        for (let slot = 0; slot < PARTICLES_PER_EDGE; slot++, p++) signals[p] = clamped;
      }
    }
    this.signalAttr.needsUpdate = true;
    this.material.uniforms.uFireTime.value = nowSeconds;
  }

  update(nowSeconds: number): void {
    this.material.uniforms.uTime.value = nowSeconds;
  }

  dispose(): void {
    this.points.geometry.dispose();
    this.material.dispose();
  }
}
