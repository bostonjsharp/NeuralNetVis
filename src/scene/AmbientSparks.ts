import * as THREE from "three";
import { SPARK_POOL_SIZE } from "./ambient";
import { maxAbsWeight, type NetworkLayout } from "./NetworkLayout";

/**
 * Idle static: a small pool of dim sparks drifting along connection edges.
 * A miniature sibling of PulseSystem — same GPU edge interpolation, one
 * Points draw call — but deliberately smaller and dimmer than the pulse
 * comets. Ambient motion must read as static electricity, never as a real
 * signal wave: the answer is earned by the wavefront.
 */
export class AmbientSparks {
  readonly points: THREE.Points;
  private readonly material: THREE.ShaderMaterial;
  private readonly startAttr: THREE.BufferAttribute;
  private readonly endAttr: THREE.BufferAttribute;
  private readonly birthAttr: THREE.BufferAttribute;
  private readonly durAttr: THREE.BufferAttribute;
  private readonly magAttr: THREE.BufferAttribute;
  private readonly layout: NetworkLayout;
  private readonly weightNorm: number[];

  constructor(layout: NetworkLayout, poolSize: number = SPARK_POOL_SIZE) {
    this.layout = layout;
    this.weightNorm = layout.edges.map((set) => maxAbsWeight(set));

    const dyn = (itemSize: number, fill = 0) => {
      const attr = new THREE.BufferAttribute(
        new Float32Array(poolSize * itemSize).fill(fill),
        itemSize
      );
      attr.setUsage(THREE.DynamicDrawUsage);
      return attr;
    };

    const geometry = new THREE.BufferGeometry();
    // `position` must exist for draw range; travel is aStart→aEnd in the shader.
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(poolSize * 3), 3)
    );
    geometry.setAttribute("aStart", (this.startAttr = dyn(3)));
    geometry.setAttribute("aEnd", (this.endAttr = dyn(3)));
    // Born in the far past with a nonzero duration: every slot starts dead.
    geometry.setAttribute("aBirth", (this.birthAttr = dyn(1, -1e9)));
    geometry.setAttribute("aDur", (this.durAttr = dyn(1, 1)));
    geometry.setAttribute("aMag", (this.magAttr = dyn(1)));

    this.material = new THREE.ShaderMaterial({
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      uniforms: {
        uTime: { value: 0 },
        uAlpha: { value: 1 },
      },
      vertexShader: /* glsl */ `
        attribute vec3 aStart;
        attribute vec3 aEnd;
        attribute float aBirth;
        attribute float aDur;
        attribute float aMag;
        uniform float uTime;
        uniform float uAlpha;
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          float p = (uTime - aBirth) / aDur;
          float visible = step(0.0, p) * (1.0 - step(1.0, p));
          float eased = p * p * (3.0 - 2.0 * p);
          vec3 pos = mix(aStart, aEnd, clamp(eased, 0.0, 1.0));
          vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
          gl_Position = projectionMatrix * mvPosition;
          float mag = abs(aMag);
          float fade = smoothstep(0.0, 0.25, p) * (1.0 - smoothstep(0.7, 1.0, p));
          vAlpha = visible * fade * (0.10 + 0.30 * mag) * uAlpha;
          vec3 warm = vec3(1.0, 0.62, 0.22);
          vec3 cool = vec3(0.30, 0.75, 1.0);
          vColor = (aMag >= 0.0 ? warm : cool) * (0.4 + 0.9 * mag);
          gl_PointSize = visible * (5.0 + 9.0 * mag) * (24.0 / max(1.0, -mvPosition.z));
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

  /** Launch one spark along an edge. Tint follows the weight's sign, so a
   *  spark tells the same warm/cool story as the wiring it rides. */
  spawn(
    slot: number,
    stage: number,
    edge: number,
    birth: number,
    duration: number,
    magnitude: number
  ): void {
    const set = this.layout.edges[stage];
    const from =
      stage === 0 ? this.layout.inputPositions : this.layout.layerPositions[stage - 1];
    const to = this.layout.layerPositions[stage];
    for (let axis = 0; axis < 3; axis++) {
      this.startAttr.setComponent(slot, axis, from[set.from[edge] * 3 + axis]);
      this.endAttr.setComponent(slot, axis, to[set.to[edge] * 3 + axis]);
    }
    this.birthAttr.setX(slot, birth);
    this.durAttr.setX(slot, duration);
    const sign = set.weight[edge] / this.weightNorm[stage] >= 0 ? 1 : -1;
    this.magAttr.setX(slot, sign * magnitude);
    this.startAttr.needsUpdate = true;
    this.endAttr.needsUpdate = true;
    this.birthAttr.needsUpdate = true;
    this.durAttr.needsUpdate = true;
    this.magAttr.needsUpdate = true;
  }

  /** Per-frame: advance the shader clock; alpha is the duck envelope, so
   *  live sparks dim (not vanish) when a fire starts. */
  update(now: number, alpha: number): void {
    this.material.uniforms.uTime.value = now;
    this.material.uniforms.uAlpha.value = alpha;
  }

  dispose(): void {
    this.points.geometry.dispose();
    this.material.dispose();
  }
}
