import * as THREE from "three";
import {
  NEURON_RADIUS_HIDDEN,
  NEURON_RADIUS_OUTPUT,
  type NetworkLayout,
} from "./NetworkLayout";

/**
 * Every downstream neuron (hidden + output) as one InstancedMesh — a single
 * draw call. Per-instance HDR color carries activation brightness into the
 * bloom pass; per-instance scale gives the pop animation.
 */
export class NeuronField {
  readonly mesh: THREE.InstancedMesh;
  readonly count: number;
  private readonly basePositions: THREE.Vector3[] = [];
  private readonly baseRadii: number[] = [];
  private readonly dummy = new THREE.Object3D();

  /** Flat instance index of a neuron. Layers: hidden… then output last. */
  readonly indexOf: (layer: number, neuron: number) => number;

  constructor(layout: NetworkLayout) {
    const counts = layout.layerPositions.map((p) => p.length / 3);
    this.count = counts.reduce((a, b) => a + b, 0);
    const offsets = counts.map((_, layer) =>
      counts.slice(0, layer).reduce((a, b) => a + b, 0)
    );
    this.indexOf = (layer, neuron) => offsets[layer] + neuron;

    const lastLayer = counts.length - 1;
    for (let layer = 0; layer < counts.length; layer++) {
      const positions = layout.layerPositions[layer];
      const radius = layer === lastLayer ? NEURON_RADIUS_OUTPUT : NEURON_RADIUS_HIDDEN;
      for (let i = 0; i < counts[layer]; i++) {
        this.basePositions.push(
          new THREE.Vector3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2])
        );
        this.baseRadii.push(radius);
      }
    }

    const geometry = new THREE.SphereGeometry(1, 24, 16);
    const material = new THREE.MeshBasicMaterial({ toneMapped: false });
    this.mesh = new THREE.InstancedMesh(geometry, material, this.count);
    this.mesh.instanceColor = new THREE.InstancedBufferAttribute(
      new Float32Array(this.count * 3),
      3
    );
    for (let i = 0; i < this.count; i++) this.setInstance(i, 0.1, 0.1, 0.16, 1);
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  /** Set one neuron's HDR color and scale multiplier. Call `commit` after a batch. */
  setInstance(i: number, r: number, g: number, b: number, scale: number): void {
    this.dummy.position.copy(this.basePositions[i]);
    this.dummy.scale.setScalar(this.baseRadii[i] * scale);
    this.dummy.updateMatrix();
    this.mesh.setMatrixAt(i, this.dummy.matrix);
    this.mesh.instanceColor!.setXYZ(i, r, g, b);
  }

  commit(): void {
    this.mesh.instanceMatrix.needsUpdate = true;
    this.mesh.instanceColor!.needsUpdate = true;
  }

  positionOf(layer: number, neuron: number): THREE.Vector3 {
    return this.basePositions[this.indexOf(layer, neuron)];
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}
