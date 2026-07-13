import * as THREE from "three";

/**
 * Sparse dim dust far behind the network — parallax depth for the orbiting
 * camera without competing with the data. Kept below the bloom threshold.
 */
export class Starfield {
  readonly points: THREE.Points;
  private readonly material: THREE.PointsMaterial;

  constructor(count = 450) {
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      // Shell between radius 55 and 110, biased away from the camera axis
      const radius = 55 + 55 * Math.random();
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.cos(phi) * 0.6;
      positions[i * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta) - 20;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.material = new THREE.PointsMaterial({
      color: new THREE.Color(0.35, 0.42, 0.6),
      size: 0.55,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
    });
    this.points = new THREE.Points(geometry, this.material);
  }

  update(elapsed: number): void {
    this.points.rotation.y = elapsed * 0.004;
  }

  dispose(): void {
    this.points.geometry.dispose();
    this.material.dispose();
  }
}
