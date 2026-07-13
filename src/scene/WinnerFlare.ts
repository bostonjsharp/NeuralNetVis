import * as THREE from "three";

/**
 * Expanding ring shockwave at the winning output neuron. One camera-facing
 * plane with a radial-ring canvas texture; scaled and faded by the winner
 * flare envelope each frame.
 */
export class WinnerFlare {
  readonly mesh: THREE.Mesh;
  private readonly material: THREE.MeshBasicMaterial;

  constructor() {
    const size = 256;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d")!;
    const gradient = ctx.createRadialGradient(
      size / 2, size / 2, size * 0.30,
      size / 2, size / 2, size * 0.5
    );
    gradient.addColorStop(0, "rgba(255,255,255,0)");
    gradient.addColorStop(0.55, "rgba(255,230,170,0.9)");
    gradient.addColorStop(0.75, "rgba(255,190,90,0.45)");
    gradient.addColorStop(1, "rgba(255,170,60,0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);

    this.material = new THREE.MeshBasicMaterial({
      map: new THREE.CanvasTexture(canvas),
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      opacity: 0,
      toneMapped: false,
    });
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), this.material);
    this.mesh.visible = false;
  }

  /** Place the ring at the winner's position (call when a fire starts). */
  setTarget(position: THREE.Vector3): void {
    this.mesh.position.copy(position);
  }

  /** Drive with the winnerFlare envelope (0 hides it). */
  update(envelope: number, camera: THREE.Camera): void {
    this.mesh.visible = envelope > 0.001;
    if (!this.mesh.visible) return;
    this.mesh.quaternion.copy(camera.quaternion);
    const scale = 1.2 + 3.6 * (1 - Math.pow(1 - envelope, 2));
    this.mesh.scale.setScalar(scale);
    this.material.opacity = 0.65 * envelope;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.map?.dispose();
    this.material.dispose();
  }
}
