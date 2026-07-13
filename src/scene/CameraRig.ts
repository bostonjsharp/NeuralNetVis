import * as THREE from "three";

export type CameraMode = "attract" | "interactive";

/** One full lap of the attract orbit, seconds. */
const ATTRACT_PERIOD = 90;

/**
 * Camera choreography. Attract mode drifts along a closed spline that
 * alternates wide establishing shots with closer flybys; interactive mode
 * settles into a fixed three-quarter framing that keeps the input plane and
 * output column visible. All transitions are critically-damped — no cuts.
 */
export class CameraRig {
  private mode: CameraMode = "attract";
  private readonly curve: THREE.CatmullRomCurve3;
  private readonly desiredPosition = new THREE.Vector3();
  private readonly desiredTarget = new THREE.Vector3();
  private readonly smoothedTarget = new THREE.Vector3(0, 0, 0);
  // Framed so the network floats in the wall's upper band, clear of the
  // HUD panels along the bottom edge.
  private readonly interactivePosition = new THREE.Vector3(-2, 0.6, 32.5);
  private readonly interactiveTarget = new THREE.Vector3(-2, -2.6, 0);

  constructor(private readonly camera: THREE.PerspectiveCamera) {
    // Waypoints keep ~20+ units out AND never swing far enough sideways to
    // look down the network's X axis — hundreds of additive connection
    // lines seen edge-on stack into a white wedge.
    this.curve = new THREE.CatmullRomCurve3(
      [
        new THREE.Vector3(-2, 2, 33),
        new THREE.Vector3(12, 5, 28),
        new THREE.Vector3(20, 1, 24),
        new THREE.Vector3(8, -4, 29),
        new THREE.Vector3(-9, 3.5, 28),
        new THREE.Vector3(-15, 0.5, 25),
        new THREE.Vector3(-8, -2.5, 31),
      ],
      true,
      "catmullrom",
      0.5
    );
    camera.position.copy(this.curve.getPoint(0));
    camera.lookAt(this.smoothedTarget);
  }

  setMode(mode: CameraMode): void {
    this.mode = mode;
  }

  update(elapsed: number, dt: number): void {
    if (this.mode === "attract") {
      this.curve.getPoint((elapsed / ATTRACT_PERIOD) % 1, this.desiredPosition);
      // Gaze drifts slowly around the network's heart
      this.desiredTarget.set(
        1.5 + 4.5 * Math.sin(elapsed * 0.05),
        1.1 * Math.sin(elapsed * 0.073),
        0
      );
    } else {
      this.desiredPosition.copy(this.interactivePosition);
      this.desiredTarget.copy(this.interactiveTarget);
    }
    // Critically-damped exponential smoothing (frame-rate independent)
    const k = 1 - Math.exp(-1.8 * dt);
    this.camera.position.lerp(this.desiredPosition, k);
    this.smoothedTarget.lerp(this.desiredTarget, k);
    this.camera.lookAt(this.smoothedTarget);
  }
}
