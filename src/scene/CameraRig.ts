import * as THREE from "three";
import { cameraDrift } from "./ambient";
import type { NetworkLayout } from "./NetworkLayout";

export type CameraMode = "attract" | "interactive";

/** One full lap of the attract orbit, seconds. */
const ATTRACT_PERIOD = 90;

/** Widest fraction of the frame the network may span. The remaining gutter is
 *  breathing room, and absorbs the smoothing lag behind a fast orbit leg. */
const FRAME_FILL = 0.92;

const WORLD_UP = new THREE.Vector3(0, 1, 0);

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
  /** Extreme points of the current brain — see NetworkLayout.bounds. */
  private bounds: Float32Array;
  // Scratch vectors for the containment pass; it runs every frame.
  private readonly forward = new THREE.Vector3();
  private readonly right = new THREE.Vector3();
  private readonly up = new THREE.Vector3();
  private readonly offset = new THREE.Vector3();

  constructor(
    private readonly camera: THREE.PerspectiveCamera,
    layout: NetworkLayout
  ) {
    this.bounds = layout.bounds;
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

  /** Reframe for a freshly swapped brain — a wider net is a taller net. */
  setLayout(layout: NetworkLayout): void {
    this.bounds = layout.bounds;
  }

  /**
   * Slide the desired position back along its own view axis until the whole
   * network fits inside FRAME_FILL of the frustum.
   *
   * Moving along that axis leaves each bound point's lateral offset alone and
   * only grows its depth, so the distance needed has a closed form: a point at
   * lateral offset l clears the frustum edge once depth reaches l / tan(θ).
   * Taking the largest shortfall over the corners fits all of them at once.
   *
   * Applying this to the *desired* pose rather than the camera keeps the path
   * continuous — the existing smoothing then eases into it like any other
   * move, so the orbit never visibly bumps against an invisible wall.
   */
  private contain(): void {
    this.forward.subVectors(this.desiredTarget, this.desiredPosition).normalize();
    this.right.crossVectors(this.forward, WORLD_UP).normalize();
    this.up.crossVectors(this.right, this.forward).normalize();

    const tanV = Math.tan((this.camera.fov * Math.PI) / 360) * FRAME_FILL;
    const tanH = tanV * this.camera.aspect;

    let pull = 0;
    for (let i = 0; i < this.bounds.length; i += 3) {
      this.offset
        .set(this.bounds[i], this.bounds[i + 1], this.bounds[i + 2])
        .sub(this.desiredPosition);
      const depth = this.offset.dot(this.forward);
      pull = Math.max(
        pull,
        Math.abs(this.offset.dot(this.right)) / tanH - depth,
        Math.abs(this.offset.dot(this.up)) / tanV - depth
      );
    }
    if (pull > 0) this.desiredPosition.addScaledVector(this.forward, -pull);
  }

  update(elapsed: number, dt: number, driftAmp = 1): void {
    if (this.mode === "attract") {
      this.curve.getPoint((elapsed / ATTRACT_PERIOD) % 1, this.desiredPosition);
      // Gaze drifts slowly around the network's heart
      this.desiredTarget.set(
        1.5 + 4.5 * Math.sin(elapsed * 0.05),
        1.1 * Math.sin(elapsed * 0.073),
        0
      );
    } else {
      // Micro-drift keeps the locked framing breathing; the duck envelope
      // stills the camera during the fire cinematic (stillness = attention).
      const drift = cameraDrift(elapsed);
      this.desiredPosition.set(
        this.interactivePosition.x + drift.px * driftAmp,
        this.interactivePosition.y + drift.py * driftAmp,
        this.interactivePosition.z + drift.pz * driftAmp
      );
      this.desiredTarget.set(
        this.interactiveTarget.x + drift.tx * driftAmp,
        this.interactiveTarget.y + drift.ty * driftAmp,
        this.interactiveTarget.z
      );
    }
    this.contain();
    // Critically-damped exponential smoothing (frame-rate independent)
    const k = 1 - Math.exp(-1.8 * dt);
    this.camera.position.lerp(this.desiredPosition, k);
    this.smoothedTarget.lerp(this.desiredTarget, k);
    this.camera.lookAt(this.smoothedTarget);
  }
}
