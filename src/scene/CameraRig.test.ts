import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { STAGE_HEIGHT, STAGE_WIDTH } from "../app/constants";
import { VARIANTS } from "../nn/variants";
import { CameraRig } from "./CameraRig";
import { buildLayout } from "./NetworkLayout";

const DT = 1 / 60;
/** One full attract lap is 90s; three laps cover the gaze drift's own beat. */
const LAP = 90;

const makeRig = (variantIndex: number) => {
  const variant = VARIANTS[variantIndex];
  const layout = buildLayout(variant.net, { inputTopK: variant.inputTopK });
  const camera = new THREE.PerspectiveCamera(40, STAGE_WIDTH / STAGE_HEIGHT, 0.1, 300);
  return { rig: new CameraRig(camera, layout), camera, layout };
};

/** Worst |x| or |y| in normalized device coords over the layout's bounds.
 *  1.0 is the frame edge, so anything above it has left the screen. */
function worstNdc(camera: THREE.PerspectiveCamera, bounds: Float32Array): number {
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();
  const p = new THREE.Vector3();
  let worst = 0;
  for (let i = 0; i < bounds.length; i += 3) {
    p.set(bounds[i], bounds[i + 1], bounds[i + 2]).project(camera);
    worst = Math.max(worst, Math.abs(p.x), Math.abs(p.y));
  }
  return worst;
}

describe("CameraRig framing", () => {
  it.each(VARIANTS.map((v, i) => [v.label, i] as const))(
    "keeps the whole network on screen through an attract lap (%s)",
    (_label, index) => {
      const { rig, camera, layout } = makeRig(index);
      let worst = 0;
      for (let step = 0; step * DT < LAP * 3; step++) {
        rig.update(step * DT, DT);
        // Skip the first second — the rig starts at a curve point with no
        // smoothing history and is allowed to ease into frame.
        if (step * DT > 1) worst = Math.max(worst, worstNdc(camera, layout.bounds));
      }
      expect(worst).toBeLessThanOrEqual(1);
    }
  );

  it("keeps the network on screen across an attract → interactive switch", () => {
    const { rig, camera, layout } = makeRig(2);
    let elapsed = 0;
    const settle = (seconds: number) => {
      for (let step = 0; step * DT < seconds; step++) {
        rig.update(elapsed, DT);
        elapsed += DT;
        if (elapsed > 1) expect(worstNdc(camera, layout.bounds)).toBeLessThanOrEqual(1);
      }
    };
    // Switch at several points around the lap — each leg approaches from a
    // different distance, and the lerp toward the fixed framing has to stay
    // inside the frame the whole way across.
    for (const hold of [12, 21, 36, 66]) {
      rig.setMode("attract");
      settle(hold);
      rig.setMode("interactive");
      settle(8);
    }
  });

  it("pulls back far enough for a taller brain than it was built with", () => {
    const { rig, camera } = makeRig(0); // Straight-through: no hidden column
    const wide = VARIANTS[3];
    const wideLayout = buildLayout(wide.net, { inputTopK: wide.inputTopK });
    rig.setLayout(wideLayout);
    let elapsed = 0;
    for (let step = 0; step * DT < LAP; step++, elapsed += DT) {
      rig.update(elapsed, DT);
      if (elapsed > 1) expect(worstNdc(camera, wideLayout.bounds)).toBeLessThanOrEqual(1);
    }
  });
});
