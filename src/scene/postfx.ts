import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";

export interface PostFx {
  composer: EffectComposer;
  bloom: UnrealBloomPass;
  render(): void;
  dispose(): void;
}

export const BLOOM_BASE_STRENGTH = 0.85;

/**
 * HDR render → bloom → tone map/sRGB. The wall is 2736×1216 (~3.3 MP);
 * bloom runs at reduced internal resolution — it's a haze, indistinguishable
 * from full res but the single biggest perf lever. MSAA on the composer
 * target keeps the 1px connection lines clean without an AA pass.
 */
export function createPostFx(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  width: number,
  height: number,
  quality: "high" | "low" = "high"
): PostFx {
  const target = new THREE.WebGLRenderTarget(width, height, {
    type: THREE.HalfFloatType,
    samples: quality === "high" ? 4 : 0,
  });
  const composer = new EffectComposer(renderer, target);
  composer.addPass(new RenderPass(scene, camera));
  const bloomScale = quality === "high" ? 2 : 4;
  const bloom = new UnrealBloomPass(
    new THREE.Vector2(width / bloomScale, height / bloomScale),
    BLOOM_BASE_STRENGTH,
    0.4,
    0.25
  );
  composer.addPass(bloom);
  composer.addPass(new OutputPass());
  return {
    composer,
    bloom,
    render: () => composer.render(),
    dispose: () => {
      composer.dispose();
      target.dispose();
    },
  };
}
