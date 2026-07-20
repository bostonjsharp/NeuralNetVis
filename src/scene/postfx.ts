import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";

export interface PostFx {
  composer: EffectComposer;
  bloom: UnrealBloomPass;
  render(): void;
  /** Resize the whole chain to a new drawing-buffer size. */
  setSize(width: number, height: number): void;
  dispose(): void;
}

/** Kept modest: on a 2736px-wide wall a strong bloom is physically blinding. */
export const BLOOM_BASE_STRENGTH = 0.55;

/**
 * HDR render → bloom → tone map/sRGB. `msaa`/`bloomScale` come from the
 * adaptive quality ladder: MSAA on the composer target keeps the 1px
 * connection lines clean without an AA pass, and bloom runs at reduced
 * internal resolution — it's a haze, indistinguishable from full res but
 * the single biggest perf lever after MSAA itself.
 */
export function createPostFx(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  width: number,
  height: number,
  options: { msaa: number; bloomScale: number }
): PostFx {
  const { msaa, bloomScale } = options;
  const target = new THREE.WebGLRenderTarget(width, height, {
    type: THREE.HalfFloatType,
    samples: msaa,
  });
  const composer = new EffectComposer(renderer, target);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(
    new THREE.Vector2(width / bloomScale, height / bloomScale),
    BLOOM_BASE_STRENGTH,
    0.4,
    0.3
  );
  composer.addPass(bloom);
  composer.addPass(new OutputPass());
  const setSize = (w: number, h: number) => {
    composer.setSize(w, h);
    // The composer just told every pass the full buffer size — re-shrink
    // the bloom chain to its undersampled resolution.
    bloom.setSize(w / bloomScale, h / bloomScale);
  };
  return {
    composer,
    bloom,
    render: () => composer.render(),
    setSize,
    dispose: () => {
      composer.dispose();
      target.dispose();
    },
  };
}
