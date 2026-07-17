import * as THREE from "three";
import { STAGE_HEIGHT, STAGE_WIDTH } from "../app/constants";
import { perf } from "../app/perf";
import type { ForwardResult } from "../nn/inference";
import type { Net } from "../nn/weights";
import { CameraRig, type CameraMode } from "./CameraRig";
import { ConnectionMesh } from "./ConnectionMesh";
import { installContextLossRecovery } from "./contextLossRecovery";
import {
  inputRamp,
  neuronPop,
  neuronReveal,
  stageGlow,
  winnerFlare,
} from "./fire";
import { InputPlane } from "./InputPlane";
import { NetworkLayout, buildLayout } from "./NetworkLayout";
import { NeuronField } from "./NeuronField";
import { OutputGlyphs } from "./OutputGlyphs";
import { createPostFx, BLOOM_BASE_STRENGTH } from "./postfx";
import { PulseSystem } from "./PulseSystem";
import { Starfield } from "./Starfield";
import { WinnerFlare } from "./WinnerFlare";

export interface SceneApi {
  /** Start the forward-pass cinematic for a computed inference. */
  fire(result: ForwardResult): void;
  setMode(mode: CameraMode): void;
  /** Live-mirror pixels onto the input plane (e.g. while drawing). */
  setInputPixels(pixels: Float32Array): void;
  dispose(): void;
}

/**
 * The only module that touches WebGL. Owns the renderer, the render loop,
 * and per-frame animation state; everything spatial/temporal it consumes
 * (layout, timeline curves) lives in pure modules.
 */
export function createScene(
  canvas: HTMLCanvasElement,
  net: Net,
  quality: "high" | "low" = "high"
): SceneApi {
  const layout: NetworkLayout = buildLayout(net);

  // Three handles a context loss that restores; this reboots the page when
  // the driver never restores it (the unattended-wall failure mode).
  const disposeContextWatch = installContextLossRecovery(canvas);

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false, // MSAA happens on the composer's render target
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(1); // the wall is exactly 2736×1216 — never oversample
  renderer.setSize(STAGE_WIDTH, STAGE_HEIGHT, false);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050510);
  const camera = new THREE.PerspectiveCamera(40, STAGE_WIDTH / STAGE_HEIGHT, 0.1, 300);

  const starfield = new Starfield();
  const inputPlane = new InputPlane();
  const neurons = new NeuronField(layout);
  const connections = new ConnectionMesh(layout);
  const pulses = new PulseSystem(layout);
  const glyphs = new OutputGlyphs(layout);
  const flare = new WinnerFlare();
  scene.add(starfield.points, inputPlane.mesh, connections.lines, neurons.mesh);
  scene.add(pulses.points, glyphs.group, flare.mesh);

  const rig = new CameraRig(camera);
  const post = createPostFx(renderer, scene, camera, STAGE_WIDTH, STAGE_HEIGHT, quality);

  const layerCounts = layout.layerPositions.map((p) => p.length / 3);

  // Active cinematic (persists after completion so the last result stays lit)
  let fireStart = -1e9;
  let fireResult: ForwardResult | null = null;
  let normalized: Float32Array[] = [];

  const clock = new THREE.Clock();
  let elapsed = 0;
  let raf = 0;

  function updateNeurons(tSinceFire: number) {
    const flareEnv = fireResult ? winnerFlare(tSinceFire) : 0;
    for (let layer = 0; layer < 3; layer++) {
      const count = layerCounts[layer];
      for (let i = 0; i < count; i++) {
        const instance = neurons.indexOf(layer, i);
        const shimmer = 0.02 * Math.sin(elapsed * 1.3 + instance * 0.71);
        let brightness = 0.07 + shimmer;
        let scale = 1;
        let warmth = 0;
        if (fireResult) {
          const reveal = neuronReveal(tSinceFire, layer, i, count);
          // Perceptual curve: keep weak activations visibly dimmer than
          // strong ones instead of letting bloom crush everything to white.
          const level = Math.pow(normalized[layer][i], 1.6);
          brightness += reveal * level * 1.05;
          scale = neuronPop(tSinceFire, layer, i, count) * (1 + 0.22 * level * reveal);
          if (layer === 2 && i === fireResult.argmax) {
            brightness += 0.4 * flareEnv;
            scale += 0.3 * flareEnv;
            warmth = Math.max(flareEnv, 0.4 * reveal * level);
          }
        }
        // Cool blue-white body, warmed toward gold for the winner
        const r = brightness * (0.55 + 0.65 * warmth);
        const g = brightness * (0.72 + 0.28 * warmth);
        const b = brightness * (1.0 - 0.45 * warmth);
        neurons.setInstance(instance, r, g, b, scale);
      }
    }
    neurons.commit();
    return flareEnv;
  }

  function frame() {
    raf = requestAnimationFrame(frame);
    const dt = Math.min(clock.getDelta(), 0.1);
    perf.recordFrame(dt * 1000);
    elapsed += dt;
    const tSinceFire = elapsed - fireStart;

    rig.update(elapsed, dt);
    starfield.update(elapsed);
    pulses.update(elapsed);
    inputPlane.setBrightness(fireResult ? inputRamp(tSinceFire) : 1);
    // Stage 0's 512 lines share one small screen region — halve its glow lift
    connections.setStageGlow(
      fireResult ? 0.5 * stageGlow(tSinceFire, 0) : 0,
      fireResult ? stageGlow(tSinceFire, 1) : 0,
      fireResult ? stageGlow(tSinceFire, 2) : 0
    );
    const flareEnv = updateNeurons(tSinceFire);
    const outputReveal = fireResult ? neuronReveal(tSinceFire, 2, 0, 1) : 0;
    glyphs.update(fireResult ? fireResult.probs : null, Math.max(outputReveal, fireResult ? 0.35 : 0.2), fireResult?.argmax ?? 0, flareEnv);
    flare.update(flareEnv, camera);
    post.bloom.strength = BLOOM_BASE_STRENGTH + 0.12 * flareEnv;

    post.render();
  }
  frame();

  return {
    fire(result) {
      fireResult = result;
      fireStart = elapsed;
      // Normalize each layer's activations to 0..1 for display brightness
      normalized = [1, 2, 3].map((layerIdx) => {
        const source = result.activations[layerIdx];
        const values =
          layerIdx === 3 ? result.probs : source; // output brightness follows probabilities
        let max = 0;
        for (let i = 0; i < values.length; i++) max = Math.max(max, values[i]);
        const norm = max > 0 ? 1 / max : 0;
        return Float32Array.from(values, (v) => v * norm);
      });
      inputPlane.setPixels(result.activations[0]);
      pulses.fire(result.activations, elapsed);
      flare.setTarget(neurons.positionOf(2, result.argmax));
    },
    setMode(mode) {
      rig.setMode(mode);
    },
    setInputPixels(pixels) {
      inputPlane.setPixels(pixels);
    },
    dispose() {
      cancelAnimationFrame(raf);
      disposeContextWatch();
      post.dispose();
      starfield.dispose();
      inputPlane.dispose();
      neurons.dispose();
      connections.dispose();
      pulses.dispose();
      glyphs.dispose();
      flare.dispose();
      renderer.dispose();
    },
  };
}
