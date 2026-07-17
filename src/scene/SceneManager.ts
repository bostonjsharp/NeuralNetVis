import * as THREE from "three";
import { STAGE_HEIGHT, STAGE_WIDTH } from "../app/constants";
import { perf } from "../app/perf";
import type { ForwardResult } from "../nn/inference";
import type { Net } from "../nn/weights";
import { CameraRig, type CameraMode } from "./CameraRig";
import { ConnectionMesh } from "./ConnectionMesh";
import { installContextLossRecovery } from "./contextLossRecovery";
import {
  clamp01,
  inputRamp,
  makeFireTimeline,
  neuronPop,
  neuronReveal,
  smoothstep,
  stageGlow,
  winnerFlare,
} from "./fire";
import { InputPlane } from "./InputPlane";
import { buildLayout, type LayoutOptions, type NetworkLayout } from "./NetworkLayout";
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
  /**
   * Morph the network into a different brain: the middle implodes, the
   * subsystems rebuild for the new topology, the new middle cascades in.
   * The input plane and output column never move. Calls onDone when the
   * reshape has finished (≈1.6s).
   */
  setNet(net: Net, options: LayoutOptions, onDone: () => void): void;
  dispose(): void;
}

/** Old hidden layers implode… */
const MORPH_OUT_S = 0.55;
/** …then the new topology cascades in, done by this total. */
const MORPH_TOTAL_S = 1.6;

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
  let layout: NetworkLayout = buildLayout(net);
  let fire = makeFireTimeline(layout.edges.length);

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
  let neurons = new NeuronField(layout);
  let connections = new ConnectionMesh(layout);
  let pulses = new PulseSystem(layout, fire);
  // The output column is position-identical across every brain — the glyphs
  // are built once and survive all swaps untouched.
  const glyphs = new OutputGlyphs(layout);
  const flare = new WinnerFlare();
  scene.add(starfield.points, inputPlane.mesh, connections.lines, neurons.mesh);
  scene.add(pulses.points, glyphs.group, flare.mesh);

  const rig = new CameraRig(camera);
  const post = createPostFx(renderer, scene, camera, STAGE_WIDTH, STAGE_HEIGHT, quality);

  let layerCounts = layout.layerPositions.map((p) => p.length / 3);

  // Active cinematic (persists after completion so the last result stays lit)
  let fireStart = -1e9;
  let fireResult: ForwardResult | null = null;
  let normalized: Float32Array[] = [];

  // Active brain-swap morph
  let morphState: {
    start: number;
    net: Net;
    options: LayoutOptions;
    onDone: () => void;
    swapped: boolean;
  } | null = null;

  function swapSubsystems(nextNet: Net, options: LayoutOptions): void {
    scene.remove(neurons.mesh, connections.lines, pulses.points);
    neurons.dispose();
    connections.dispose();
    pulses.dispose();
    layout = buildLayout(nextNet, options);
    fire = makeFireTimeline(layout.edges.length);
    neurons = new NeuronField(layout);
    connections = new ConnectionMesh(layout);
    pulses = new PulseSystem(layout, fire);
    scene.add(connections.lines, neurons.mesh, pulses.points);
    layerCounts = layout.layerPositions.map((p) => p.length / 3);
  }

  const clock = new THREE.Clock();
  let elapsed = 0;
  let raf = 0;

  /** Hidden-neuron scale during the morph: implode together, cascade back in. */
  function morphScale(
    morph: { phase: "out" | "in"; u: number } | null,
    layer: number,
    i: number,
    count: number
  ): number {
    if (!morph || layer === layerCounts.length - 1) return 1; // output persists
    if (morph.phase === "out") return 1 - smoothstep(0, 1, morph.u);
    // Reuse the reveal-cascade feel: top of the column leads the reweave
    const stagger = (i / Math.max(1, count - 1)) * 0.45;
    return smoothstep(0, 0.55, morph.u - stagger);
  }

  function updateNeurons(
    tSinceFire: number,
    morph: { phase: "out" | "in"; u: number } | null
  ) {
    const flareEnv = fireResult ? winnerFlare(fire, tSinceFire) : 0;
    const lastLayer = layerCounts.length - 1;
    for (let layer = 0; layer < layerCounts.length; layer++) {
      const count = layerCounts[layer];
      for (let i = 0; i < count; i++) {
        const instance = neurons.indexOf(layer, i);
        const shimmer = 0.02 * Math.sin(elapsed * 1.3 + instance * 0.71);
        let brightness = 0.07 + shimmer;
        let scale = 1;
        let warmth = 0;
        if (fireResult) {
          const reveal = neuronReveal(fire, tSinceFire, layer, i, count);
          // Perceptual curve: keep weak activations visibly dimmer than
          // strong ones instead of letting bloom crush everything to white.
          const level = Math.pow(normalized[layer][i], 1.6);
          brightness += reveal * level * 1.05;
          scale = neuronPop(fire, tSinceFire, layer, i, count) * (1 + 0.22 * level * reveal);
          if (layer === lastLayer && i === fireResult.argmax) {
            brightness += 0.4 * flareEnv;
            scale += 0.3 * flareEnv;
            warmth = Math.max(flareEnv, 0.4 * reveal * level);
          }
        }
        // Cool blue-white body, warmed toward gold for the winner
        const r = brightness * (0.55 + 0.65 * warmth);
        const g = brightness * (0.72 + 0.28 * warmth);
        const b = brightness * (1.0 - 0.45 * warmth);
        neurons.setInstance(instance, r, g, b, scale * morphScale(morph, layer, i, count));
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

    // Brain-swap morph: implode → swap subsystems at the midpoint → reweave
    let morph: { phase: "out" | "in"; u: number } | null = null;
    let edgeFade = 1;
    if (morphState) {
      const mt = elapsed - morphState.start;
      if (mt < MORPH_OUT_S) {
        morph = { phase: "out", u: mt / MORPH_OUT_S };
        edgeFade = 1 - morph.u;
      } else {
        if (!morphState.swapped) {
          swapSubsystems(morphState.net, morphState.options);
          morphState.swapped = true;
        }
        morph = { phase: "in", u: clamp01((mt - MORPH_OUT_S) / (MORPH_TOTAL_S - MORPH_OUT_S)) };
        edgeFade = morph.u;
        if (mt >= MORPH_TOTAL_S) {
          const { onDone } = morphState;
          morphState = null;
          morph = null;
          edgeFade = 1;
          onDone();
        }
      }
    }

    rig.update(elapsed, dt);
    starfield.update(elapsed);
    pulses.update(elapsed);
    inputPlane.setBrightness(fireResult ? inputRamp(tSinceFire) : 1);
    connections.setFade(edgeFade);
    // Stage 0's many lines share one small screen region — halve its glow lift
    connections.setStageGlow(
      layout.edges.map((_, stage) =>
        fireResult ? (stage === 0 ? 0.5 : 1) * stageGlow(fire, tSinceFire, stage) : 0
      )
    );
    const flareEnv = updateNeurons(tSinceFire, morph);
    const outputReveal = fireResult
      ? neuronReveal(fire, tSinceFire, layerCounts.length - 1, 0, 1)
      : 0;
    glyphs.update(fireResult ? fireResult.probs : null, Math.max(outputReveal, fireResult ? 0.35 : 0.2), fireResult?.argmax ?? 0, flareEnv);
    flare.update(flareEnv, camera);
    post.bloom.strength = BLOOM_BASE_STRENGTH + 0.12 * flareEnv;

    post.render();
  }
  frame();

  return {
    fire(result) {
      // The morph advances on rAF frames while callers schedule re-fires on
      // wall-clock timers — under load a fire can arrive mid-morph. Land the
      // pending swap first so the scene's topology matches the result.
      if (morphState) {
        if (!morphState.swapped) swapSubsystems(morphState.net, morphState.options);
        const { onDone } = morphState;
        morphState = null;
        onDone();
      }
      // A result computed for a different topology than the built scene
      // must never animate — it would index out of every buffer.
      if (result.activations.length - 1 !== layerCounts.length) return;
      fireResult = result;
      fireStart = elapsed;
      // Normalize each layer's activations to 0..1 for display brightness
      normalized = layerCounts.map((_, j) => {
        const layerIdx = j + 1; // activations[0] is the input plane
        const source = result.activations[layerIdx];
        const values =
          j === layerCounts.length - 1 ? result.probs : source; // output brightness follows probabilities
        let max = 0;
        for (let i = 0; i < values.length; i++) max = Math.max(max, values[i]);
        const norm = max > 0 ? 1 / max : 0;
        return Float32Array.from(values, (v) => v * norm);
      });
      inputPlane.setPixels(result.activations[0]);
      pulses.fire(result.activations, elapsed);
      flare.setTarget(neurons.positionOf(layerCounts.length - 1, result.argmax));
    },
    setMode(mode) {
      rig.setMode(mode);
    },
    setNet(nextNet, options, onDone) {
      if (morphState) {
        // Land the pending morph before starting over — an orphaned onDone
        // would strand the app's state machine in "morph" forever.
        if (!morphState.swapped) swapSubsystems(morphState.net, morphState.options);
        morphState.onDone();
      }
      // A lit result about the old brain must not survive the swap
      fireResult = null;
      fireStart = -1e9;
      morphState = { start: elapsed, net: nextNet, options, onDone, swapped: false };
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
