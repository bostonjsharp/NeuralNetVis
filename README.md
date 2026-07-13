# Inside a Neural Network 🧠✨

An educational, cinematic 3D visualization of a **real** neural network for
the BYU footron video wall. A 784→16→16→10 MLP digit recognizer (trained on
MNIST to **95.6% test accuracy**, weights baked in) runs actual inference in
the browser — every glowing pulse you see is real math.

- **Attract loop:** the network continuously classifies real handwritten
  digits; signal comets stream along learned weight connections while the
  camera drifts on a cinematic orbit.
- **Walk-up interaction:** move the mouse, draw a digit on the pad (or tap a
  sample), and watch *your* drawing get normalized, streamed through the
  layers, and answered — with honest confidence ("got it, 32% sure" happens,
  and that's the lesson).
- **Color language:** orange connections excite the next neuron, blue ones
  inhibit; brightness = activation strength.

## Run it

```bash
npm install
npm run dev       # view at the printed localhost URL
npm test          # vitest suite (inference math, state machine, layout)
npm run build     # static build -> web/ (footron layout)
```

Dev viewports letterbox-scale the wall's fixed 2736×1216 canvas. Add
`?quality=low` for weaker GPUs (quarter-res bloom, no MSAA).

## Retraining the network

```bash
npm run train     # downloads MNIST once into .mnist-cache/, ~2 min pure-Node SGD
```

Exports `src/assets/weights.json` (base64 Float32 matrices) and
`src/assets/samples.json` (60 test digits, 6 per class, including one
deliberately ambiguous digit per class for the attract loop). Aborts below
92% test accuracy. `src/nn/assets.test.ts` cross-checks the committed assets
against the runtime forward pass so the two can never drift apart.

## Architecture

- `src/nn/` — pure TS: weight decoding, forward pass (returns per-layer
  activations for the viz), MNIST-style drawing normalization (bbox crop,
  20px scale, center-of-mass centering — mandatory for drawn-digit accuracy).
- `src/app/` — pure reducer state machine (`attract → draw → infer → result`)
  and the attract-loop sample scheduler.
- `src/scene/` — Three.js. `SceneManager.ts` is the only WebGL touchpoint;
  layout (`NetworkLayout.ts`) and cinematic timing (`fire.ts`) are pure and
  unit-tested. Instanced neurons, one-draw-call connection lines, a GPU
  particle pulse system, UnrealBloom at half resolution.
- `src/hud/` — React DOM overlay: draw pad, sample strip, probability bars,
  verdicts, rotating fact cards, zone labels.

## Footron packaging

`config.json`, `wide.jpg`, `thumb.jpg`, and the built `web/` folder follow
the [footron-data](https://github.com/BYU-PCCL/footron-data) experience
layout. Input is mouse-first for now; controller/phone mapping arrives with
footron integration.
