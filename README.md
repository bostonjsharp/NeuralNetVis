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
- **Phone controls (footron):** scan the wall's QR code and a draw pad opens
  on your phone (`controls/lib/index.js`, served by footron). Finger-draw a
  digit — strokes render locally at full rate and stream to the wall at
  ~30Hz over footron's messaging router, where they replay through the exact
  same `DrawPadHandle` path the mouse uses. Connecting wakes the wall;
  buttons clear the pad and cycle brains. The client arms only when footron
  passes `?ftMsgUrl` (or `?ftmsg=1` against the dev mock: `npm run mock`,
  then draw at http://localhost:8089/).
- **Color language:** orange connections excite the next neuron, blue ones
  inhibit; brightness = activation strength.
- **Architecture playground (brain swap):** four pre-trained brains ship in
  the app — Straight-through `784→10` (92.2%), Tiny `784→8→10` (93.0%),
  Classic `784→16→16→10` (95.6%), and Wide `784→32→32→10` (97.3%). Pick a
  brain card (or tap "switch brain" on your phone) and the network
  visibly *rewires* — the middle implodes and the new topology cascades in
  while the input plane and output column hold still — then the same digit
  re-fires through the new brain with a cross-brain comparison verdict
  ("The Tiny brain said 3 at 54% — this one says 3 at 97%"). The attract
  loop swaps brains every third fire and repeats the previous digit so the
  feature demos itself.

## Run it

```bash
npm install
npm run dev       # view at the printed localhost URL
npm test          # vitest suite (inference math, state machine, layout)
npm run build     # static build -> web/ (footron layout)
```

Dev viewports letterbox-scale the wall's fixed 2736×1216 canvas; the
drawing buffer renders at the displayed pixel count, never above it.
Effect quality is governed by measured frame times — sustained over-budget
p95 walks down a ladder (MSAA off → quarter-res bloom → 0.75× render
scale), logged to the console. `?quality=high` or `?quality=low` pins a
rung and disables adaptation.

## Retraining the networks

```bash
npm run train           # the classic brain (also refreshes samples.json)
npm run train -- wide   # one variant
npm run train -- all    # every variant, ~8 min pure-Node SGD
```

Downloads MNIST once into `.mnist-cache/`. Exports
`src/assets/weights-<id>.json` per variant (base64 Float32 matrices; classic
also keeps the original `weights.json` name) and `src/assets/samples.json`
(60 test digits, 6 per class, including one deliberately ambiguous digit per
class, always picked by the classic net). Each variant has an honest
per-architecture accuracy gate. `src/nn/assets.test.ts` cross-checks every
committed brain against the runtime forward pass so they can never drift.

## Architecture

- `src/nn/` — pure TS: weight decoding, forward pass (returns per-layer
  activations for the viz), MNIST-style drawing normalization (bbox crop,
  20px scale, center-of-mass centering — mandatory for drawn-digit accuracy).
- `src/app/` — pure reducer state machine (`attract → draw → infer → result`)
  and the attract-loop sample scheduler.
- `src/input/` — phone-controls protocol (pure, unit-tested) and the
  connection to footron's messaging router; `controls/lib/` is the panel
  footron serves to the phone, `dev/messaging-mock.mjs` a local router mock.
- `src/scene/` — Three.js. `SceneManager.ts` is the only WebGL touchpoint;
  layout (`NetworkLayout.ts`) and cinematic timing (`fire.ts`) are pure and
  unit-tested. Instanced neurons, one-draw-call connection lines, a GPU
  particle pulse system, UnrealBloom at half resolution.
- `src/hud/` — React DOM overlay: draw pad, sample strip, probability bars,
  verdicts, rotating fact cards, zone labels.

## Footron packaging

`config.json`, `wide.jpg`, `thumb.jpg`, and the built `web/` folder follow
the [footron-data](https://github.com/BYU-PCCL/footron-data) experience
layout; `controls/lib/index.js` is the phone-controls panel footron-data's
build compiles against footron-web.
