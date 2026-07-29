# Ambient Motion — Design Spec

**Date:** 2026-07-29
**Status:** Approved approach A (layer into existing systems), pending spec review

## Problem

Interactive mode is the stillest moment of the exhibit. The instant a visitor
engages, the camera locks to a fixed framing (`CameraRig.interactivePosition`),
and until they fire, the only motion is a near-invisible per-neuron shimmer
(±0.02 brightness) and a starfield rotating at 0.004 rad/s. The scene feels
flat exactly when someone is looking at it up close — both while drawing and
across the whole interactive mode.

## Goal

Three layers of subtle motion — *felt, not watched*:

1. **Camera micro-drift** — interactive framing breathes instead of locking rigid.
2. **Coherent neuron breathing** — brightness/scale swells travel across the
   network like a slow tide.
3. **Ambient sparks + draw reactivity** — rare wandering sparks along
   connections when idle; while the visitor inks, sparks concentrate on
   stage-0 edges leaving recently-inked input pixels and the stage-0 wiring
   warms slightly.

All ambient motion is multiplied by a single **duck envelope** so the fire
cinematic and the brain-swap morph always own the stage. Nothing ambient may
read as real signal propagation — the answer is earned by the wavefront, and
ambient sparks must read as idle static, never computation.

## Architecture

### New pure module: `src/scene/ambient.ts` (sibling of `fire.ts`)

All math, no WebGL, unit-testable. Exports:

- `cameraDrift(elapsed)` → offset vector for the interactive camera.
  Lissajous-style: three sines with incommensurate periods (~11s / 17s / 23s).
  Amplitude ≈ 0.35 world units (position), ≈ 0.15 (look-target).
- `breathe(elapsed, x, y)` → 0..1 traveling wave, period ~8s; phase depends on
  world position so swells roll across the network rather than twinkling
  per-neuron.
- `AmbientState` — holds `excitement` (0..1). `bumpExcitement(amount)` on fresh
  ink; exponential decay, τ ≈ 1.5s.
- `duckEnvelope(fireActive, tSinceFire, fireTotal, morphing)` → 0..1 master
  ambient volume. Ramps to ~0.15 over 0.3s when a fire starts; recovers to 1
  over ~1s after `fireTotal`; hard 0 during a morph.
- `SparkScheduler` — pure spawn logic for `AmbientSparks` (see below).
  Injected RNG for deterministic tests; Poisson-style rate accumulator
  (`rate × dt`), tracks free pool slots, never exceeds pool size.

All tuning constants (amplitudes, periods, rates, τ, duck floor) are named
exports at the top of `ambient.ts` — single place to retune on the wall.

### New scene object: `src/scene/AmbientSparks.ts`

Miniature sibling of `PulseSystem`:

- One `THREE.Points` pool, **~48 particles** — one extra draw call.
- Each spark travels one connection edge (GPU interpolation via
  `aStart`/`aEnd` attributes, eased in the vertex shader), tinted warm/cool by
  weight sign — but **smaller and dimmer** than pulse comets: static
  electricity, not signal.
- Spark lifetime ≤ 0.8s.
- Spawn behavior (decided by `SparkScheduler`):
  - **Idle:** ~2–3 alive at once, random edges anywhere.
  - **Drawing (high excitement):** up to ~8–10 alive, ≥80% biased to stage-0
    edges whose source input pixel was recently inked.
- Rebuilt on brain swap inside `swapSubsystems`, same lifecycle as
  `PulseSystem`.

### Changes to existing files

**`CameraRig.ts`** — interactive desired position/target become
`interactive* + cameraDrift(elapsed) × driftAmp`, where `driftAmp` is the duck
envelope passed from the scene. Existing critically-damped smoothing absorbs
any harshness. Attract mode untouched.

**`SceneManager.ts`** — four edits:

1. Replace the per-neuron shimmer term with `breathe`: brightness +~0.03 and
   scale ±2%, both × duck.
2. `setInputPixels` diffs incoming pixels against the previous buffer:
   positive deltas bump `excitement`; indices that gained ink go into a small
   recently-inked ring buffer for the scheduler's stage-0 bias. Runs on stroke
   events only.
3. `frame()` computes duck once per frame, decays excitement, and feeds both
   to camera rig, breathing, sparks, and glow.
4. Faint stage-0 glow lift (`~0.15 × excitement × duck`) via the existing
   `setStageGlow` path while drawing.

**No `SceneApi` changes.** The scene observes drawing through
`setInputPixels`, which the app already calls on every stroke update. The app
layer does not know ambient motion exists.

## Per-frame data flow

```
setInputPixels(pixels)          frame() @ 60fps
  │ diff vs previous              │
  ├─ bumpExcitement(Σ new ink)    ├─ duck = duckEnvelope(fireActive, tSinceFire, fire.total, morphing)
  └─ push inked indices           ├─ excitement decays toward 0 (τ ≈ 1.5s)
     into ring buffer             ├─ CameraRig.update(..., driftAmp = duck)
                                  ├─ updateNeurons: brightness += breathe × 0.03 × duck
                                  │                 scale × (1 + 0.02 × breathe × duck)
                                  ├─ sparks.update(now, scheduler.spawns(excitement × duck, inkedRing))
                                  └─ connections.setStageGlow(stage0 += 0.15 × excitement × duck)
```

## Ducking rules

| Scene state | Ambient level |
|---|---|
| Idle / drawing, no fire | 1.0 |
| Fire cinematic in flight | → 0.15 over 0.3s (camera drift ducks too — stillness reads as attention) |
| Cinematic done (`tSinceFire > fire.total`) | → 1.0 over ~1s |
| Brain-swap morph | 0 |
| Attract mode | breathing + idle sparks active; camera owned by the spline. (The attract loop's sample load does blip excitement via `setInputPixels`, but the simultaneous fire ducks ambient to 0.15 and excitement decays in ~1.5s — no visible effect.) |

## Edge cases

- **Brain swap:** `AmbientSparks` disposed/rebuilt with the new layout;
  recently-inked ring buffer cleared; excitement reset to 0.
- **Fire arriving mid-spark:** live sparks finish their ≤0.8s flight at ducked
  alpha — no single-frame pop.
- **Pad clear:** all-zero pixels; positive-delta-only diffing means clearing
  never bumps excitement.
- **Adaptive quality:** ambient cost is negligible (48 points + a few adds in
  the already-running neuron loop); it gets no quality-ladder rung and
  survives all rungs. The duck scalar is the kill-switch if wall profiling
  ever disagrees.

## Testing

New `ambient.test.ts` over the pure module:

1. `duckEnvelope` — 1 at idle; ≤0.15 shortly after fire start; recovers after
   `fireTotal`; 0 while morphing.
2. Excitement — bump clamps to 1; decays below 0.05 within ~3τ; zero-delta
   input never bumps.
3. `breathe` — bounded [0,1]; distant neurons differ in phase; deterministic.
4. `cameraDrift` — bounded by amplitude; actually moves; no visible repeat
   over several periods.
5. `SparkScheduler` — deterministic with injected RNG; idle concurrency ≈
   target; ≥80% stage-0 inked-edge bias at high excitement; respects pool
   size; recycles slots after lifetime.

`AmbientSparks` buffer-writing glue stays thin (same testing posture as
`PulseSystem`).

## Non-goals

- No activity mimicking real signal propagation (fire cinematic owns that
  story).
- No new app-level state, props, or `SceneApi` methods.
- No shader rewrites of `NeuronField`, `ConnectionMesh`, or `PulseSystem`.
- No post-verdict afterglow (considered, not selected).
