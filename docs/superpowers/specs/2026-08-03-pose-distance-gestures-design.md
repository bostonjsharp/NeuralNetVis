# Pose-Driven Gestures at Distance — Design

**Date:** 2026-08-03
**Status:** Approved (brainstorm 2026-08-03)

## Problem

The hand model (GestureRecognizer) fails at real footron distance (~2–4m): a
hand is only a few dozen pixels in the 640×360 feed, below what the model can
resolve. Visitors standing where they naturally stand get no interaction. The
body-pose model (`pose_landmarker_lite.task`) already runs in the stack and
reads reliably at that range — but today it drives exactly one thing, the
forearms-crossed ✕ clear.

## Goal

The full gesture vocabulary works at 2–4m using body pose, while close-range
behavior (desk testing, up-close visitors) stays byte-identical. No new
hardware, no new models, no changes to `GestureController`, `App.tsx`, or any
dwell logic.

## Non-Goals (YAGNI)

- 4K crop-and-rerun hand hybrid (revisit only if reach-to-draw fails on site).
- Depth-camera / sidecar architecture (Option C from the brainstorm).
- New app verbs. The vocabulary stays: wake, ✕ clear, draw, brain-cycle.

## Architecture: a two-tier FramePipeline

`FramePipeline.update()` gains a second signal source. Both tiers emit the
same `GestureState`, which is why nothing downstream changes.

- **Close tier (unchanged):** a hand passing `isCloseEnough` drives exactly
  today's path — fist pen, 👍 brain-cycle, palm cursor.
- **Far tier (new):** no close hand, but a body pose with visible
  shoulders/wrists → synthesize `GestureState` from pose landmarks.

Tier selection is per-frame with the existing `ABSENCE_GRACE_FRAMES`-style
hysteresis so a flickering hand detection doesn't strobe between tiers: the
close tier owns the state while a close hand exists or within its grace
window; the far tier takes over only after the close tier fully releases.

## Far-tier verb mapping

| Verb | Detection (pose landmarks, normalized camera space) |
|---|---|
| Wake (`raised`) | Active wrist above the same-side shoulder (y margin ≈ 0.05). The controller's existing 5s dwell stays the false-positive filter. |
| ✕ Clear (`crossed`) | `forearmsCrossed` — already pose-based, unchanged. |
| Cursor (`x`,`y`) | Active wrist through a shoulder-anchored `PosePadMapper` (below), then the existing `OneEuroFilter` pair. |
| Pen (`pose`) | `reachEvidence` (below) fed into the existing `PenLatch`. |
| Brain-cycle (`thumbsUp`) | Arm held straight out to the side: wrist beyond the shoulder laterally by > ~0.8 shoulder-widths with \|wrist.y − shoulder.y\| < ~0.5 shoulder-widths, latched via the snappy thumb-latch config; the controller's 1.5s dwell filters intent. Field keeps its `thumbsUp` name — it means "the brain-cycle verb is latched." |

Geometric separation of the three arm verbs: wake = up, draw = forward,
brain-cycle = sideways. No pair can be confused by the detector.

### Active-arm selection

Sticky, mirroring `pickPrimaryHand`: on acquisition an engaged wrist (raised,
out, or reaching) beats a merely higher one; afterwards the active arm keeps
identity (left/right is stable in pose output — no distance matching needed,
unlike anonymous hands). One exception, a deliberate post-review amendment:
an engaged other arm takes over immediately once the active arm rests past
its hip — without it a visitor who swapped hands was locked out until
tracking dropped. A vanished active wrist hands over only to an engaged arm;
otherwise the far tier's absence grace runs and selection restarts fresh.

### `reachEvidence` — pen down at distance

Reaching toward the screen drives the wrist's `z` strongly negative relative
to the shoulder. Evidence in `PenEvidence` shape:

- `fist` (pen-down drive): ramp of `(shoulder.z − wrist.z) / shoulderWidth`
  between a rest threshold and a full-reach threshold (constants, tuned on
  site via the debug overlay).
- `open` (pen-up drive): ramp the opposite way when the wrist z sits at or
  behind rest, or when the wrist drops below the hip line (arm hanging).
- Low visibility on wrist/shoulder → `NO_EVIDENCE`, exactly like an
  unreadable hand: the latch's decay lifts the pen on its own.

`PenLatch` is unchanged — it already integrates any 0..1 evidence stream;
this is a third evidence function beside `penEvidence` and `thumbEvidence`.

**Known risk:** pose `z` is MediaPipe's noisiest channel. The latch's
hysteresis + decay absorb flicker; if it still flaps on site, the fallback
evidence is 2D: wrist inside the pad box and above hip = drawing-capable,
wrist dropped = pen up. The overlay makes this a tuning call, not a redesign.

### `PosePadMapper` — cursor mapping

Same contract as `PadMapper` (drag-anchored virtual box, mirrored X, cursor
starts centered) but anchored to the body: box width = k × shoulderWidth
(k = 2.5 initially, an exported tuning constant), anchor seeded at the wrist
on acquisition, edge-dragging
identical. Shoulder width plays the role hand-span plays today, so the
mapping is distance- and position-neutral for the same reason.

## Module layout

New pure module `src/gesture/poseGesture.ts` — the pose-side twin of
`handGesture.ts`, which at ~380 lines shouldn't absorb another subsystem:

- Moves in: `POSE`, `PoseLandmark`, `segmentsIntersect`, `forearmsCrossed`
  (imports in `framePipeline.ts`/tests update; `handGesture.ts` re-exports
  nothing).
- New: extended `POSE` indices (shoulders 11/12, hips 23/24),
  `shoulderWidth`, `poseRaised`, `armOut`, `reachEvidence`, `PosePadMapper`,
  `pickActiveArm`, visibility guards.
- All constants exported and named — they are the on-site tuning surface.

`FramePipeline` gains far-tier state (active arm, pose filters/mapper/latches
— separate instances from the close tier's so tier handoff can't smear state).

## Inference scheduling

Today: hands every frame, pose every 6th (~10Hz). ~10Hz is fine for an 800ms
✕ hold but far too slow for a cursor. Scheduling becomes tier-driven:

- Close tier driving → hands stride 1, pose stride 6 (today's shape).
- Far tier driving (or nobody present) → **pose stride 1, hands stride 4**
  (the hand model's only far-tier job is noticing someone stepped close).

The pipeline's frame output reports which tier drove; the two pump loops
(worker and inline) pick next-frame strides from it. `POSE_FRAME_STRIDE`
becomes a pair of named stride constants beside the new hand stride. The
worker init protocol drops the fixed `poseStride` field.

The app-level `captureIntervalMs` attract-mode throttle is orthogonal and
unchanged.

## Debug overlay

Extend `HandDebugFrame` and the G-key overlay:

- Pose skeleton segments (shoulders–elbows–wrists) with per-point visibility.
- Reach charge bar (far-tier `PenLatch.level()`), like today's pen charge.
- Driving-tier indicator (CLOSE / FAR / none).

This is the on-site tuning instrument; it ships with the feature, not after.

## Testing

`FramePipeline` and `poseGesture.ts` stay pure — TDD throughout:

- `poseGesture.test.ts`: synthetic pose skeletons for raised / reach ramp /
  arm-out / visibility guards / mapper drag & mirroring / active-arm
  stickiness.
- `framePipeline.test.ts` additions: tier selection + hysteresis (close hand
  beats pose; handoff after grace), far-tier wake/draw/cycle end-to-end
  through the latch, absence handling with pose-only frames.
- Existing tests must pass unmodified — that is the "close tier unchanged"
  guarantee, enforced.

## Error handling

- Pose model missing (today's `poseLandmarker: null` path): far tier simply
  never engages; behavior degrades to exactly today's.
- Low-visibility pose frames yield `NO_EVIDENCE` / no cursor movement rather
  than guesses — same "uncertainty is silent" rule the pen latch was built on.
