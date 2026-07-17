import { useEffect, useReducer, useRef, useState } from "react";
import samplesJson from "./assets/samples.json";
import {
  ATTRACT_FIRE_INTERVAL_MS,
  AUTO_FIRE_DELAY_MS,
  DRAW_PAD_SIZE,
  IDLE_MS,
  STAGE_HEIGHT,
  STAGE_WIDTH,
} from "./app/constants";
import { attractPlan, createAttractScheduler, type AttractScheduler } from "./app/attractLoop";
import { initialState, reduce, type InferenceSummary } from "./app/state";
import Hud from "./hud/Hud";
import Stage from "./hud/Stage";
import { GestureController } from "./gesture/gestureController";
import { startHandTracking, type GestureState } from "./gesture/handTracker";
import DebugPanel, { type DebugData } from "./hud/DebugPanel";
import type { DrawPadHandle } from "./hud/DrawPad";
import { forwardPass, type ForwardResult } from "./nn/inference";
import { downsampleTo28, preprocessDrawing } from "./nn/preprocess";
import { DEFAULT_VARIANT_ID, getVariant, nextVariantId } from "./nn/variants";
import { decodeSamplePixels, type SamplesJson } from "./nn/weights";
import { makeFireTimeline } from "./scene/fire";
import { createScene, type SceneApi } from "./scene/SceneManager";
import { useIdleReset } from "./useIdleReset";

const { samples } = samplesJson as SamplesJson;

/** Cinematic timing depends on how many pulse stages the active brain has. */
const timelineFor = (brainId: string) =>
  makeFireTimeline(getVariant(brainId).net.shape.length - 1);

export default function App() {
  const [state, dispatch] = useReducer(reduce, initialState);
  const stateRef = useRef(state);
  stateRef.current = state;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<SceneApi | null>(null);
  const schedulerRef = useRef<AttractScheduler | null>(null);
  schedulerRef.current ??= createAttractScheduler(samples.length);

  const [displayed, setDisplayed] = useState<InferenceSummary | null>(null);
  const displayedRef = useRef(displayed);
  displayedRef.current = displayed;
  /** The previous brain's verdict on the same input (the comparison beat). */
  const [comparison, setComparison] = useState<InferenceSummary | null>(null);
  const prevSummaryRef = useRef<InferenceSummary | null>(null);
  const prevPixelsRef = useRef<Float32Array | null>(null);
  const [padReset, setPadReset] = useState(0);
  const [gestureActive, setGestureActive] = useState(false);
  const [wakeProgress, setWakeProgress] = useState(0);
  const padRef = useRef<DrawPadHandle>(null);
  const [debugOpen, setDebugOpen] = useState(
    () => new URLSearchParams(window.location.search).has("debug")
  );
  const debugOpenRef = useRef(debugOpen);
  debugOpenRef.current = debugOpen;
  const debugDataRef = useRef<DebugData>({ video: null, frame: null });
  const barsTimer = useRef(0);
  const cinematicTimer = useRef(0);
  const autoFireTimer = useRef(0);
  /** The most recent classified input — re-fired through a freshly swapped brain. */
  const lastInputRef = useRef<{
    pixels: Float32Array;
    source: InferenceSummary["source"];
    sampleLabel?: number;
  } | null>(null);

  const attract = state.mode === "attract";

  // Scene lifecycle — the one place WebGL is born and dies
  useEffect(() => {
    const quality =
      new URLSearchParams(window.location.search).get("quality") === "low" ? "low" : "high";
    sceneRef.current = createScene(
      canvasRef.current!,
      getVariant(DEFAULT_VARIANT_ID).net,
      quality
    );
    return () => {
      sceneRef.current?.dispose();
      sceneRef.current = null;
    };
  }, []);

  // Brain swap: cancel every pending timer, clear the verdict, and hand the
  // scene the new net; morphDone advances the state machine when it lands.
  const prevBrainRef = useRef(state.brainId);
  useEffect(() => {
    if (prevBrainRef.current === state.brainId) return;
    prevBrainRef.current = state.brainId;
    window.clearTimeout(autoFireTimer.current);
    window.clearTimeout(cinematicTimer.current);
    window.clearTimeout(barsTimer.current);
    // Keep the on-screen verdict as the comparison for the re-fire
    prevSummaryRef.current = displayedRef.current;
    setDisplayed(null);
    setComparison(null);
    const variant = getVariant(state.brainId);
    sceneRef.current?.setNet(variant.net, { inputTopK: variant.inputTopK }, () =>
      dispatch({ type: "morphDone" })
    );
  }, [state.brainId]);

  useEffect(() => {
    sceneRef.current?.setMode(attract ? "attract" : "interactive");
  }, [attract]);

  const showResultLater = (summary: InferenceSummary) => {
    window.clearTimeout(barsTimer.current);
    setDisplayed(null);
    setComparison(null);
    // A comparison only makes sense against the same input through a
    // different brain — consumed once, when the bars land.
    const against = prevSummaryRef.current;
    prevSummaryRef.current = null;
    // The HUD bars land in sync with the cinematic's output reveal
    const revealMs = timelineFor(summary.brainId).layerPop.at(-1)! * 1000;
    barsTimer.current = window.setTimeout(() => {
      setDisplayed(summary);
      setComparison(against && against.brainId !== summary.brainId ? against : null);
    }, revealMs);
  };

  const runInference = (
    pixels: Float32Array,
    source: InferenceSummary["source"],
    sampleLabel?: number
  ): ForwardResult => {
    const brainId = stateRef.current.brainId;
    const result = forwardPass(getVariant(brainId).net, pixels);
    const summary: InferenceSummary = {
      probs: Array.from(result.probs),
      argmax: result.argmax,
      source,
      sampleLabel,
      brainId,
    };
    // A NEW input invalidates any cross-brain comparison; a re-fire of the
    // exact same pixels array (morph landing, attract repeat) keeps it.
    if (prevPixelsRef.current !== pixels) prevSummaryRef.current = null;
    prevPixelsRef.current = pixels;
    lastInputRef.current = { pixels, source, sampleLabel };
    sceneRef.current?.setInputPixels(pixels);
    sceneRef.current?.fire(result);
    dispatch({ type: "fire", summary });
    showResultLater(summary);
    return result;
  };

  // Attract loop: a shuffled sample digit flows through the network on a
  // timer — and every third fire swaps brains and repeats the previous
  // digit, so the playground demos itself to passersby.
  const attractFireIndex = useRef(0);
  useEffect(() => {
    if (!attract) return;
    attractFireIndex.current = 0;
    let swapTimer = 0;
    const fireSample = () => {
      const plan = attractPlan(attractFireIndex.current++);
      if (plan.swapBrain && lastInputRef.current) {
        dispatch({ type: "selectBrain", id: nextVariantId(stateRef.current.brainId) });
        const held = lastInputRef.current;
        // Re-fire the held digit once the ~1.6s morph has landed
        swapTimer = window.setTimeout(() => {
          if (stateRef.current.mode !== "attract") return;
          runInference(held.pixels, held.source, held.sampleLabel);
        }, 1800);
        return;
      }
      const idx = schedulerRef.current!.next();
      runInference(decodeSamplePixels(samples[idx].pixels), "sample", samples[idx].label);
    };
    const first = window.setTimeout(fireSample, 800);
    const interval = window.setInterval(fireSample, ATTRACT_FIRE_INTERVAL_MS);
    return () => {
      window.clearTimeout(first);
      window.clearInterval(interval);
      window.clearTimeout(swapTimer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attract]);

  // Pointer activity hands the wall to the visitor. Movement must be
  // deliberate (a real drag, not the synthetic pointermove Chrome emits on
  // load when the cursor happens to rest over the window).
  useEffect(() => {
    if (!attract) return;
    let origin: { x: number; y: number } | null = null;
    let deliberateMoves = 0;
    const onMove = (e: PointerEvent) => {
      origin ??= { x: e.clientX, y: e.clientY };
      // A human drag produces a stream of far-from-origin moves; a synthetic
      // cursor jump (page load, automation snapshots) produces one or two.
      if (Math.hypot(e.clientX - origin.x, e.clientY - origin.y) > 40) deliberateMoves++;
      if (deliberateMoves >= 4) dispatch({ type: "userActive" });
    };
    const onDown = () => dispatch({ type: "userActive" });
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerdown", onDown);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerdown", onDown);
    };
  }, [attract]);

  // Entering interactive mode: blank pad, blank plane, clean verdict
  useEffect(() => {
    if (attract) return;
    // A bars timer from an attract-mode fire may still be pending — a stale
    // verdict would pop into the fresh draw panel without this.
    window.clearTimeout(barsTimer.current);
    setPadReset((k) => k + 1);
    setDisplayed(null);
    // The attract loop's last sample must not re-fire into a fresh visitor's
    // empty pad if they swap brains before drawing anything.
    lastInputRef.current = null;
    sceneRef.current?.setInputPixels(new Float32Array(784));
    return () => {
      window.clearTimeout(autoFireTimer.current);
      window.clearTimeout(cinematicTimer.current);
      window.clearTimeout(barsTimer.current);
    };
  }, [attract]);

  useIdleReset(!attract, IDLE_MS, () => {
    dispatch({ type: "idleTimeout" });
    setDisplayed(null);
  });

  // Optional webcam hand control, tuned for a camera 2+ meters away:
  // raise a hand and HOLD IT ~5s (a progress ring fills) to start, then
  // ✊ fist = pen down, ✋ open = pen lifted between strokes, and crossing
  // both arms in an ✕ clears the pad. Fails silently into mouse-only when
  // there's no camera/permission. All hold-timing and latching rules live
  // in GestureController (pure, unit-tested); this effect only interprets
  // its commands against dispatch/DrawPad.
  useEffect(() => {
    // ?nocam: A/B lever for hunting render hitches — runs the exact same app
    // with the vision pipeline (and its main-thread cost) absent.
    if (new URLSearchParams(window.location.search).has("nocam")) return;
    let dispose: (() => void) | undefined;
    let cancelled = false;
    const controller = new GestureController();
    // Idle-reset only needs to know a hand exists at second granularity —
    // dispatching per camera frame ran the listener 60×/s for nothing.
    let lastActivityDispatch = 0;
    const onGesture = (g: GestureState) => {
      const now = performance.now();
      if (g.present && now - lastActivityDispatch > 1000) {
        lastActivityDispatch = now;
        window.dispatchEvent(new Event("gesture-activity"));
      }
      for (const cmd of controller.update(g, stateRef.current.mode, now)) {
        switch (cmd.type) {
          case "wake":
            dispatch({ type: "userActive" });
            break;
          case "wakeProgress":
            setWakeProgress(cmd.value);
            break;
          case "clear":
            handleClear();
            break;
          case "cycleBrain":
            dispatch({ type: "selectBrain", id: nextVariantId(stateRef.current.brainId) });
            break;
          case "penDown":
            padRef.current?.penDown(cmd.x * DRAW_PAD_SIZE, cmd.y * DRAW_PAD_SIZE);
            break;
          case "penMove":
            padRef.current?.penMove(cmd.x * DRAW_PAD_SIZE, cmd.y * DRAW_PAD_SIZE);
            break;
          case "penUp":
            padRef.current?.penUp();
            break;
          case "cursor":
            padRef.current?.setCursor(
              cmd.cursor && {
                x: cmd.cursor.x * DRAW_PAD_SIZE,
                y: cmd.cursor.y * DRAW_PAD_SIZE,
                drawing: cmd.cursor.drawing,
              }
            );
            break;
        }
      }
    };
    // Windows webcams are typically single-consumer — if another app holds
    // the camera at load, keep retrying instead of giving up forever.
    let retryTimer = 0;
    const attempt = () => {
      startHandTracking(onGesture, {
        attachVideo: (video) => {
          debugDataRef.current.video = video;
        },
        wantFrames: () => debugOpenRef.current,
        onFrame: (frame) => {
          debugDataRef.current.frame = frame;
        },
      })
        .then((stop) => {
          if (cancelled) stop();
          else {
            dispose = stop;
            setGestureActive(true);
          }
        })
        .catch((err: unknown) => {
          console.warn(
            "hand tracking unavailable (no camera, permission denied, or camera in use) — retrying in 15s:",
            err instanceof Error ? err.message : err
          );
          if (!cancelled) retryTimer = window.setTimeout(attempt, 15_000);
        });
    };
    attempt();
    return () => {
      cancelled = true;
      window.clearTimeout(retryTimer);
      dispose?.();
    };
  }, []);

  const fireInteractive = (
    pixels: Float32Array,
    source: InferenceSummary["source"],
    sampleLabel?: number
  ) => {
    const { mode } = stateRef.current;
    if (mode === "infer" || mode === "attract" || mode === "morph") return;
    runInference(pixels, source, sampleLabel);
    window.clearTimeout(cinematicTimer.current);
    cinematicTimer.current = window.setTimeout(
      () => dispatch({ type: "cinematicDone" }),
      timelineFor(stateRef.current.brainId).total * 1000
    );
  };

  // The comparison beat: when a morph lands (morph → draw) with an input
  // still on the pad, the same digit immediately re-fires through the new
  // brain so the visitor sees the confidence difference.
  const prevModeRef = useRef(state.mode);
  useEffect(() => {
    const prev = prevModeRef.current;
    prevModeRef.current = state.mode;
    if (prev === "morph" && state.mode === "draw" && lastInputRef.current) {
      const { pixels, source, sampleLabel } = lastInputRef.current;
      fireInteractive(pixels, source, sampleLabel);
    }
  });

  const handleStrokeStart = () => {
    window.clearTimeout(autoFireTimer.current);
    dispatch({ type: "strokeStart" });
  };

  const handleDraw = (pixels: Float32Array, w: number, h: number) => {
    sceneRef.current?.setInputPixels(downsampleTo28(pixels, w, h));
  };

  const handleStrokeEnd = (pixels: Float32Array, w: number, h: number) => {
    sceneRef.current?.setInputPixels(downsampleTo28(pixels, w, h));
    window.clearTimeout(autoFireTimer.current);
    autoFireTimer.current = window.setTimeout(() => {
      const input = preprocessDrawing(pixels, w, h);
      let ink = 0;
      for (const v of input) ink += v;
      if (ink < 1) return; // effectively empty — don't classify noise
      fireInteractive(input, "drawn");
    }, AUTO_FIRE_DELAY_MS);
  };

  const handleSample = (sampleIndex: number) => {
    window.clearTimeout(autoFireTimer.current);
    setPadReset((k) => k + 1);
    const sample = samples[sampleIndex];
    fireInteractive(decodeSamplePixels(sample.pixels), "sample", sample.label);
  };

  // Dev-only: G toggles the gesture debug overlay (also via ?debug);
  // B cycles through the brain variants.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "g" || e.key === "G") setDebugOpen((open) => !open);
      if (e.key === "b" || e.key === "B") {
        dispatch({ type: "selectBrain", id: nextVariantId(stateRef.current.brainId) });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const handleClear = () => {
    const { mode } = stateRef.current;
    if (mode === "infer" || mode === "morph") return;
    window.clearTimeout(autoFireTimer.current);
    setPadReset((k) => k + 1);
    setDisplayed(null);
    lastInputRef.current = null;
    sceneRef.current?.setInputPixels(new Float32Array(784));
    dispatch({ type: "clear" });
  };

  return (
    <Stage>
      <canvas
        ref={canvasRef}
        className="scene-canvas"
        width={STAGE_WIDTH}
        height={STAGE_HEIGHT}
      />
      <div className="vignette" />
      <Hud
        mode={state.mode}
        brainId={state.brainId}
        displayed={displayed}
        comparison={comparison}
        padReset={padReset}
        padRef={padRef}
        gestureActive={gestureActive}
        wakeProgress={wakeProgress}
        onStrokeStart={handleStrokeStart}
        onDraw={handleDraw}
        onStrokeEnd={handleStrokeEnd}
        onSample={handleSample}
        onSelectBrain={(id) => dispatch({ type: "selectBrain", id })}
        onClear={handleClear}
      />
      {debugOpen && <DebugPanel dataRef={debugDataRef} />}
    </Stage>
  );
}
