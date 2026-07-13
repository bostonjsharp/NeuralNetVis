import { useEffect, useReducer, useRef, useState } from "react";
import samplesJson from "./assets/samples.json";
import weightsJson from "./assets/weights.json";
import {
  ATTRACT_FIRE_INTERVAL_MS,
  AUTO_FIRE_DELAY_MS,
  DRAW_PAD_SIZE,
  IDLE_MS,
  STAGE_HEIGHT,
  STAGE_WIDTH,
} from "./app/constants";
import { createAttractScheduler, type AttractScheduler } from "./app/attractLoop";
import { initialState, reduce, type InferenceSummary } from "./app/state";
import Hud from "./hud/Hud";
import Stage from "./hud/Stage";
import { startHandTracking, type GestureState } from "./gesture/handTracker";
import type { DrawPadHandle } from "./hud/DrawPad";
import { forwardPass, type ForwardResult } from "./nn/inference";
import { downsampleTo28, preprocessDrawing } from "./nn/preprocess";
import {
  decodeSamplePixels,
  loadNet,
  type SamplesJson,
  type WeightsJson,
} from "./nn/weights";
import { FIRE, FIRE_TOTAL_MS } from "./scene/fire";
import { createScene, type SceneApi } from "./scene/SceneManager";
import { useIdleReset } from "./useIdleReset";

const net = loadNet(weightsJson as WeightsJson);
const { samples } = samplesJson as SamplesJson;

/** When the HUD bars land, synced to the cinematic's output reveal. */
const OUTPUT_REVEAL_MS = FIRE.layerPop[2] * 1000;

export default function App() {
  const [state, dispatch] = useReducer(reduce, initialState);
  const stateRef = useRef(state);
  stateRef.current = state;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<SceneApi | null>(null);
  const schedulerRef = useRef<AttractScheduler | null>(null);
  schedulerRef.current ??= createAttractScheduler(samples.length);

  const [displayed, setDisplayed] = useState<InferenceSummary | null>(null);
  const [padReset, setPadReset] = useState(0);
  const [gestureActive, setGestureActive] = useState(false);
  const padRef = useRef<DrawPadHandle>(null);
  const gestureDrawingRef = useRef(false);
  const barsTimer = useRef(0);
  const cinematicTimer = useRef(0);
  const autoFireTimer = useRef(0);

  const attract = state.mode === "attract";

  // Scene lifecycle — the one place WebGL is born and dies
  useEffect(() => {
    const quality =
      new URLSearchParams(window.location.search).get("quality") === "low" ? "low" : "high";
    sceneRef.current = createScene(canvasRef.current!, net, quality);
    return () => {
      sceneRef.current?.dispose();
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    sceneRef.current?.setMode(attract ? "attract" : "interactive");
  }, [attract]);

  const showResultLater = (summary: InferenceSummary) => {
    window.clearTimeout(barsTimer.current);
    setDisplayed(null);
    barsTimer.current = window.setTimeout(() => setDisplayed(summary), OUTPUT_REVEAL_MS);
  };

  const runInference = (
    pixels: Float32Array,
    source: InferenceSummary["source"],
    sampleLabel?: number
  ): ForwardResult => {
    const result = forwardPass(net, pixels);
    const summary: InferenceSummary = {
      probs: Array.from(result.probs),
      argmax: result.argmax,
      source,
      sampleLabel,
    };
    sceneRef.current?.setInputPixels(pixels);
    sceneRef.current?.fire(result);
    dispatch({ type: "fire", summary });
    showResultLater(summary);
    return result;
  };

  // Attract loop: a shuffled sample digit flows through the network on a timer
  useEffect(() => {
    if (!attract) return;
    const fireSample = () => {
      const idx = schedulerRef.current!.next();
      runInference(decodeSamplePixels(samples[idx].pixels), "sample", samples[idx].label);
    };
    const first = window.setTimeout(fireSample, 800);
    const interval = window.setInterval(fireSample, ATTRACT_FIRE_INTERVAL_MS);
    return () => {
      window.clearTimeout(first);
      window.clearInterval(interval);
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
    sceneRef.current?.setInputPixels(new Float32Array(784));
    return () => {
      window.clearTimeout(autoFireTimer.current);
      window.clearTimeout(cinematicTimer.current);
      window.clearTimeout(barsTimer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attract]);

  useIdleReset(!attract, IDLE_MS, () => {
    dispatch({ type: "idleTimeout" });
    setDisplayed(null);
  });

  // Optional webcam hand control: ✊ fist = pen down, ✋ open = pen lifted
  // (reposition between strokes; the drawing auto-fires once the pen stays
  // up), ✌️ held = clear the pad. Fails silently into mouse-only when
  // there's no camera/permission.
  useEffect(() => {
    let dispose: (() => void) | undefined;
    let cancelled = false;
    let prevPose: GestureState["pose"] = "open";
    let attractPresence = 0;
    const onGesture = (g: GestureState) => {
      if (g.present) window.dispatchEvent(new Event("gesture-activity"));
      const mode = stateRef.current.mode;
      if (!g.present) {
        prevPose = "open";
        attractPresence = 0;
        if (gestureDrawingRef.current) {
          gestureDrawingRef.current = false;
          padRef.current?.penUp();
        }
        padRef.current?.setCursor(null);
        return;
      }
      const poseChanged = g.pose !== prevPose;
      prevPose = g.pose;
      if (mode === "attract") {
        // A hand held up briefly (~0.7s of frames) or a fist wakes the wall —
        // brief enough to feel instant, long enough that a passerby's hand
        // flashing through the camera doesn't yank the attract loop.
        attractPresence++;
        if (g.pose === "fist" || attractPresence >= 20) {
          attractPresence = 0;
          dispatch({ type: "userActive" });
        }
        return;
      }
      attractPresence = 0;
      const x = g.x * DRAW_PAD_SIZE;
      const y = g.y * DRAW_PAD_SIZE;
      if (g.pose === "fist") {
        if (gestureDrawingRef.current) {
          padRef.current?.penMove(x, y);
        } else {
          gestureDrawingRef.current = true;
          padRef.current?.penDown(x, y);
        }
      } else if (gestureDrawingRef.current) {
        gestureDrawingRef.current = false;
        padRef.current?.penUp();
      }
      if (g.pose === "two" && poseChanged) handleClear();
      padRef.current?.setCursor({ x, y, drawing: g.pose === "fist" && mode !== "infer" });
    };
    // Windows webcams are typically single-consumer — if another app holds
    // the camera at load, keep retrying instead of giving up forever.
    let retryTimer = 0;
    const attempt = () => {
      startHandTracking(onGesture)
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
    if (stateRef.current.mode === "infer" || stateRef.current.mode === "attract") return;
    runInference(pixels, source, sampleLabel);
    window.clearTimeout(cinematicTimer.current);
    cinematicTimer.current = window.setTimeout(
      () => dispatch({ type: "cinematicDone" }),
      FIRE_TOTAL_MS
    );
  };

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

  const handleClear = () => {
    if (stateRef.current.mode === "infer") return;
    window.clearTimeout(autoFireTimer.current);
    setPadReset((k) => k + 1);
    setDisplayed(null);
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
        displayed={displayed}
        padReset={padReset}
        padRef={padRef}
        gestureActive={gestureActive}
        onStrokeStart={handleStrokeStart}
        onDraw={handleDraw}
        onStrokeEnd={handleStrokeEnd}
        onSample={handleSample}
        onClear={handleClear}
      />
    </Stage>
  );
}
