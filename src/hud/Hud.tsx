import type { Ref } from "react";
import type { InferenceSummary, Mode } from "../app/state";
import DrawPad, { type DrawPadHandle } from "./DrawPad";
import FactCards from "./FactCards";
import Labels from "./Labels";
import OutputBars, { verdictText } from "./OutputBars";
import SampleStrip from "./SampleStrip";

interface HudProps {
  mode: Mode;
  /** Result to show in the bars/verdict (null until the cinematic's output beat). */
  displayed: InferenceSummary | null;
  padReset: number;
  padRef: Ref<DrawPadHandle>;
  /** True when webcam hand tracking is running. */
  gestureActive: boolean;
  /** 0..1 fill of the raise-hand-to-start hold (attract mode only). */
  wakeProgress: number;
  onStrokeStart: () => void;
  onDraw: (pixels: Float32Array, width: number, height: number) => void;
  onStrokeEnd: (pixels: Float32Array, width: number, height: number) => void;
  onSample: (sampleIndex: number) => void;
  onClear: () => void;
}

export default function Hud({
  mode,
  displayed,
  padReset,
  padRef,
  gestureActive,
  wakeProgress,
  onStrokeStart,
  onDraw,
  onStrokeEnd,
  onSample,
  onClear,
}: HudProps) {
  const attract = mode === "attract";
  const busy = mode === "infer" || mode === "morph";
  return (
    <div className={`hud${attract ? " hud--attract" : ""}`}>
      <Labels />

      {attract ? (
        <>
          <header className="hud-title">
            <h1>Inside a Neural Network</h1>
            <p>A real artificial brain — 13,002 learned connections, thinking in front of you</p>
          </header>
          {displayed && (
            <div className="hud-verdict hud-verdict--attract">{fullVerdict(displayed)}</div>
          )}
          <div className="hud-hint">
            {gestureActive
              ? "Raise your hand ✋ and hold it there — or move the mouse — to try it"
              : "Move the mouse to try it yourself"}
          </div>
          {wakeProgress > 0 && (
            <div className="wake-ring-wrap">
              <div className="wake-ring-face">
                <div
                  className="wake-ring"
                  style={{
                    background: `conic-gradient(var(--cool) ${wakeProgress * 360}deg, rgba(80, 110, 180, 0.22) 0deg)`,
                  }}
                />
                <span className="wake-ring__hand">✋</span>
              </div>
              <div className="wake-ring__label">
                Keep your hand up… {Math.ceil((1 - wakeProgress) * 5)}
              </div>
            </div>
          )}
          <FactCards />
        </>
      ) : (
        <>
          <div className="hud-panel">
            <DrawPad
              ref={padRef}
              disabled={busy}
              resetKey={padReset}
              onStrokeStart={onStrokeStart}
              onDraw={onDraw}
              onStrokeEnd={onStrokeEnd}
            />
            <div className="hud-panel__side">
              <div className="hud-panel__title">
                {mode === "morph" ? "Rewiring…" : busy ? "Thinking…" : "Draw a digit 0–9"}
              </div>
              {gestureActive && !busy && (
                <div className="hud-panel__gesture">
                  ✊ draws · ✋ lifts the pen · cross your arms ✕ to clear
                </div>
              )}
              <button className="hud-panel__clear" onClick={onClear} disabled={busy}>
                Clear
              </button>
              <div className="hud-panel__try">…or try a real handwritten digit:</div>
              <SampleStrip disabled={busy} onPick={onSample} />
            </div>
          </div>
          <aside className="hud-output">
            {displayed && <div className="hud-verdict">{fullVerdict(displayed)}</div>}
            <OutputBars result={displayed} />
          </aside>
        </>
      )}
    </div>
  );
}

function fullVerdict(result: InferenceSummary): string {
  if (result.source === "sample" && result.sampleLabel !== undefined) {
    const confidence = Math.round(result.probs[result.argmax] * 100);
    return result.argmax === result.sampleLabel
      ? `Shown a handwritten ${result.sampleLabel} — got it, ${confidence}% sure`
      : `Shown a handwritten ${result.sampleLabel} — it guessed ${result.argmax}. Even AI gets fooled!`;
  }
  return verdictText(result);
}
