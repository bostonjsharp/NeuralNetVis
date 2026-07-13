import type { InferenceSummary, Mode } from "../app/state";
import DrawPad from "./DrawPad";
import FactCards from "./FactCards";
import Labels from "./Labels";
import OutputBars, { verdictText } from "./OutputBars";
import SampleStrip from "./SampleStrip";

interface HudProps {
  mode: Mode;
  /** Result to show in the bars/verdict (null until the cinematic's output beat). */
  displayed: InferenceSummary | null;
  padReset: number;
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
  onStrokeStart,
  onDraw,
  onStrokeEnd,
  onSample,
  onClear,
}: HudProps) {
  const attract = mode === "attract";
  const busy = mode === "infer";
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
          <div className="hud-hint">Move the mouse to try it yourself</div>
          <FactCards />
        </>
      ) : (
        <>
          <div className="hud-panel">
            <DrawPad
              disabled={busy}
              resetKey={padReset}
              onStrokeStart={onStrokeStart}
              onDraw={onDraw}
              onStrokeEnd={onStrokeEnd}
            />
            <div className="hud-panel__side">
              <div className="hud-panel__title">{busy ? "Thinking…" : "Draw a digit 0–9"}</div>
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
