import type { Ref } from "react";
import type { InferenceSummary, Mode } from "../app/state";
import { getVariant } from "../nn/variants";
import BrainCards from "./BrainCards";
import DrawPad, { type DrawPadHandle } from "./DrawPad";
import FactCards from "./FactCards";
import Labels from "./Labels";
import OutputBars, { verdictText } from "./OutputBars";
import SampleStrip from "./SampleStrip";

interface HudProps {
  mode: Mode;
  /** The active brain variant id (see nn/variants). */
  brainId: string;
  /** Result to show in the bars/verdict (null until the cinematic's output beat). */
  displayed: InferenceSummary | null;
  /** The previous brain's verdict on the same input, when one exists. */
  comparison: InferenceSummary | null;
  padReset: number;
  padRef: Ref<DrawPadHandle>;
  /** True when webcam hand tracking is running. */
  gestureActive: boolean;
  /** 0..1 fill of the raise-hand hold (wake in attract, brain-cycle here). */
  wakeProgress: number;
  onStrokeStart: () => void;
  onDraw: (pixels: Float32Array, width: number, height: number) => void;
  onStrokeEnd: (pixels: Float32Array, width: number, height: number) => void;
  onSample: (sampleIndex: number) => void;
  onSelectBrain: (id: string) => void;
  onClear: () => void;
}

export default function Hud({
  mode,
  brainId,
  displayed,
  comparison,
  padReset,
  padRef,
  gestureActive,
  wakeProgress,
  onStrokeStart,
  onDraw,
  onStrokeEnd,
  onSample,
  onSelectBrain,
  onClear,
}: HudProps) {
  const attract = mode === "attract";
  const busy = mode === "infer" || mode === "morph";
  const variant = getVariant(brainId);
  return (
    <div className={`hud${attract ? " hud--attract" : ""}`}>
      <Labels variant={variant} />

      {attract ? (
        <>
          <header className="hud-title">
            <h1>Inside a Neural Network</h1>
            <p>
              A real artificial brain — {variant.paramCount.toLocaleString()} learned
              connections, thinking in front of you
            </p>
          </header>
          {displayed && (
            <div className="hud-verdict hud-verdict--attract">
              {fullVerdict(displayed)}
              {comparison && (
                <span className="hud-verdict__compare">{compareText(comparison, displayed)}</span>
              )}
            </div>
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
          <BrainCards activeId={brainId} disabled readOnly onSelect={onSelectBrain} />
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
                  ✊ draws · ✋ lifts the pen · 👍 switches brains · cross your arms ✕ to clear
                </div>
              )}
              <button className="hud-panel__clear" onClick={onClear} disabled={busy}>
                Clear
              </button>
              <div className="hud-panel__try">…or try a real handwritten digit:</div>
              <SampleStrip disabled={busy} onPick={onSample} />
            </div>
          </div>
          <BrainCards activeId={brainId} disabled={busy} onSelect={onSelectBrain} />
          {wakeProgress > 0 && (
            <div className="wake-ring-wrap wake-ring-wrap--brain">
              <div className="wake-ring-face">
                <div
                  className="wake-ring"
                  style={{
                    background: `conic-gradient(var(--warm) ${wakeProgress * 360}deg, rgba(80, 110, 180, 0.22) 0deg)`,
                  }}
                />
                <span className="wake-ring__hand">👍</span>
              </div>
              <div className="wake-ring__label">Keep that 👍 up…</div>
            </div>
          )}
          <aside className="hud-output">
            {displayed && (
              <div className="hud-verdict">
                {fullVerdict(displayed)}
                {comparison && (
                  <span className="hud-verdict__compare">
                    {compareText(comparison, displayed)}
                  </span>
                )}
              </div>
            )}
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

/** The cross-brain beat: same digit, two brains, honest numbers. */
function compareText(previous: InferenceSummary, current: InferenceSummary): string {
  const pct = (s: InferenceSummary) => Math.round(s.probs[s.argmax] * 100);
  const prevLabel = getVariant(previous.brainId).label;
  return previous.argmax === current.argmax
    ? `The ${prevLabel} brain also said ${previous.argmax} — but was ${pct(previous)}% sure vs ${pct(current)}% now.`
    : `The ${prevLabel} brain said ${previous.argmax} (${pct(previous)}%) — this one says ${current.argmax} (${pct(current)}%).`;
}
