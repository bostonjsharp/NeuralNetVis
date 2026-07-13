import type { InferenceSummary } from "../app/state";

/**
 * The 10 output probabilities as crisp DOM bars — exact percentages beside
 * the in-scene glow. `result` is null until the cinematic's output stage,
 * so the bars land in sync with the 3D reveal.
 */
export default function OutputBars({ result }: { result: InferenceSummary | null }) {
  return (
    <div className="output-bars" data-live={result !== null}>
      {Array.from({ length: 10 }, (_, digit) => {
        const p = result ? result.probs[digit] : 0;
        const isWinner = result !== null && digit === result.argmax;
        return (
          <div key={digit} className={`output-bar${isWinner ? " output-bar--winner" : ""}`}>
            <span className="output-bar__digit">{digit}</span>
            <span className="output-bar__track">
              <span className="output-bar__fill" style={{ width: `${p * 100}%` }} />
            </span>
            <span className="output-bar__pct">{result ? `${Math.round(p * 100)}%` : ""}</span>
          </div>
        );
      })}
    </div>
  );
}

/** Plain-language confidence verdict — honest when the network is unsure. */
export function verdictText(result: InferenceSummary): string {
  const confidence = result.probs[result.argmax];
  const runnerUp = result.probs
    .map((p, digit) => ({ p, digit }))
    .filter(({ digit }) => digit !== result.argmax)
    .sort((a, b) => b.p - a.p)[0];
  if (confidence >= 0.7) {
    return `It thinks this is a ${result.argmax} — ${Math.round(confidence * 100)}% sure`;
  }
  if (confidence >= 0.45) {
    return `Probably a ${result.argmax} (${Math.round(confidence * 100)}%) — but ${
      runnerUp.digit
    } got ${Math.round(runnerUp.p * 100)}%`;
  }
  return `It's really not sure! Best guess: ${result.argmax} at ${Math.round(confidence * 100)}%`;
}
