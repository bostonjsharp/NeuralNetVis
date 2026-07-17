/**
 * Pure app state machine. The scene and HUD render it; App.tsx dispatches.
 *
 *   attract ──userActive──▶ draw ──fire──▶ infer ──cinematicDone──▶ result
 *      ▲                     ▲  ◀──────strokeStart/clear──────────────┘
 *      │                     │◀──morphDone── morph ◀──selectBrain── draw/result
 *      └────────────idleTimeout (from any interactive mode)───────────┘
 *
 * Attract mode also fires (sample digits on a timer) but never leaves
 * attract for it; `infer` locks input until the cinematic completes, and
 * `morph` locks it while the network reshapes into a different brain.
 */

export type Mode = "attract" | "draw" | "infer" | "result" | "morph";

export interface InferenceSummary {
  probs: number[];
  argmax: number;
  source: "drawn" | "sample";
  /** True label when the input was a bundled sample digit. */
  sampleLabel?: number;
  /** Which brain produced this verdict (for cross-brain comparisons). */
  brainId: string;
}

export interface AppState {
  mode: Mode;
  /** Latest completed or in-flight inference, for the HUD bars/verdict. */
  current: InferenceSummary | null;
  /** Which brain variant the scene is showing (see nn/variants). */
  brainId: string;
}

export type AppEvent =
  | { type: "userActive" }
  | { type: "strokeStart" }
  | { type: "clear" }
  | { type: "fire"; summary: InferenceSummary }
  | { type: "cinematicDone" }
  | { type: "selectBrain"; id: string }
  | { type: "morphDone" }
  | { type: "idleTimeout" };

export const initialState: AppState = { mode: "attract", current: null, brainId: "classic" };

export function reduce(state: AppState, event: AppEvent): AppState {
  const { brainId } = state;
  switch (event.type) {
    case "userActive":
      return state.mode === "attract" ? { mode: "draw", current: null, brainId } : state;

    case "strokeStart":
      // A fresh stroke over a verdict starts a new attempt; ignored mid-cinematic.
      return state.mode === "result" ? { mode: "draw", current: null, brainId } : state;

    case "clear":
      return state.mode === "draw" || state.mode === "result"
        ? { mode: "draw", current: null, brainId }
        : state;

    case "fire":
      if (state.mode === "attract") {
        return { mode: "attract", current: event.summary, brainId };
      }
      if (state.mode === "draw" || state.mode === "result") {
        return { mode: "infer", current: event.summary, brainId };
      }
      return state; // already inferring or morphing — input locked

    case "cinematicDone":
      return state.mode === "infer"
        ? { mode: "result", current: state.current, brainId }
        : state;

    case "selectBrain":
      if (event.id === brainId) return state;
      // A stale verdict about another brain would be a lie — always clear it.
      if (state.mode === "attract") {
        return { mode: "attract", current: null, brainId: event.id };
      }
      if (state.mode === "draw" || state.mode === "result") {
        return { mode: "morph", current: null, brainId: event.id };
      }
      return state; // infer/morph — locked

    case "morphDone":
      return state.mode === "morph" ? { mode: "draw", current: null, brainId } : state;

    case "idleTimeout":
      return state.mode === "attract" ? state : { mode: "attract", current: null, brainId };
  }
}
