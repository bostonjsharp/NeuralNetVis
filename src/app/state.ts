/**
 * Pure app state machine. The scene and HUD render it; App.tsx dispatches.
 *
 *   attract ──userActive──▶ draw ──fire──▶ infer ──cinematicDone──▶ result
 *      ▲                     ▲  ◀──────strokeStart/clear──────────────┘
 *      └────────────idleTimeout (from any interactive mode)───────────┘
 *
 * Attract mode also fires (sample digits on a timer) but never leaves
 * attract for it; `infer` locks input until the cinematic completes.
 */

export type Mode = "attract" | "draw" | "infer" | "result";

export interface InferenceSummary {
  probs: number[];
  argmax: number;
  source: "drawn" | "sample";
  /** True label when the input was a bundled sample digit. */
  sampleLabel?: number;
}

export interface AppState {
  mode: Mode;
  /** Latest completed or in-flight inference, for the HUD bars/verdict. */
  current: InferenceSummary | null;
}

export type AppEvent =
  | { type: "userActive" }
  | { type: "strokeStart" }
  | { type: "clear" }
  | { type: "fire"; summary: InferenceSummary }
  | { type: "cinematicDone" }
  | { type: "idleTimeout" };

export const initialState: AppState = { mode: "attract", current: null };

export function reduce(state: AppState, event: AppEvent): AppState {
  switch (event.type) {
    case "userActive":
      return state.mode === "attract" ? { mode: "draw", current: null } : state;

    case "strokeStart":
      // A fresh stroke over a verdict starts a new attempt; ignored mid-cinematic.
      return state.mode === "result" ? { mode: "draw", current: null } : state;

    case "clear":
      return state.mode === "draw" || state.mode === "result"
        ? { mode: "draw", current: null }
        : state;

    case "fire":
      if (state.mode === "attract") return { mode: "attract", current: event.summary };
      if (state.mode === "draw" || state.mode === "result") {
        return { mode: "infer", current: event.summary };
      }
      return state; // already inferring — input locked

    case "cinematicDone":
      return state.mode === "infer" ? { mode: "result", current: state.current } : state;

    case "idleTimeout":
      return state.mode === "attract" ? state : { mode: "attract", current: null };
  }
}
