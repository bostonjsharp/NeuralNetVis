import { describe, expect, it } from "vitest";
import { initialState, reduce, type AppState, type InferenceSummary } from "./state";

const summary: InferenceSummary = {
  probs: [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
  argmax: 0,
  source: "drawn",
};

const at = (mode: AppState["mode"], current: InferenceSummary | null = null): AppState => ({
  mode,
  current,
});

describe("app state machine", () => {
  it("starts in attract", () => {
    expect(initialState.mode).toBe("attract");
  });

  it("attract → draw on user activity", () => {
    expect(reduce(initialState, { type: "userActive" }).mode).toBe("draw");
  });

  it("attract fire stays in attract but records the inference", () => {
    const next = reduce(initialState, { type: "fire", summary });
    expect(next.mode).toBe("attract");
    expect(next.current).toBe(summary);
  });

  it("draw → infer on fire", () => {
    const next = reduce(at("draw"), { type: "fire", summary });
    expect(next).toEqual(at("infer", summary));
  });

  it("infer → result on cinematicDone, keeping the inference", () => {
    const next = reduce(at("infer", summary), { type: "cinematicDone" });
    expect(next).toEqual(at("result", summary));
  });

  it("result → draw when a new stroke starts, clearing the verdict", () => {
    expect(reduce(at("result", summary), { type: "strokeStart" })).toEqual(at("draw"));
  });

  it("result → infer on refire (amended drawing)", () => {
    expect(reduce(at("result", summary), { type: "fire", summary }).mode).toBe("infer");
  });

  it("locks input during infer", () => {
    const infer = at("infer", summary);
    expect(reduce(infer, { type: "fire", summary })).toBe(infer);
    expect(reduce(infer, { type: "clear" })).toBe(infer);
    expect(reduce(infer, { type: "strokeStart" })).toBe(infer);
    expect(reduce(infer, { type: "userActive" })).toBe(infer);
  });

  it("clear resets draw and result to a blank draw", () => {
    expect(reduce(at("result", summary), { type: "clear" })).toEqual(at("draw"));
    expect(reduce(at("draw", summary), { type: "clear" })).toEqual(at("draw"));
  });

  it("idleTimeout returns every interactive mode to attract", () => {
    for (const mode of ["draw", "infer", "result"] as const) {
      expect(reduce(at(mode, summary), { type: "idleTimeout" })).toEqual(at("attract"));
    }
  });

  it("idleTimeout in attract is a no-op (no state churn)", () => {
    const attract = at("attract", summary);
    expect(reduce(attract, { type: "idleTimeout" })).toBe(attract);
  });
});
