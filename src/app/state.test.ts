import { describe, expect, it } from "vitest";
import { initialState, reduce, type AppState, type InferenceSummary } from "./state";

const summary: InferenceSummary = {
  probs: [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
  argmax: 0,
  source: "drawn",
};

const at = (
  mode: AppState["mode"],
  current: InferenceSummary | null = null,
  brainId = "classic"
): AppState => ({
  mode,
  current,
  brainId,
});

describe("app state machine", () => {
  it("starts in attract with the classic brain", () => {
    expect(initialState.mode).toBe("attract");
    expect(initialState.brainId).toBe("classic");
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
    for (const mode of ["draw", "infer", "result", "morph"] as const) {
      expect(reduce(at(mode, summary), { type: "idleTimeout" })).toEqual(at("attract"));
    }
  });

  it("idleTimeout in attract is a no-op (no state churn)", () => {
    const attract = at("attract", summary);
    expect(reduce(attract, { type: "idleTimeout" })).toBe(attract);
  });
});

describe("brain swapping", () => {
  it("selectBrain in draw starts a morph to the new brain, clearing the verdict", () => {
    const next = reduce(at("draw", summary), { type: "selectBrain", id: "wide" });
    expect(next).toEqual(at("morph", null, "wide"));
  });

  it("selectBrain in result starts a morph", () => {
    const next = reduce(at("result", summary), { type: "selectBrain", id: "tiny" });
    expect(next).toEqual(at("morph", null, "tiny"));
  });

  it("selectBrain in attract swaps in place without leaving attract", () => {
    const next = reduce(at("attract", summary), { type: "selectBrain", id: "linear" });
    expect(next).toEqual(at("attract", null, "linear"));
  });

  it("selectBrain is ignored mid-cinematic and mid-morph", () => {
    const infer = at("infer", summary);
    expect(reduce(infer, { type: "selectBrain", id: "wide" })).toBe(infer);
    const morph = at("morph", null, "tiny");
    expect(reduce(morph, { type: "selectBrain", id: "wide" })).toBe(morph);
  });

  it("selecting the already-active brain is a no-op", () => {
    const draw = at("draw", summary);
    expect(reduce(draw, { type: "selectBrain", id: "classic" })).toBe(draw);
  });

  it("morphDone lands in draw, ready for the re-fire", () => {
    const next = reduce(at("morph", null, "wide"), { type: "morphDone" });
    expect(next).toEqual(at("draw", null, "wide"));
  });

  it("morphDone outside morph is ignored", () => {
    const draw = at("draw");
    expect(reduce(draw, { type: "morphDone" })).toBe(draw);
  });

  it("morph locks drawing input", () => {
    const morph = at("morph", null, "wide");
    expect(reduce(morph, { type: "fire", summary })).toBe(morph);
    expect(reduce(morph, { type: "strokeStart" })).toBe(morph);
    expect(reduce(morph, { type: "clear" })).toBe(morph);
  });

  it("idleTimeout mid-morph returns to attract keeping the chosen brain", () => {
    const next = reduce(at("morph", null, "wide"), { type: "idleTimeout" });
    expect(next).toEqual(at("attract", null, "wide"));
  });
});
