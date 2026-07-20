import { describe, expect, it } from "vitest";
import { FIRE, FIRE_TOTAL_MS, makeFireTimeline, stepsDue } from "./fire";

describe("makeFireTimeline", () => {
  it("reproduces the hand-tuned classic timeline exactly at 3 stages", () => {
    const fire = makeFireTimeline(3);
    expect(fire.stageStart).toEqual([0.3, 1.55, 2.55]);
    expect(fire.stageDur).toEqual([1.1, 0.85, 0.7]);
    expect(fire.layerPop).toEqual([1.3, 2.3, 3.15]);
    expect(fire.winnerFlare).toBeCloseTo(3.5, 10);
    expect(fire.total).toBeCloseTo(4.4, 10);
  });

  it.each([1, 2, 3])("keeps %i-stage timelines causally ordered", (stages) => {
    const fire = makeFireTimeline(stages);
    expect(fire.stageStart).toHaveLength(stages);
    expect(fire.stageDur).toHaveLength(stages);
    expect(fire.layerPop).toHaveLength(stages);
    for (let s = 0; s < stages; s++) {
      // A layer pops only after its pulse wave has been flying a while
      expect(fire.layerPop[s]).toBeGreaterThan(fire.stageStart[s]);
      if (s > 0) {
        // The next wave leaves after the previous layer lit up
        expect(fire.stageStart[s]).toBeGreaterThan(fire.layerPop[s - 1]);
      }
    }
    expect(fire.winnerFlare).toBeGreaterThan(fire.layerPop[stages - 1]);
    expect(fire.total).toBeGreaterThan(fire.winnerFlare);
  });

  it("rejects stage counts outside 1..3", () => {
    expect(() => makeFireTimeline(0)).toThrow();
    expect(() => makeFireTimeline(4)).toThrow();
  });

  it("keeps the legacy FIRE export equal to the 3-stage timeline", () => {
    expect(FIRE).toEqual(makeFireTimeline(3));
    expect(FIRE_TOTAL_MS).toBeCloseTo(4400, 10);
  });
});

describe("stepsDue", () => {
  const fire = makeFireTimeline(3);

  it("is 0 before the first wave lands", () => {
    expect(stepsDue(fire, 0)).toBe(0);
    expect(stepsDue(fire, fire.layerPop[0] - 0.01)).toBe(0);
  });

  it("increments exactly at each layerPop", () => {
    expect(stepsDue(fire, fire.layerPop[0])).toBe(1);
    expect(stepsDue(fire, fire.layerPop[1] - 0.01)).toBe(1);
    expect(stepsDue(fire, fire.layerPop[1])).toBe(2);
    expect(stepsDue(fire, fire.layerPop[2])).toBe(3);
  });

  it("stays at the stage count forever after", () => {
    expect(stepsDue(fire, 1000)).toBe(3);
  });

  it("handles negative time (before fire) and single-stage timelines", () => {
    expect(stepsDue(fire, -5)).toBe(0);
    const single = makeFireTimeline(1);
    expect(stepsDue(single, single.layerPop[0])).toBe(1);
    expect(stepsDue(single, 1000)).toBe(1);
  });
});
