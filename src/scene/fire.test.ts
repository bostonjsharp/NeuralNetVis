import { describe, expect, it } from "vitest";
import { FIRE, FIRE_TOTAL_MS, makeFireTimeline } from "./fire";

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
