import { describe, expect, it } from "vitest";
import {
  AdaptiveQuality,
  QUALITY_LADDER,
  renderResolution,
  startLevelFor,
} from "./adaptiveQuality";

describe("renderResolution", () => {
  it("renders at full native size on the wall itself (scale 1, dpr 1)", () => {
    expect(renderResolution(1, 1, 1)).toEqual({ width: 2736, height: 1216 });
  });

  it("shrinks the buffer to the displayed pixel count on a smaller viewport", () => {
    // A 1536-wide laptop window at dpr 1.25 displays 1920 physical px
    const { width, height } = renderResolution(1536 / 2736, 1.25, 1);
    expect(width).toBe(1920);
    expect(height).toBe(Math.round(1216 * (1536 / 2736) * 1.25));
  });

  it("never exceeds native resolution even on high-dpr oversized viewports", () => {
    expect(renderResolution(1, 2, 1)).toEqual({ width: 2736, height: 1216 });
    expect(renderResolution(1.5, 1, 1)).toEqual({ width: 2736, height: 1216 });
  });

  it("applies the quality ladder's render scale on top of the display scale", () => {
    const { width } = renderResolution(1, 1, 0.75);
    expect(width).toBe(Math.round(2736 * 0.75));
  });

  it("never collapses below a readable floor on tiny embed viewports", () => {
    const { width, height } = renderResolution(0.05, 1, 1);
    expect(width).toBeGreaterThanOrEqual(2736 * 0.3);
    expect(height).toBeGreaterThanOrEqual(1216 * 0.3);
  });
});

describe("startLevelFor", () => {
  it("maps the quality query param onto ladder entry points", () => {
    expect(startLevelFor("high")).toBe(0);
    expect(startLevelFor("low")).toBe(2);
    expect(startLevelFor("auto")).toBe(0);
  });
});

describe("AdaptiveQuality", () => {
  const good = { p95: 14, frames: 240 };
  const bad = { p95: 30, frames: 240 };

  it("stays put while frames are within budget", () => {
    const q = new AdaptiveQuality({ startLevel: 0 });
    expect(q.update(good, 20_000)).toBeNull();
    expect(q.level).toBe(0);
  });

  it("steps down one rung after sustained over-budget frames", () => {
    const q = new AdaptiveQuality({ startLevel: 0 });
    expect(q.update(bad, 20_000)).toBe(1);
    expect(q.level).toBe(1);
  });

  it("ignores readings until the warmup window has passed", () => {
    const q = new AdaptiveQuality({ startLevel: 0 });
    expect(q.update(bad, 1_000)).toBeNull();
    expect(q.level).toBe(0);
  });

  it("ignores sparse samples that cannot support a p95", () => {
    const q = new AdaptiveQuality({ startLevel: 0 });
    expect(q.update({ p95: 30, frames: 20 }, 20_000)).toBeNull();
  });

  it("waits out the cooldown before stepping again", () => {
    const q = new AdaptiveQuality({ startLevel: 0 });
    expect(q.update(bad, 20_000)).toBe(1);
    // Old slow frames still in the ring right after the change — ignore them
    expect(q.update(bad, 24_000)).toBeNull();
    expect(q.update(bad, 40_000)).toBe(2);
  });

  it("stops at the bottom rung", () => {
    const q = new AdaptiveQuality({ startLevel: QUALITY_LADDER.length - 1 });
    expect(q.update(bad, 60_000)).toBeNull();
    expect(q.level).toBe(QUALITY_LADDER.length - 1);
  });

  it("never adapts when locked (explicit ?quality= pin)", () => {
    const q = new AdaptiveQuality({ startLevel: 0, locked: true });
    expect(q.update(bad, 60_000)).toBeNull();
    expect(q.level).toBe(0);
  });

  it("descends the ladder monotonically toward cheaper settings", () => {
    // Each rung must cost no more than the previous: MSAA and bloom
    // resolution only ever decrease, render scale only ever shrinks.
    for (let i = 1; i < QUALITY_LADDER.length; i++) {
      expect(QUALITY_LADDER[i].msaa).toBeLessThanOrEqual(QUALITY_LADDER[i - 1].msaa);
      expect(QUALITY_LADDER[i].bloomScale).toBeGreaterThanOrEqual(
        QUALITY_LADDER[i - 1].bloomScale
      );
      expect(QUALITY_LADDER[i].renderScale).toBeLessThanOrEqual(
        QUALITY_LADDER[i - 1].renderScale
      );
    }
  });
});
