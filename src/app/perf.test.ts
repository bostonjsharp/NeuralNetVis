import { describe, expect, it } from "vitest";
import { createPerfMonitor } from "./perf";

describe("perf monitor", () => {
  it("reports zeros before any frames are recorded", () => {
    const perf = createPerfMonitor();
    expect(perf.snapshot()).toEqual({
      frames: 0,
      p50: 0,
      p95: 0,
      worst: 0,
      over17: 0,
      over25: 0,
      marks: {},
    });
  });

  it("computes percentiles and worst over recorded frame times", () => {
    const perf = createPerfMonitor();
    // 1..100ms — p50 lands mid-pack, p95 near the top
    for (let ms = 1; ms <= 100; ms++) perf.recordFrame(ms);
    const snap = perf.snapshot();
    expect(snap.frames).toBe(100);
    expect(snap.p50).toBeGreaterThanOrEqual(50);
    expect(snap.p50).toBeLessThanOrEqual(51);
    expect(snap.p95).toBeGreaterThanOrEqual(95);
    expect(snap.p95).toBeLessThanOrEqual(96);
    expect(snap.worst).toBe(100);
  });

  it("counts long frames beyond the 60fps and hitch thresholds", () => {
    const perf = createPerfMonitor();
    perf.recordFrame(10); // fine
    perf.recordFrame(17); // missed 16.7ms budget
    perf.recordFrame(30); // visible hitch
    const snap = perf.snapshot();
    expect(snap.over17).toBe(2);
    expect(snap.over25).toBe(1);
  });

  it("evicts the oldest frames once the ring buffer is full", () => {
    const perf = createPerfMonitor(4);
    for (const ms of [100, 100, 100, 100]) perf.recordFrame(ms);
    for (const ms of [1, 1, 1, 1]) perf.recordFrame(ms);
    const snap = perf.snapshot();
    expect(snap.frames).toBe(4);
    expect(snap.worst).toBe(1); // the 100ms frames aged out
  });

  it("tracks named marks with last value, worst, and count", () => {
    const perf = createPerfMonitor();
    perf.recordMark("gesture", 4);
    perf.recordMark("gesture", 9);
    perf.recordMark("gesture", 6);
    perf.recordMark("pose", 12);
    const { marks } = perf.snapshot();
    expect(marks.gesture).toEqual({ last: 6, worst: 9, count: 3 });
    expect(marks.pose).toEqual({ last: 12, worst: 12, count: 1 });
  });

  it("keeps marks independent of frame statistics", () => {
    const perf = createPerfMonitor();
    perf.recordMark("gesture", 50);
    expect(perf.snapshot().frames).toBe(0);
    expect(perf.snapshot().worst).toBe(0);
  });
});
