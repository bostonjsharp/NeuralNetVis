import { describe, expect, it } from "vitest";
import { createAttractScheduler } from "./attractLoop";

function seededRand(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("attract scheduler", () => {
  it("visits every sample once per cycle", () => {
    const scheduler = createAttractScheduler(10, seededRand(1));
    const seen = new Set(Array.from({ length: 10 }, () => scheduler.next()));
    expect(seen.size).toBe(10);
  });

  it("never repeats a sample across a cycle boundary", () => {
    for (let seed = 1; seed <= 25; seed++) {
      const scheduler = createAttractScheduler(5, seededRand(seed));
      let prev = -1;
      for (let i = 0; i < 50; i++) {
        const next = scheduler.next();
        expect(next).not.toBe(prev);
        prev = next;
      }
    }
  });

  it("handles a single-sample pool without spinning forever", () => {
    const scheduler = createAttractScheduler(1, seededRand(7));
    expect(scheduler.next()).toBe(0);
    expect(scheduler.next()).toBe(0);
  });

  it("rejects an empty pool", () => {
    expect(() => createAttractScheduler(0)).toThrow();
  });
});
