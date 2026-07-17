/**
 * Pure frame-time statistics for diagnosing render hitches — no DOM, no
 * timers. The scene loop records each frame's duration, instrumented spots
 * (MediaPipe inference) record named marks, and the debug overlay polls
 * snapshot() at its own rate.
 */

export interface MarkStats {
  last: number;
  worst: number;
  count: number;
}

export interface PerfSnapshot {
  /** Frames currently held in the ring buffer. */
  frames: number;
  p50: number;
  p95: number;
  worst: number;
  /** Frames over the 60fps budget (16.7ms) in the buffer. */
  over17: number;
  /** Frames over 25ms — hitches a visitor can see. */
  over25: number;
  marks: Record<string, MarkStats>;
}

export interface PerfMonitor {
  recordFrame(ms: number): void;
  recordMark(name: string, ms: number): void;
  snapshot(): PerfSnapshot;
}

const FRAME_BUDGET_MS = 16.7;
const HITCH_MS = 25;

export function createPerfMonitor(capacity = 240): PerfMonitor {
  const ring = new Float32Array(capacity);
  let head = 0;
  let count = 0;
  const marks: Record<string, MarkStats> = {};

  const percentile = (sorted: number[], q: number) =>
    sorted.length === 0 ? 0 : sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];

  return {
    recordFrame(ms) {
      ring[head] = ms;
      head = (head + 1) % capacity;
      count = Math.min(count + 1, capacity);
    },
    recordMark(name, ms) {
      const mark = marks[name];
      if (mark) {
        mark.last = ms;
        mark.worst = Math.max(mark.worst, ms);
        mark.count++;
      } else {
        marks[name] = { last: ms, worst: ms, count: 1 };
      }
    },
    snapshot() {
      const frames = Array.from(ring.subarray(0, count)).sort((a, b) => a - b);
      let over17 = 0;
      let over25 = 0;
      for (const ms of frames) {
        if (ms > FRAME_BUDGET_MS) over17++;
        if (ms > HITCH_MS) over25++;
      }
      return {
        frames: count,
        p50: percentile(frames, 0.5),
        p95: percentile(frames, 0.95),
        worst: frames.length ? frames[frames.length - 1] : 0,
        over17,
        over25,
        marks: Object.fromEntries(Object.entries(marks).map(([k, v]) => [k, { ...v }])),
      };
    },
  };
}

/** Shared app-wide monitor: scene loop and gesture tracker feed it. */
export const perf = createPerfMonitor();
