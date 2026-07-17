import { describe, expect, it } from "vitest";
import {
  ABSENCE_GRACE_FRAMES,
  FramePipeline,
  type PipelineFrameInput,
} from "./framePipeline";
import type { GestureCategory, Landmark, PoseLandmark } from "./handGesture";

/** A readable 21-landmark hand: palm at (x, y), wrist `span` below it. */
const makeHand = (x: number, y: number, span = 0.04): Landmark[] => {
  const landmarks = Array.from({ length: 21 }, () => ({ x, y }));
  landmarks[0] = { x, y: y + span }; // WRIST
  return landmarks;
};

const FIST: GestureCategory[] = [{ categoryName: "Closed_Fist", score: 0.9 }];
const OPEN: GestureCategory[] = [{ categoryName: "Open_Palm", score: 0.9 }];

/** Forearm segments crossing in an ✕ (indices 13-16 populated). */
const crossedPose = (): PoseLandmark[] => {
  const pose: PoseLandmark[] = Array.from({ length: 17 }, () => ({ x: 0, y: 0, visibility: 1 }));
  pose[13] = { x: 0.3, y: 0.5, visibility: 1 }; // left elbow
  pose[15] = { x: 0.7, y: 0.5, visibility: 1 }; // left wrist
  pose[14] = { x: 0.5, y: 0.3, visibility: 1 }; // right elbow
  pose[16] = { x: 0.5, y: 0.7, visibility: 1 }; // right wrist
  return pose;
};

const frame = (over: Partial<PipelineFrameInput> = {}): PipelineFrameInput => ({
  allHands: [],
  allGestures: [],
  poseRan: false,
  bodyPose: undefined,
  nowMs: 0,
  wantDebug: false,
  ...over,
});

/** Drive `n` frames of the same input, 33ms apart, returning the last output. */
const run = (
  pipeline: FramePipeline,
  n: number,
  over: Partial<PipelineFrameInput>,
  startMs = 0
) => {
  let out = pipeline.update(frame({ ...over, nowMs: startMs }));
  for (let i = 1; i < n; i++) {
    out = pipeline.update(frame({ ...over, nowMs: startMs + i * 33 }));
  }
  return out;
};

describe("FramePipeline", () => {
  it("emits nothing while no hand has ever been seen", () => {
    const pipeline = new FramePipeline();
    expect(pipeline.update(frame()).state).toBeNull();
  });

  it("reports a present hand with the pen up by default", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 3, { allHands: [makeHand(0.5, 0.6)], allGestures: [OPEN] });
    expect(out.state).toMatchObject({ present: true, pose: "open" });
  });

  it("latches the pen down after sustained fist evidence", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 6, { allHands: [makeHand(0.5, 0.6)], allGestures: [FIST] });
    expect(out.state).toMatchObject({ present: true, pose: "fist" });
  });

  it("rides through a brief dropout without lifting the pen or losing presence", () => {
    const pipeline = new FramePipeline();
    run(pipeline, 6, { allHands: [makeHand(0.5, 0.6)], allGestures: [FIST] });
    const out = run(pipeline, 3, {}, 6 * 33);
    expect(out.state).toMatchObject({ present: true, pose: "fist" });
  });

  it("bleeds the latch and lifts the pen during a long unreadable absence", () => {
    const pipeline = new FramePipeline();
    run(pipeline, 6, { allHands: [makeHand(0.5, 0.6)], allGestures: [FIST] });
    const out = run(pipeline, ABSENCE_GRACE_FRAMES, {}, 6 * 33);
    expect(out.state).toMatchObject({ present: true, pose: "open" });
  });

  it("reports absence after the grace window, then goes quiet", () => {
    const pipeline = new FramePipeline();
    run(pipeline, 3, { allHands: [makeHand(0.5, 0.6)], allGestures: [OPEN] });
    const gone = run(pipeline, ABSENCE_GRACE_FRAMES + 1, {}, 3 * 33);
    expect(gone.state).toMatchObject({ present: false });
    expect(pipeline.update(frame({ nowMs: 9999 })).state).toBeNull();
  });

  it("ignores hands too small to be inside the visitor zone", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 3, {
      allHands: [makeHand(0.5, 0.6, 0.01)],
      allGestures: [OPEN],
    });
    expect(out.state).toBeNull();
  });

  it("keeps the pen hand sticky when a second hand appears", () => {
    const pipeline = new FramePipeline();
    run(pipeline, 3, { allHands: [makeHand(0.7, 0.6)], allGestures: [OPEN] });
    const out = run(
      pipeline,
      2,
      {
        allHands: [makeHand(0.2, 0.2), makeHand(0.7, 0.6)],
        allGestures: [OPEN, OPEN],
        wantDebug: true,
      },
      3 * 33
    );
    expect(out.debug?.primaryIndex).toBe(1);
  });

  it("reports raised when the palm is high in frame", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 2, { allHands: [makeHand(0.5, 0.2)], allGestures: [OPEN] });
    expect(out.state).toMatchObject({ present: true, raised: true });
  });

  it("detects ✕ from two wrists held together", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 2, {
      allHands: [makeHand(0.48, 0.5), makeHand(0.52, 0.5)],
      allGestures: [OPEN, OPEN],
    });
    expect(out.state).toMatchObject({ crossed: true });
  });

  it("detects ✕ from crossed forearms and holds the pose between stride frames", () => {
    const pipeline = new FramePipeline();
    const seen = pipeline.update(
      frame({
        allHands: [makeHand(0.5, 0.6)],
        allGestures: [OPEN],
        poseRan: true,
        bodyPose: crossedPose(),
      })
    );
    expect(seen.state).toMatchObject({ crossed: true });
    // Pose model didn't run this frame — the held reading still applies
    const held = pipeline.update(
      frame({ allHands: [makeHand(0.5, 0.6)], allGestures: [OPEN], nowMs: 33 })
    );
    expect(held.state).toMatchObject({ crossed: true });
    // A fresh pose reading of nobody clears it
    const cleared = pipeline.update(
      frame({ allHands: [makeHand(0.5, 0.6)], allGestures: [OPEN], poseRan: true, nowMs: 66 })
    );
    expect(cleared.state).toMatchObject({ crossed: false });
  });

  it("builds a debug frame only when asked", () => {
    const pipeline = new FramePipeline();
    const silent = run(pipeline, 2, { allHands: [makeHand(0.5, 0.6)], allGestures: [OPEN] });
    expect(silent.debug).toBeNull();
    const chatty = pipeline.update(
      frame({ allHands: [makeHand(0.5, 0.6)], allGestures: [OPEN], wantDebug: true, nowMs: 66 })
    );
    expect(chatty.debug).not.toBeNull();
    expect(chatty.debug!.hands).toHaveLength(1);
    expect(chatty.debug!.state).toMatchObject({ present: true });
  });
});
