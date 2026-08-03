import { describe, expect, it } from "vitest";
import {
  ABSENCE_GRACE_FRAMES,
  FramePipeline,
  type PipelineFrameInput,
} from "./framePipeline";
import type { GestureCategory, Landmark } from "./handGesture";
import { POSE, type PoseLandmark } from "./poseGesture";

/** A readable 21-landmark hand: palm at (x, y), wrist `span` below it. */
const makeHand = (x: number, y: number, span = 0.04): Landmark[] => {
  const landmarks = Array.from({ length: 21 }, () => ({ x, y }));
  landmarks[0] = { x, y: y + span }; // WRIST
  return landmarks;
};

const FIST: GestureCategory[] = [{ categoryName: "Closed_Fist", score: 0.9 }];
const OPEN: GestureCategory[] = [{ categoryName: "Open_Palm", score: 0.9 }];
const THUMB: GestureCategory[] = [{ categoryName: "Thumb_Up", score: 0.9 }];

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

  it("latches thumbs-up after sustained evidence and keeps the pen neutral", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 6, { allHands: [makeHand(0.5, 0.6)], allGestures: [THUMB] });
    expect(out.state).toMatchObject({ present: true, thumbsUp: true, pose: "open" });
  });

  it("does not read a fist or open palm as thumbs-up", () => {
    const pipeline = new FramePipeline();
    const fist = run(pipeline, 6, { allHands: [makeHand(0.5, 0.6)], allGestures: [FIST] });
    expect(fist.state).toMatchObject({ thumbsUp: false });
    const open = run(pipeline, 6, { allHands: [makeHand(0.5, 0.6)], allGestures: [OPEN] }, 999);
    expect(open.state).toMatchObject({ thumbsUp: false });
  });

  it("rides a thumbs-up through a brief tracking dropout", () => {
    const pipeline = new FramePipeline();
    run(pipeline, 6, { allHands: [makeHand(0.5, 0.6)], allGestures: [THUMB] });
    // A couple of unreadable frames — the latch must not lose the hold
    const out = run(pipeline, 3, {}, 6 * 33);
    expect(out.state).toMatchObject({ present: true, thumbsUp: true });
  });

  it("releases thumbs-up promptly when the hand changes pose", () => {
    const pipeline = new FramePipeline();
    run(pipeline, 6, { allHands: [makeHand(0.5, 0.6)], allGestures: [THUMB] });
    const out = run(
      pipeline,
      4,
      { allHands: [makeHand(0.5, 0.6)], allGestures: [OPEN] },
      6 * 33
    );
    expect(out.state).toMatchObject({ thumbsUp: false });
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

/** Full standing body for far-tier tests: shoulders 0.2 apart, arms down. */
const standingBody = (): PoseLandmark[] => {
  const pose: PoseLandmark[] = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.5,
    z: 0,
    visibility: 0,
  }));
  const set = (i: number, x: number, y: number, z = 0) => {
    pose[i] = { x, y, z, visibility: 1 };
  };
  set(POSE.leftShoulder, 0.6, 0.35);
  set(POSE.rightShoulder, 0.4, 0.35);
  set(POSE.leftElbow, 0.65, 0.5);
  set(POSE.rightElbow, 0.35, 0.5);
  set(POSE.leftWrist, 0.65, 0.62);
  set(POSE.rightWrist, 0.35, 0.62);
  set(POSE.leftHip, 0.55, 0.65);
  set(POSE.rightHip, 0.45, 0.65);
  return pose;
};

const reachingBody = (): PoseLandmark[] => {
  const pose = standingBody();
  pose[POSE.leftWrist] = { x: 0.6, y: 0.4, z: -0.4, visibility: 1 };
  return pose;
};

const raisedBody = (): PoseLandmark[] => {
  const pose = standingBody();
  pose[POSE.leftWrist] = { x: 0.62, y: 0.2, z: 0, visibility: 1 };
  return pose;
};

const armOutBody = (): PoseLandmark[] => {
  const pose = standingBody();
  pose[POSE.leftWrist] = { x: 0.85, y: 0.37, z: 0, visibility: 1 };
  return pose;
};

const crossedBody = (): PoseLandmark[] => {
  const pose = standingBody();
  pose[POSE.leftElbow] = { x: 0.6, y: 0.5, z: 0, visibility: 1 };
  pose[POSE.leftWrist] = { x: 0.4, y: 0.4, z: 0, visibility: 1 };
  pose[POSE.rightElbow] = { x: 0.4, y: 0.5, z: 0, visibility: 1 };
  pose[POSE.rightWrist] = { x: 0.6, y: 0.4, z: 0, visibility: 1 };
  return pose;
};

describe("FramePipeline far tier", () => {
  it("reports presence with the pen up from body pose alone", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 3, { poseRan: true, bodyPose: standingBody() });
    expect(out.state).toMatchObject({ present: true, pose: "open" });
    expect(out.tier).toBe("far");
  });

  it("latches the pen down after sustained reach", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 8, { poseRan: true, bodyPose: reachingBody() });
    expect(out.state).toMatchObject({ present: true, pose: "fist" });
  });

  it("reports raised for a wrist held above the shoulder", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 3, { poseRan: true, bodyPose: raisedBody() });
    expect(out.state).toMatchObject({ raised: true });
  });

  it("latches the brain-cycle verb for an arm held out to the side", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 8, { poseRan: true, bodyPose: armOutBody() });
    expect(out.state).toMatchObject({ thumbsUp: true });
  });

  it("reports the ✕ from crossed forearms", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 3, { poseRan: true, bodyPose: crossedBody() });
    expect(out.state).toMatchObject({ crossed: true });
  });

  it("moves the cursor with the active wrist (mirrored)", () => {
    const pipeline = new FramePipeline();
    run(pipeline, 5, { poseRan: true, bodyPose: raisedBody() });
    const moved = raisedBody();
    moved[POSE.leftWrist] = { x: 0.72, y: 0.2, z: 0, visibility: 1 };
    const out = run(pipeline, 30, { poseRan: true, bodyPose: moved }, 5 * 33);
    // +0.1 wrist travel in a 0.5-wide pad box → −0.2 in mirrored pad space
    expect(out.state!.x).toBeLessThan(0.45);
  });

  it("a close hand takes the state over from the far tier", () => {
    const pipeline = new FramePipeline();
    run(pipeline, 5, { poseRan: true, bodyPose: reachingBody() });
    const out = run(
      pipeline,
      3,
      {
        allHands: [makeHand(0.5, 0.6)],
        allGestures: [OPEN],
        poseRan: true,
        bodyPose: reachingBody(),
      },
      5 * 33
    );
    expect(out.tier).toBe("close");
    expect(out.state).toMatchObject({ present: true, pose: "open" });
  });

  it("takes over only after the close tier fully releases", () => {
    const pipeline = new FramePipeline();
    run(pipeline, 3, { allHands: [makeHand(0.5, 0.6)], allGestures: [OPEN] });
    // Hand gone but body visible: the close tier's grace window holds.
    const during = run(
      pipeline,
      ABSENCE_GRACE_FRAMES,
      { poseRan: true, bodyPose: standingBody() },
      3 * 33
    );
    expect(during.tier).toBe("close");
    // One more frame releases close (present:false), then far engages.
    const released = pipeline.update(
      frame({ poseRan: true, bodyPose: standingBody(), nowMs: 999 })
    );
    expect(released.state).toMatchObject({ present: false });
    const far = run(pipeline, 3, { poseRan: true, bodyPose: standingBody() }, 1100);
    expect(far.tier).toBe("far");
    expect(far.state).toMatchObject({ present: true });
  });

  it("bleeds out and reports absence when the body disappears", () => {
    const pipeline = new FramePipeline();
    run(pipeline, 8, { poseRan: true, bodyPose: reachingBody() });
    const out = run(
      pipeline,
      ABSENCE_GRACE_FRAMES + 1,
      { poseRan: true, bodyPose: undefined },
      8 * 33
    );
    expect(out.state).toMatchObject({ present: false });
    expect(pipeline.update(frame({ poseRan: true, nowMs: 9999 })).state).toBeNull();
  });

  it("exposes tier, body pose, and reach charge to the debug overlay", () => {
    const pipeline = new FramePipeline();
    const out = run(pipeline, 3, {
      poseRan: true,
      bodyPose: reachingBody(),
      wantDebug: true,
    });
    expect(out.debug?.tier).toBe("far");
    expect(out.debug?.bodyPose).not.toBeNull();
    expect(out.debug?.farReach?.side).toBe("left");
    expect(out.debug?.farReach?.charge).toBeGreaterThan(0);
  });
});
