import { describe, expect, it } from "vitest";
import {
  armEngaged,
  armOut,
  armResting,
  forearmsCrossed,
  pickActiveArm,
  POSE,
  poseRaised,
  reachEvidence,
  segmentsIntersect,
  shoulderWidth,
  type PoseLandmark,
} from "./poseGesture";

/** Sparse pose array with only the named indices populated. */
const poseWith = (points: Record<number, PoseLandmark>): PoseLandmark[] => {
  const pose: PoseLandmark[] = Array.from({ length: 33 }, () => ({
    x: 0,
    y: 0,
    visibility: 0,
  }));
  for (const [i, p] of Object.entries(points)) pose[Number(i)] = p;
  return pose;
};

describe("segmentsIntersect", () => {
  it("finds a true crossing", () => {
    expect(
      segmentsIntersect({ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 }, { x: 1, y: 0 })
    ).toBe(true);
    expect(
      segmentsIntersect({ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 1 })
    ).toBe(false);
  });
});

describe("forearmsCrossed", () => {
  it("detects forearm segments crossing in an ✕", () => {
    const pose = poseWith({
      [POSE.leftElbow]: { x: 0.6, y: 0.6, visibility: 1 },
      [POSE.leftWrist]: { x: 0.4, y: 0.4, visibility: 1 },
      [POSE.rightElbow]: { x: 0.4, y: 0.6, visibility: 1 },
      [POSE.rightWrist]: { x: 0.6, y: 0.4, visibility: 1 },
    });
    expect(forearmsCrossed(pose)).toBe(true);
  });

  it("stays false for parallel forearms", () => {
    const pose = poseWith({
      [POSE.leftElbow]: { x: 0.6, y: 0.6, visibility: 1 },
      [POSE.leftWrist]: { x: 0.6, y: 0.4, visibility: 1 },
      [POSE.rightElbow]: { x: 0.4, y: 0.6, visibility: 1 },
      [POSE.rightWrist]: { x: 0.4, y: 0.4, visibility: 1 },
    });
    expect(forearmsCrossed(pose)).toBe(false);
  });

  it("refuses to guess from a low-visibility landmark", () => {
    const pose = poseWith({
      [POSE.leftElbow]: { x: 0.6, y: 0.6, visibility: 0.2 },
      [POSE.leftWrist]: { x: 0.4, y: 0.4, visibility: 1 },
      [POSE.rightElbow]: { x: 0.4, y: 0.6, visibility: 1 },
      [POSE.rightWrist]: { x: 0.6, y: 0.4, visibility: 1 },
    });
    expect(forearmsCrossed(pose)).toBe(false);
  });
});

/** A visitor standing square to the camera: shoulders 0.2 apart at y 0.35,
 *  arms hanging (wrists just above the hip line at y 0.65). */
const standing = (): PoseLandmark[] => {
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

describe("shoulderWidth", () => {
  it("measures shoulder-to-shoulder distance", () => {
    expect(shoulderWidth(standing())).toBeCloseTo(0.2);
  });

  it("returns 0 when a shoulder is unreadable", () => {
    const pose = standing();
    pose[POSE.leftShoulder] = { x: 0.6, y: 0.35, z: 0, visibility: 0.1 };
    expect(shoulderWidth(pose)).toBe(0);
  });
});

describe("poseRaised", () => {
  it("is true when the wrist is above the shoulder by the margin", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.62, y: 0.2, z: 0, visibility: 1 };
    expect(poseRaised(pose, "left")).toBe(true);
  });

  it("is false for a wrist at shoulder height (a forward reach, not a wake)", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.62, y: 0.34, z: 0, visibility: 1 };
    expect(poseRaised(pose, "left")).toBe(false);
  });
});

describe("armOut", () => {
  it("detects an arm held straight out to the side", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.85, y: 0.37, z: 0, visibility: 1 };
    expect(armOut(pose, "left")).toBe(true);
  });

  it("rejects a raised arm (vertical, not lateral)", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.62, y: 0.2, z: 0, visibility: 1 };
    expect(armOut(pose, "left")).toBe(false);
  });

  it("rejects a forward reach (wrist stays near the shoulder in 2D)", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.63, y: 0.38, z: -0.4, visibility: 1 };
    expect(armOut(pose, "left")).toBe(false);
  });

  it("rejects a diagonal raise (lateral passes, the vertical bound fails)", () => {
    const pose = standing();
    // lateral 1.25 shoulder-widths (> 0.8) but vertical 1.0 (≥ 0.5)
    pose[POSE.leftWrist] = { x: 0.85, y: 0.15, z: 0, visibility: 1 };
    expect(armOut(pose, "left")).toBe(false);
  });
});

describe("reachEvidence", () => {
  it("drives pen-up when the arm hangs at rest", () => {
    // Wrist z at shoulder depth → reach 0, well below REACH_REST.
    expect(reachEvidence(standing(), "left")).toMatchObject({ fist: 0, open: 1 });
  });

  it("ramps the pen-down drive between rest and full reach", () => {
    const pose = standing();
    // reach = (0 − (−0.2)) / 0.2 = 1.0 → halfway between REACH_REST and REACH_FULL
    pose[POSE.leftWrist] = { x: 0.6, y: 0.4, z: -0.2, visibility: 1 };
    const e = reachEvidence(pose, "left");
    expect(e.fist).toBeCloseTo(0.5);
    expect(e.open).toBe(0);
  });

  it("saturates at full reach", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.6, y: 0.4, z: -0.4, visibility: 1 };
    expect(reachEvidence(pose, "left")).toMatchObject({ fist: 1, open: 0 });
  });

  it("treats a wrist below the hip as firmly pen-up regardless of z", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.6, y: 0.75, z: -0.4, visibility: 1 };
    expect(reachEvidence(pose, "left")).toMatchObject({ fist: 0, open: 1 });
  });

  it("yields no evidence when z is missing or the wrist is unreadable", () => {
    const noZ = standing();
    noZ[POSE.leftWrist] = { x: 0.6, y: 0.4, visibility: 1 };
    expect(reachEvidence(noZ, "left")).toMatchObject({ fist: 0, open: 0 });
    const dim = standing();
    dim[POSE.leftWrist] = { x: 0.6, y: 0.4, z: -0.4, visibility: 0.2 };
    expect(reachEvidence(dim, "left")).toMatchObject({ fist: 0, open: 0 });
  });
});

describe("armEngaged", () => {
  it("is true for each far-tier verb — raise, arm-out, reach", () => {
    const raised = standing();
    raised[POSE.leftWrist] = { x: 0.62, y: 0.2, z: 0, visibility: 1 };
    expect(armEngaged(raised, "left")).toBe(true);
    const out = standing();
    out[POSE.leftWrist] = { x: 0.85, y: 0.37, z: 0, visibility: 1 };
    expect(armEngaged(out, "left")).toBe(true);
    const reaching = standing();
    reaching[POSE.leftWrist] = { x: 0.6, y: 0.4, z: -0.2, visibility: 1 };
    expect(armEngaged(reaching, "left")).toBe(true);
  });

  it("is false for an arm just standing there", () => {
    expect(armEngaged(standing(), "left")).toBe(false);
    expect(armEngaged(standing(), "right")).toBe(false);
  });
});

describe("armResting", () => {
  it("is true only when the wrist hangs past the hip", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.65, y: 0.7, z: 0, visibility: 1 };
    expect(armResting(pose, "left")).toBe(true);
    // The standing wrist sits just above the hip line — not yet resting.
    expect(armResting(standing(), "left")).toBe(false);
  });

  it("refuses to guess when the hip is unreadable", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.65, y: 0.7, z: 0, visibility: 1 };
    pose[POSE.leftHip] = { x: 0.55, y: 0.65, z: 0, visibility: 0.1 };
    expect(armResting(pose, "left")).toBe(false);
  });
});

describe("pickActiveArm", () => {
  it("prefers the higher wrist on first acquisition", () => {
    const pose = standing();
    pose[POSE.rightWrist] = { x: 0.35, y: 0.3, z: 0, visibility: 1 };
    expect(pickActiveArm(pose, null)).toBe("right");
  });

  it("prefers an engaged wrist over a higher idle one on acquisition", () => {
    const pose = standing();
    // Left reaches (engaged) while the idle right wrist happens to sit higher.
    pose[POSE.leftWrist] = { x: 0.6, y: 0.4, z: -0.2, visibility: 1 };
    pose[POSE.rightWrist] = { x: 0.35, y: 0.33, z: 0, visibility: 1 };
    expect(pickActiveArm(pose, null)).toBe("left");
  });

  it("breaks a both-engaged acquisition by wrist height", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.62, y: 0.2, z: 0, visibility: 1 };
    pose[POSE.rightWrist] = { x: 0.38, y: 0.25, z: 0, visibility: 1 };
    expect(pickActiveArm(pose, null)).toBe("left");
  });

  it("stays sticky to the current arm while its wrist reads", () => {
    const pose = standing();
    pose[POSE.rightWrist] = { x: 0.35, y: 0.3, z: 0, visibility: 1 };
    expect(pickActiveArm(pose, "left")).toBe("left");
  });

  it("lets an engaged arm take over once the current arm rests", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.65, y: 0.7, z: 0, visibility: 1 }; // past the hip
    pose[POSE.rightWrist] = { x: 0.35, y: 0.2, z: 0, visibility: 1 }; // raised
    expect(pickActiveArm(pose, "left")).toBe("right");
  });

  it("refuses takeover while the current arm is still speaking", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.6, y: 0.4, z: -0.4, visibility: 1 }; // mid-reach
    pose[POSE.rightWrist] = { x: 0.35, y: 0.2, z: 0, visibility: 1 }; // raised
    expect(pickActiveArm(pose, "left")).toBe("left");
  });

  it("refuses takeover by an idle arm even when the current arm rests", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.65, y: 0.7, z: 0, visibility: 1 };
    expect(pickActiveArm(pose, "left")).toBe("left");
  });

  it("hands an invisible wrist over only to an engaged arm, else null", () => {
    const pose = standing();
    pose[POSE.leftWrist] = { x: 0.65, y: 0.62, z: 0, visibility: 0.1 };
    // Idle other arm inherits nothing — the far grace window re-picks fresh.
    expect(pickActiveArm(pose, "left")).toBeNull();
    pose[POSE.rightWrist] = { x: 0.35, y: 0.2, z: 0, visibility: 1 }; // raised
    expect(pickActiveArm(pose, "left")).toBe("right");
    pose[POSE.rightWrist] = { x: 0.35, y: 0.62, z: 0, visibility: 0.1 };
    expect(pickActiveArm(pose, "left")).toBeNull();
  });
});
