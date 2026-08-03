import { describe, expect, it } from "vitest";
import { forearmsCrossed, POSE, segmentsIntersect, type PoseLandmark } from "./poseGesture";

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
