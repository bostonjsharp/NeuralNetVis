import { describe, expect, it } from "vitest";
import {
  classifyHand,
  isCloseEnough,
  isRaised,
  mapToPad,
  PALM,
  PoseStabilizer,
  type Landmark,
} from "./handGesture";

/** Synthetic 21-landmark hand: wrist at origin, fingers along -y. */
function hand(tipDistance: number, pipDistance = 0.3): Landmark[] {
  const landmarks: Landmark[] = Array.from({ length: 21 }, () => ({ x: 0, y: 0 }));
  for (const [pip, tip] of [
    [6, 8],
    [10, 12],
    [14, 16],
    [18, 20],
  ]) {
    landmarks[pip] = { x: 0, y: -pipDistance };
    landmarks[tip] = { x: 0, y: -tipDistance };
  }
  return landmarks;
}

describe("classifyHand", () => {
  it("reads extended fingers as open", () => {
    expect(classifyHand(hand(0.6))).toBe("open");
  });

  it("reads folded fingers as a fist", () => {
    expect(classifyHand(hand(0.15))).toBe("fist");
  });

  it("reads index+middle only as the ✌️ clear pose", () => {
    const landmarks = hand(0.6);
    // Fold ring and pinky, leaving index+middle extended
    landmarks[16] = { x: 0, y: -0.1 };
    landmarks[20] = { x: 0, y: -0.1 };
    expect(classifyHand(landmarks)).toBe("two");
  });

  it("reads other two-extended combos as unknown", () => {
    const landmarks = hand(0.6);
    // Fold index and middle, leaving ring+pinky extended
    landmarks[8] = { x: 0, y: -0.1 };
    landmarks[12] = { x: 0, y: -0.1 };
    expect(classifyHand(landmarks)).toBe("unknown");
  });

  it("a half-open hand is unknown — casual poses never register", () => {
    // Tips barely past the middle joints: neither clearly extended nor folded
    expect(classifyHand(hand(0.33, 0.3))).toBe("unknown");
  });
});

describe("distance and raise gates", () => {
  it("ignores hands too small in frame (far beyond the visitor zone)", () => {
    const landmarks = hand(0.6);
    landmarks[PALM] = { x: 0, y: -0.02 }; // tiny wrist→palm span
    expect(isCloseEnough(landmarks)).toBe(false);
    landmarks[PALM] = { x: 0, y: -0.2 };
    expect(isCloseEnough(landmarks)).toBe(true);
  });

  it("only counts a palm high in the frame as raised", () => {
    const landmarks = hand(0.6);
    landmarks[PALM] = { x: 0.5, y: 0.2 }; // camera y grows downward
    expect(isRaised(landmarks)).toBe(true);
    landmarks[PALM] = { x: 0.5, y: 0.6 };
    expect(isRaised(landmarks)).toBe(false);
  });
});

describe("mapToPad", () => {
  it("mirrors horizontally", () => {
    // Hand at camera-left → pad-right after mirroring
    expect(mapToPad(0.22, 0.5).x).toBeCloseTo(1);
    expect(mapToPad(0.78, 0.5).x).toBeCloseTo(0);
  });

  it("maps the central region across the full pad", () => {
    const center = mapToPad(0.5, 0.5);
    expect(center.x).toBeCloseTo(0.5);
    expect(center.y).toBeCloseTo(0.5);
  });

  it("clamps outside the region", () => {
    expect(mapToPad(0.02, 0.98)).toEqual({ x: 1, y: 1 });
    expect(mapToPad(0.98, 0.02)).toEqual({ x: 0, y: 0 });
  });
});

describe("PoseStabilizer", () => {
  const holds = { fist: 3, open: 3, two: 5 };

  it("requires consecutive frames before flipping state", () => {
    const stabilizer = new PoseStabilizer(holds);
    expect(stabilizer.update("fist")).toBe("open");
    expect(stabilizer.update("fist")).toBe("open");
    expect(stabilizer.update("fist")).toBe("fist");
  });

  it("a flicker restarts the streak", () => {
    const stabilizer = new PoseStabilizer(holds);
    stabilizer.update("fist");
    stabilizer.update("fist");
    stabilizer.update("open"); // raw === stable → breaks the fist streak
    stabilizer.update("fist");
    expect(stabilizer.update("fist")).toBe("open"); // streak is only 2 again
    expect(stabilizer.update("fist")).toBe("fist");
  });

  it("unknown frames keep the current state without breaking a lock", () => {
    const stabilizer = new PoseStabilizer(holds);
    for (let i = 0; i < 3; i++) stabilizer.update("fist");
    expect(stabilizer.update("unknown")).toBe("fist");
  });

  it("the ✌️ clear pose needs its longer deliberate hold", () => {
    const stabilizer = new PoseStabilizer(holds);
    for (let i = 0; i < 4; i++) {
      expect(stabilizer.update("two")).toBe("open");
    }
    expect(stabilizer.update("two")).toBe("two");
  });
});
