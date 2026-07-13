import { describe, expect, it } from "vitest";
import {
  classifyHand,
  isCloseEnough,
  isRaised,
  PadMapper,
  PALM,
  PoseStabilizer,
  wristsClose,
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

  it("reads a partially folded hand as unknown", () => {
    const landmarks = hand(0.6);
    // Fold index and middle, leaving ring+pinky extended
    landmarks[8] = { x: 0, y: -0.1 };
    landmarks[12] = { x: 0, y: -0.1 };
    expect(classifyHand(landmarks)).toBe("unknown");
  });

  it("a half-open hand is unknown — casual poses never register", () => {
    // Ratio 1.2: between the folded (<1.12) and extended (>1.25) bands
    expect(classifyHand(hand(0.36, 0.3))).toBe("unknown");
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

  it("wristsClose detects crossed forearms", () => {
    expect(wristsClose({ x: 0.45, y: 0.5 }, { x: 0.55, y: 0.5 })).toBe(true);
    expect(wristsClose({ x: 0.2, y: 0.5 }, { x: 0.8, y: 0.5 })).toBe(false);
  });
});

describe("PadMapper", () => {
  // reach 5 × span 0.04 → a 0.2-wide camera box covers the whole pad
  const SPAN = 0.04;

  it("starts the cursor at pad center wherever the hand appears", () => {
    const mapper = new PadMapper(5);
    expect(mapper.update({ x: 0.13, y: 0.81 }, SPAN)).toEqual({ x: 0.5, y: 0.5 });
  });

  it("scales movement by hand size, mirrored horizontally", () => {
    const mapper = new PadMapper(5);
    mapper.update({ x: 0.5, y: 0.5 }, SPAN);
    // Quarter-box left in camera space (user's right) → cursor right
    const moved = mapper.update({ x: 0.45, y: 0.55 }, SPAN);
    expect(moved.x).toBeCloseTo(0.75);
    expect(moved.y).toBeCloseTo(0.75);
  });

  it("the same physical motion covers the pad at any distance", () => {
    // Far hand: half the span → half the camera motion for the same stroke
    const near = new PadMapper(5);
    near.update({ x: 0.5, y: 0.5 }, 0.04);
    const far = new PadMapper(5);
    far.update({ x: 0.5, y: 0.5 }, 0.02);
    expect(near.update({ x: 0.46, y: 0.5 }, 0.04).x).toBeCloseTo(
      far.update({ x: 0.48, y: 0.5 }, 0.02).x
    );
  });

  it("drags the box when pushed past an edge, no dead zone on return", () => {
    const mapper = new PadMapper(5);
    mapper.update({ x: 0.5, y: 0.5 }, SPAN);
    // Push far past the box's left camera edge → pinned at pad right
    expect(mapper.update({ x: 0.2, y: 0.5 }, SPAN).x).toBeCloseTo(1);
    // Any move back immediately walks the cursor off the pin
    expect(mapper.update({ x: 0.22, y: 0.5 }, SPAN).x).toBeCloseTo(0.9);
  });

  it("reset re-centers on the next hand", () => {
    const mapper = new PadMapper(5);
    mapper.update({ x: 0.5, y: 0.5 }, SPAN);
    mapper.update({ x: 0.4, y: 0.4 }, SPAN);
    mapper.reset();
    expect(mapper.update({ x: 0.9, y: 0.9 }, SPAN)).toEqual({ x: 0.5, y: 0.5 });
  });
});

describe("PoseStabilizer", () => {
  const holds = { fist: 3, open: 3 };

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
});
