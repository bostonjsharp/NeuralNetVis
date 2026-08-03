import { describe, expect, it } from "vitest";
import {
  isCloseEnough,
  isRaised,
  NO_EVIDENCE,
  OneEuroFilter,
  PadMapper,
  PALM,
  penEvidence,
  PenLatch,
  pickPrimaryHand,
  thumbEvidence,
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

describe("penEvidence", () => {
  it("reads a recognized fist as fist evidence, at the model's own confidence", () => {
    expect(penEvidence([{ categoryName: "Closed_Fist", score: 0.91 }])).toEqual({
      fist: 0.91,
      open: 0,
    });
  });

  it("reads any splayed-finger gesture as open evidence", () => {
    for (const name of ["Open_Palm", "Victory", "Pointing_Up"]) {
      expect(penEvidence([{ categoryName: name, score: 0.8 }]).open).toBe(0.8);
    }
  });

  it("an unrecognized hand is NOT evidence of a fist — it is no evidence at all", () => {
    // The old classifier collapsed every uncertain frame into "unknown",
    // which froze the pen. Uncertainty must stay neutral so the latch can
    // decay it back to pen-up.
    expect(penEvidence([{ categoryName: "None", score: 0.6 }])).toEqual(NO_EVIDENCE);
    expect(penEvidence([])).toEqual(NO_EVIDENCE);
  });

  it("curled-but-not-fist gestures stay neutral rather than faking a release", () => {
    // Thumb_Up is a folded hand; calling it "open" would drop strokes.
    expect(penEvidence([{ categoryName: "Thumb_Up", score: 0.7 }])).toEqual(NO_EVIDENCE);
  });
});

describe("thumbEvidence", () => {
  it("reads a recognized 👍 as for-evidence at the model's confidence", () => {
    expect(thumbEvidence([{ categoryName: "Thumb_Up", score: 0.85 }])).toEqual({
      fist: 0.85,
      open: 0,
    });
  });

  it("reads any other named gesture as against-evidence", () => {
    for (const name of ["Closed_Fist", "Open_Palm", "Victory", "Pointing_Up", "Thumb_Down"]) {
      expect(thumbEvidence([{ categoryName: name, score: 0.8 }])).toEqual({
        fist: 0,
        open: 0.8,
      });
    }
  });

  it("an unreadable hand is no evidence either way — the latch decays it", () => {
    expect(thumbEvidence([{ categoryName: "None", score: 0.6 }])).toEqual(NO_EVIDENCE);
    expect(thumbEvidence([])).toEqual(NO_EVIDENCE);
  });
});

describe("PenLatch", () => {
  const fist = (score = 0.9) => ({ fist: score, open: 0 });
  const open = (score = 0.9) => ({ fist: 0, open: score });

  /** Drive the latch n frames and return the final pose. */
  const run = (latch: PenLatch, evidence: { fist: number; open: number }, frames: number) => {
    let pose = latch.update(evidence);
    for (let i = 1; i < frames; i++) pose = latch.update(evidence);
    return pose;
  };

  it("starts pen-up and presses after a few confident fist frames", () => {
    const latch = new PenLatch();
    expect(latch.update(fist())).toBe("open"); // not instant
    expect(run(latch, fist(), 6)).toBe("fist");
  });

  it("releases FASTER than it presses — an open hand must always win", () => {
    const latch = new PenLatch();
    let pressFrames = 0;
    while (latch.update(fist()) !== "fist") pressFrames++;
    let releaseFrames = 0;
    while (latch.update(open()) !== "open") releaseFrames++;
    expect(releaseFrames).toBeLessThan(pressFrames);
  });

  // THE REPORTED BUG. Sleeves make the recognizer drop out intermittently,
  // so an opening hand yields open/None/open/None… The old PoseStabilizer
  // reset its streak on every uncertain frame, so the release streak never
  // completed and the pen drew forever.
  it("releases even when uncertain frames interleave with the open hand", () => {
    const latch = new PenLatch();
    run(latch, fist(), 10);
    expect(latch.pose()).toBe("fist");
    let pose = latch.pose();
    for (let i = 0; i < 20; i++) {
      pose = latch.update(i % 2 === 0 ? open(0.8) : NO_EVIDENCE);
    }
    expect(pose).toBe("open");
  });

  it("a hand the model cannot read at all decays the pen up — uncertainty fails safe", () => {
    const latch = new PenLatch();
    run(latch, fist(), 10);
    expect(run(latch, NO_EVIDENCE, 30)).toBe("open");
  });

  it("but a couple of unreadable frames mid-stroke do not chop the stroke", () => {
    const latch = new PenLatch();
    run(latch, fist(), 10);
    expect(run(latch, NO_EVIDENCE, 3)).toBe("fist");
    expect(latch.update(fist())).toBe("fist");
  });

  it("survives a stroke where only one frame in four reads as a fist", () => {
    const latch = new PenLatch();
    run(latch, fist(), 10);
    let pose = latch.pose();
    for (let i = 0; i < 40; i++) {
      pose = latch.update(i % 4 === 0 ? fist() : NO_EVIDENCE);
    }
    expect(pose).toBe("fist");
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

describe("pickPrimaryHand", () => {
  it("picks the lifted (highest) hand on first acquisition", () => {
    const palms = [
      { x: 0.3, y: 0.7 },
      { x: 0.7, y: 0.3 },
    ];
    expect(pickPrimaryHand(palms, null)).toBe(1);
  });

  it("sticks to the hand nearest the pen's last position", () => {
    const last = { x: 0.3, y: 0.7 };
    const palms = [
      { x: 0.7, y: 0.3 }, // a second hand higher in frame
      { x: 0.32, y: 0.69 }, // the pen hand, barely moved
    ];
    expect(pickPrimaryHand(palms, last)).toBe(1);
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

describe("OneEuroFilter", () => {
  const DT = 1 / 30;

  /** The filter this replaces: fixed exponential smoothing, α = 0.35. */
  const fixedAlpha = (alpha: number) => {
    let y: number | null = null;
    return (x: number) => (y = y === null ? x : y + (x - y) * alpha);
  };

  it("lags far less than fixed smoothing on a fast stroke", () => {
    // A hand crossing the whole pad in ~0.3s — a quick digit stroke.
    const speed = 3.0; // pad-widths per second
    const euro = new OneEuroFilter();
    const fixed = fixedAlpha(0.35);
    let euroErr = 0;
    let fixedErr = 0;
    for (let i = 1; i <= 15; i++) {
      const truth = i * speed * DT;
      euroErr = Math.abs(truth - euro.filter(truth, DT));
      fixedErr = Math.abs(truth - fixed(truth));
    }
    // Steady-state tracking error while moving fast
    expect(euroErr).toBeLessThan(fixedErr * 0.5);
  });

  it("still smooths harder than fixed smoothing when the hand is nearly still", () => {
    // Jitter around a stationary point — no speed, so heavy smoothing.
    const jitter = [0.5, 0.512, 0.49, 0.508, 0.494, 0.506, 0.492, 0.504];
    const euro = new OneEuroFilter();
    const fixed = fixedAlpha(0.35);
    let euroSwing = 0;
    let fixedSwing = 0;
    let prevEuro: number | null = null;
    let prevFixed: number | null = null;
    for (const value of jitter) {
      const e = euro.filter(value, DT);
      const f = fixed(value);
      if (prevEuro !== null) euroSwing += Math.abs(e - prevEuro);
      if (prevFixed !== null) fixedSwing += Math.abs(f - prevFixed);
      prevEuro = e;
      prevFixed = f;
    }
    expect(euroSwing).toBeLessThan(fixedSwing);
  });

  it("reset re-seeds on the next sample instead of gliding in", () => {
    const euro = new OneEuroFilter();
    for (let i = 0; i < 10; i++) euro.filter(0.2, DT);
    euro.reset();
    expect(euro.filter(0.9, DT)).toBe(0.9);
  });

  it("survives a zero or negative timestep without exploding", () => {
    const euro = new OneEuroFilter();
    euro.filter(0.5, DT);
    expect(Number.isFinite(euro.filter(0.6, 0))).toBe(true);
  });
});

describe("PenLatch hand loss", () => {
  it("a hand that leaves the frame lifts the pen — it never keeps drawing", () => {
    // The pose-wrist fallback used to hold the last pose for ~3s while
    // steering the cursor from the body model, so a dropped hand kept
    // painting. Absence is now just an absence of evidence.
    const latch = new PenLatch();
    for (let i = 0; i < 10; i++) latch.update({ fist: 0.9, open: 0 });
    expect(latch.pose()).toBe("fist");
    let pose = latch.pose();
    for (let i = 0; i < 30; i++) pose = latch.update(NO_EVIDENCE);
    expect(pose).toBe("open");
  });

  it("reset drops straight to pen-up", () => {
    const latch = new PenLatch();
    for (let i = 0; i < 10; i++) latch.update({ fist: 0.9, open: 0 });
    latch.reset();
    expect(latch.pose()).toBe("open");
  });
});
