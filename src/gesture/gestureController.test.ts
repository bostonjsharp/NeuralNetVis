import { describe, expect, it } from "vitest";
import {
  BRAIN_HOLD_MS,
  CROSS_HOLD_MS,
  GestureController,
  WAKE_HOLD_MS,
  type GestureCommand,
} from "./gestureController";
import type { GestureState } from "./handTracker";

/** A visible hand in a neutral pose; override what the test cares about. */
function hand(overrides: Partial<GestureState> = {}): GestureState {
  return {
    present: true,
    x: 0.5,
    y: 0.5,
    pose: "open",
    raised: false,
    crossed: false,
    thumbsUp: false,
    ...overrides,
  };
}

const absent = (): GestureState => hand({ present: false });

const types = (commands: GestureCommand[]) => commands.map((c) => c.type);

describe("wake hold (attract mode)", () => {
  it("wakes only after the hand stays raised for the full hold", () => {
    const c = new GestureController();
    expect(types(c.update(hand({ raised: true }), "attract", 0))).not.toContain("wake");
    expect(types(c.update(hand({ raised: true }), "attract", 2500))).not.toContain("wake");
    expect(types(c.update(hand({ raised: true }), "attract", WAKE_HOLD_MS - 1))).not.toContain(
      "wake"
    );
    expect(types(c.update(hand({ raised: true }), "attract", WAKE_HOLD_MS))).toContain("wake");
  });

  it("lowering the hand resets the hold timer", () => {
    const c = new GestureController();
    c.update(hand({ raised: true }), "attract", 0);
    c.update(hand({ raised: true }), "attract", 3000);
    c.update(hand({ raised: false }), "attract", 3500);
    c.update(hand({ raised: true }), "attract", 4000);
    // 5s elapsed since the FIRST raise, but only 2s since the re-raise
    expect(types(c.update(hand({ raised: true }), "attract", 6000))).not.toContain("wake");
    expect(types(c.update(hand({ raised: true }), "attract", 4000 + WAKE_HOLD_MS))).toContain(
      "wake"
    );
  });

  it("hand absence resets the hold timer", () => {
    const c = new GestureController();
    c.update(hand({ raised: true }), "attract", 0);
    c.update(absent(), "attract", 3000);
    c.update(hand({ raised: true }), "attract", 4000);
    expect(types(c.update(hand({ raised: true }), "attract", 8000))).not.toContain("wake");
  });

  it("reports quantized progress and skips no-change frames", () => {
    const c = new GestureController();
    c.update(hand({ raised: true }), "attract", 0);
    const half = c.update(hand({ raised: true }), "attract", WAKE_HOLD_MS / 2);
    expect(half).toContainEqual({ type: "wakeProgress", value: 0.5 });
    // 10ms later the quantized value (1/50 steps) hasn't moved — no re-emit
    const noChange = c.update(hand({ raised: true }), "attract", WAKE_HOLD_MS / 2 + 10);
    expect(types(noChange)).not.toContain("wakeProgress");
  });

  it("resets progress to zero when the hand lowers", () => {
    const c = new GestureController();
    c.update(hand({ raised: true }), "attract", 0);
    c.update(hand({ raised: true }), "attract", 2000);
    expect(c.update(hand({ raised: false }), "attract", 2100)).toContainEqual({
      type: "wakeProgress",
      value: 0,
    });
  });

  it("emits wake exactly once per completed hold", () => {
    const c = new GestureController();
    c.update(hand({ raised: true }), "attract", 0);
    expect(types(c.update(hand({ raised: true }), "attract", WAKE_HOLD_MS))).toContain("wake");
    // Still raised on the next frame (mode hasn't flipped yet) — no double fire
    expect(types(c.update(hand({ raised: true }), "attract", WAKE_HOLD_MS + 33))).not.toContain(
      "wake"
    );
  });
});

describe("pen control (interactive modes)", () => {
  it("a fist puts the pen down, then moves it", () => {
    const c = new GestureController();
    const first = c.update(hand({ pose: "fist", x: 0.2, y: 0.3 }), "draw", 0);
    expect(first).toContainEqual({ type: "penDown", x: 0.2, y: 0.3 });
    const second = c.update(hand({ pose: "fist", x: 0.4, y: 0.5 }), "draw", 33);
    expect(second).toContainEqual({ type: "penMove", x: 0.4, y: 0.5 });
    expect(types(second)).not.toContain("penDown");
  });

  it("an open hand lifts the pen only if it was down", () => {
    const c = new GestureController();
    c.update(hand({ pose: "fist" }), "draw", 0);
    expect(types(c.update(hand({ pose: "open" }), "draw", 33))).toContain("penUp");
    expect(types(c.update(hand({ pose: "open" }), "draw", 66))).not.toContain("penUp");
  });

  it("the cursor tracks the hand and reflects drawing state", () => {
    const c = new GestureController();
    const drawing = c.update(hand({ pose: "fist", x: 0.1, y: 0.9 }), "draw", 0);
    expect(drawing).toContainEqual({
      type: "cursor",
      cursor: { x: 0.1, y: 0.9, drawing: true },
    });
    // Mid-cinematic the pad is locked: cursor shows but never reads as drawing
    const locked = c.update(hand({ pose: "fist", x: 0.1, y: 0.9 }), "infer", 33);
    expect(locked).toContainEqual({
      type: "cursor",
      cursor: { x: 0.1, y: 0.9, drawing: false },
    });
  });

  it("hand absence lifts the pen and hides the cursor", () => {
    const c = new GestureController();
    c.update(hand({ pose: "fist" }), "draw", 0);
    const gone = c.update(absent(), "draw", 33);
    expect(types(gone)).toContain("penUp");
    expect(gone).toContainEqual({ type: "cursor", cursor: null });
  });

  it("a fist in attract mode never draws", () => {
    const c = new GestureController();
    const cmds = c.update(hand({ pose: "fist" }), "attract", 0);
    expect(types(cmds)).not.toContain("penDown");
    expect(types(cmds)).not.toContain("cursor");
  });

  it("returning to attract mid-stroke lifts the pen", () => {
    const c = new GestureController();
    c.update(hand({ pose: "fist" }), "draw", 0);
    // Idle timeout flipped the app back to attract while the fist was held
    expect(types(c.update(hand({ pose: "fist" }), "attract", 33))).toContain("penUp");
  });
});

describe("arms-crossed clear", () => {
  it("clears once after the hold, not every frame it stays crossed", () => {
    const c = new GestureController();
    expect(types(c.update(hand({ crossed: true }), "draw", 0))).not.toContain("clear");
    expect(types(c.update(hand({ crossed: true }), "draw", CROSS_HOLD_MS))).toContain("clear");
    expect(types(c.update(hand({ crossed: true }), "draw", CROSS_HOLD_MS + 500))).not.toContain(
      "clear"
    );
  });

  it("crossing lifts the pen and hides the cursor", () => {
    const c = new GestureController();
    c.update(hand({ pose: "fist" }), "draw", 0);
    const crossed = c.update(hand({ pose: "fist", crossed: true }), "draw", 33);
    expect(types(crossed)).toContain("penUp");
    expect(crossed).toContainEqual({ type: "cursor", cursor: null });
  });

  it("uncrossing re-arms the clear", () => {
    const c = new GestureController();
    c.update(hand({ crossed: true }), "draw", 0);
    c.update(hand({ crossed: true }), "draw", CROSS_HOLD_MS); // fires
    c.update(hand({ crossed: false }), "draw", CROSS_HOLD_MS + 100);
    c.update(hand({ crossed: true }), "draw", CROSS_HOLD_MS + 200);
    expect(
      types(c.update(hand({ crossed: true }), "draw", CROSS_HOLD_MS + 200 + CROSS_HOLD_MS))
    ).toContain("clear");
  });
});

describe("brain-cycle hold (interactive modes)", () => {
  it("cycles the brain after a sustained 👍 in draw mode", () => {
    const c = new GestureController();
    expect(types(c.update(hand({ thumbsUp: true }), "draw", 0))).not.toContain("cycleBrain");
    expect(types(c.update(hand({ thumbsUp: true }), "draw", BRAIN_HOLD_MS - 1))).not.toContain(
      "cycleBrain"
    );
    expect(types(c.update(hand({ thumbsUp: true }), "draw", BRAIN_HOLD_MS))).toContain(
      "cycleBrain"
    );
  });

  it("works at chest height — no raise required", () => {
    const c = new GestureController();
    c.update(hand({ thumbsUp: true, raised: false }), "draw", 0);
    expect(
      types(c.update(hand({ thumbsUp: true, raised: false }), "draw", BRAIN_HOLD_MS))
    ).toContain("cycleBrain");
  });

  it("a raised open hand alone never cycles (the old colliding verb)", () => {
    const c = new GestureController();
    c.update(hand({ raised: true }), "draw", 0);
    expect(
      types(c.update(hand({ raised: true }), "draw", BRAIN_HOLD_MS + 500))
    ).not.toContain("cycleBrain");
  });

  it("shows ring progress while holding and resets when the thumb drops", () => {
    const c = new GestureController();
    c.update(hand({ thumbsUp: true }), "draw", 0);
    const mid = c.update(hand({ thumbsUp: true }), "draw", BRAIN_HOLD_MS / 2);
    const progress = mid.find((cmd) => cmd.type === "wakeProgress");
    expect(progress).toBeDefined();
    expect((progress as { value: number }).value).toBeCloseTo(0.5, 1);
    // Dropping the thumb resets — re-holding must start over
    c.update(hand({ thumbsUp: false }), "draw", BRAIN_HOLD_MS / 2 + 100);
    expect(
      types(c.update(hand({ thumbsUp: true }), "draw", BRAIN_HOLD_MS + 200))
    ).not.toContain("cycleBrain");
  });

  it("crossed arms suppress the cycle hold", () => {
    const c = new GestureController();
    c.update(hand({ thumbsUp: true, crossed: true }), "draw", 0);
    expect(
      types(c.update(hand({ thumbsUp: true, crossed: true }), "draw", BRAIN_HOLD_MS + 500))
    ).not.toContain("cycleBrain");
  });

  it("never cycles during infer or morph", () => {
    for (const mode of ["infer", "morph"] as const) {
      const c = new GestureController();
      c.update(hand({ thumbsUp: true }), mode, 0);
      expect(
        types(c.update(hand({ thumbsUp: true }), mode, BRAIN_HOLD_MS + 500))
      ).not.toContain("cycleBrain");
    }
  });

  it("a continuous hold cycles again after another full hold", () => {
    const c = new GestureController();
    c.update(hand({ thumbsUp: true }), "draw", 0);
    expect(types(c.update(hand({ thumbsUp: true }), "draw", BRAIN_HOLD_MS))).toContain(
      "cycleBrain"
    );
    expect(
      types(c.update(hand({ thumbsUp: true }), "draw", BRAIN_HOLD_MS + 10))
    ).not.toContain("cycleBrain");
    // The second hold restarts on the frame after the fire (t=HOLD+10)
    expect(
      types(c.update(hand({ thumbsUp: true }), "draw", BRAIN_HOLD_MS * 2 + 10))
    ).toContain("cycleBrain");
  });
});
