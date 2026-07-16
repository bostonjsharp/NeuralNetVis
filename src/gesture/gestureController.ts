import type { Mode } from "../app/state";
import type { GestureState } from "./handTracker";

/** Raised-hand dwell before attract mode hands over — the sustained hold,
 *  not the pose, is what filters out passersby at 2m. */
export const WAKE_HOLD_MS = 5000;
/** Arms-crossed dwell before the pad clears. */
export const CROSS_HOLD_MS = 800;

/** Wake progress is quantized to this many steps so ~30fps camera callbacks
 *  produce ~50 HUD renders per fill, not 150. */
const WAKE_PROGRESS_STEPS = 50;

export type GestureCommand =
  | { type: "wake" }
  | { type: "wakeProgress"; value: number }
  | { type: "clear" }
  | { type: "penDown"; x: number; y: number }
  | { type: "penMove"; x: number; y: number }
  | { type: "penUp" }
  | { type: "cursor"; cursor: { x: number; y: number; drawing: boolean } | null };

/**
 * Pure gesture orchestration: folds the tracker's per-frame GestureState and
 * the app mode into commands (wake, clear, pen, cursor). All dwell-timing
 * state lives here; time arrives as a parameter, so every "hold it for N
 * seconds" rule is unit-testable with literal timestamps. App.tsx is left as
 * a dumb interpreter mapping commands onto dispatch/DrawPad.
 *
 * Coordinates pass through normalized (0..1); the interpreter scales to pad
 * pixels.
 */
export class GestureController {
  private wakeStart: number | null = null;
  private lastWakeProgress = 0;
  private crossStart: number | null = null;
  private crossFired = false;
  private drawing = false;
  private cursorShown = false;

  update(g: GestureState, mode: Mode, now: number): GestureCommand[] {
    const commands: GestureCommand[] = [];

    const setProgress = (p: number) => {
      const q = Math.min(1, Math.round(p * WAKE_PROGRESS_STEPS) / WAKE_PROGRESS_STEPS);
      if (q !== this.lastWakeProgress) {
        this.lastWakeProgress = q;
        commands.push({ type: "wakeProgress", value: q });
      }
    };
    const liftPen = () => {
      if (this.drawing) {
        this.drawing = false;
        commands.push({ type: "penUp" });
      }
    };
    const hideCursor = () => {
      if (this.cursorShown) {
        this.cursorShown = false;
        commands.push({ type: "cursor", cursor: null });
      }
    };

    if (!g.present) {
      this.wakeStart = null;
      this.crossStart = null;
      this.crossFired = false;
      setProgress(0);
      liftPen();
      hideCursor();
      return commands;
    }

    if (mode === "attract") {
      // An idle timeout can flip the app back to attract mid-stroke; the pen
      // must not stay latched down or the next wake resumes a phantom stroke.
      liftPen();
      hideCursor();
      this.crossStart = null;
      this.crossFired = false;
      if (g.raised) {
        this.wakeStart ??= now;
        const progress = (now - this.wakeStart) / WAKE_HOLD_MS;
        setProgress(progress);
        if (progress >= 1) {
          this.wakeStart = null;
          setProgress(0);
          commands.push({ type: "wake" });
        }
      } else {
        this.wakeStart = null;
        setProgress(0);
      }
      return commands;
    }

    this.wakeStart = null;
    setProgress(0);

    // Latched so one held ✕ clears once, not every frame it stays crossed.
    if (g.crossed) {
      liftPen();
      this.crossStart ??= now;
      if (!this.crossFired && now - this.crossStart >= CROSS_HOLD_MS) {
        this.crossFired = true;
        commands.push({ type: "clear" });
      }
      hideCursor();
      return commands;
    }
    this.crossStart = null;
    this.crossFired = false;

    if (g.pose === "fist") {
      if (this.drawing) {
        commands.push({ type: "penMove", x: g.x, y: g.y });
      } else {
        this.drawing = true;
        commands.push({ type: "penDown", x: g.x, y: g.y });
      }
    } else {
      liftPen();
    }
    this.cursorShown = true;
    commands.push({
      type: "cursor",
      // Mid-cinematic ("infer") the pad is locked; the cursor stays visible
      // but must not read as an active pen.
      cursor: { x: g.x, y: g.y, drawing: g.pose === "fist" && mode !== "infer" },
    });
    return commands;
  }
}
