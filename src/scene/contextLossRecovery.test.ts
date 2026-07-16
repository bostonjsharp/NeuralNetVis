import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installContextLossRecovery, RESTORE_DEADLINE_MS } from "./contextLossRecovery";

describe("installContextLossRecovery", () => {
  let canvas: HTMLCanvasElement;
  let reload: ReturnType<typeof vi.fn<() => void>>;

  beforeEach(() => {
    vi.useFakeTimers();
    canvas = document.createElement("canvas");
    reload = vi.fn<() => void>();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("reloads when a lost context is never restored", () => {
    installContextLossRecovery(canvas, reload);
    canvas.dispatchEvent(new Event("webglcontextlost"));
    vi.advanceTimersByTime(RESTORE_DEADLINE_MS - 1);
    expect(reload).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it("stands down when the context is restored in time", () => {
    installContextLossRecovery(canvas, reload);
    canvas.dispatchEvent(new Event("webglcontextlost"));
    canvas.dispatchEvent(new Event("webglcontextrestored"));
    vi.advanceTimersByTime(RESTORE_DEADLINE_MS * 2);
    expect(reload).not.toHaveBeenCalled();
  });

  it("re-arms for a second loss after a successful restore", () => {
    installContextLossRecovery(canvas, reload);
    canvas.dispatchEvent(new Event("webglcontextlost"));
    canvas.dispatchEvent(new Event("webglcontextrestored"));
    canvas.dispatchEvent(new Event("webglcontextlost"));
    vi.advanceTimersByTime(RESTORE_DEADLINE_MS);
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it("dispose cancels the watch entirely", () => {
    const dispose = installContextLossRecovery(canvas, reload);
    canvas.dispatchEvent(new Event("webglcontextlost"));
    dispose();
    vi.advanceTimersByTime(RESTORE_DEADLINE_MS * 2);
    expect(reload).not.toHaveBeenCalled();
  });
});
