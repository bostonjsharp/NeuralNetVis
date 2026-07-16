import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ErrorBoundary, { REMOUNT_DELAY_MS } from "./ErrorBoundary";

let broken = false;

function Flaky() {
  if (broken) throw new Error("boom");
  return <div>alive</div>;
}

describe("ErrorBoundary", () => {
  beforeEach(() => {
    broken = false;
    vi.useFakeTimers();
    // React re-reports caught errors on console.error — keep test output clean
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("renders children when nothing is wrong", () => {
    render(
      <ErrorBoundary>
        <Flaky />
      </ErrorBoundary>
    );
    expect(screen.getByText("alive")).toBeTruthy();
  });

  it("swallows a crash into a dark veil instead of a white screen", () => {
    broken = true;
    render(
      <ErrorBoundary>
        <Flaky />
      </ErrorBoundary>
    );
    expect(screen.queryByText("alive")).toBeNull();
    expect(screen.getByTestId("crash-veil")).toBeTruthy();
  });

  it("remounts the tree after the recovery delay", () => {
    broken = true;
    render(
      <ErrorBoundary>
        <Flaky />
      </ErrorBoundary>
    );
    broken = false; // whatever was wrong has passed
    act(() => {
      vi.advanceTimersByTime(REMOUNT_DELAY_MS);
    });
    expect(screen.getByText("alive")).toBeTruthy();
  });

  it("keeps retrying if the crash persists", () => {
    broken = true;
    render(
      <ErrorBoundary>
        <Flaky />
      </ErrorBoundary>
    );
    act(() => {
      vi.advanceTimersByTime(REMOUNT_DELAY_MS); // remount also crashes
    });
    expect(screen.getByTestId("crash-veil")).toBeTruthy();
    broken = false;
    act(() => {
      vi.advanceTimersByTime(REMOUNT_DELAY_MS);
    });
    expect(screen.getByText("alive")).toBeTruthy();
  });
});
