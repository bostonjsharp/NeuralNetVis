import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

// The scene bridge is the only WebGL touchpoint — stub it, don't mock WebGL.
const sceneApi = {
  fire: vi.fn(),
  setMode: vi.fn(),
  setInputPixels: vi.fn(),
  dispose: vi.fn(),
};
vi.mock("./scene/SceneManager", () => ({
  createScene: vi.fn(() => sceneApi),
}));

import App from "./App";
import { stageScale } from "./hud/Stage";

afterEach(cleanup);

describe("App", () => {
  it("boots into attract mode with the title and education visible", () => {
    render(<App />);
    expect(screen.getByText("Inside a Neural Network")).toBeTruthy();
    expect(screen.getByText(/move the mouse/i)).toBeTruthy();
    expect(screen.getByText(/excites/)).toBeTruthy();
    expect(screen.getByText("HIDDEN LAYERS")).toBeTruthy();
  });

  it("creates the scene against the wall-sized canvas", () => {
    const { container } = render(<App />);
    const canvas = container.querySelector("canvas.scene-canvas") as HTMLCanvasElement;
    expect(canvas.width).toBe(2736);
    expect(canvas.height).toBe(1216);
  });
});

describe("stageScale", () => {
  it("letterboxes to the limiting axis", () => {
    expect(stageScale(2736, 1216)).toBe(1);
    expect(stageScale(1368, 1216)).toBeCloseTo(0.5);
    expect(stageScale(2736, 304)).toBeCloseTo(0.25);
  });

  it("falls back to 1 for degenerate viewports", () => {
    expect(stageScale(0, 0)).toBe(1);
  });
});
