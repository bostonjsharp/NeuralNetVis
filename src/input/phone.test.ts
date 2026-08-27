import { describe, expect, it } from "vitest";
import { parsePhoneMessage, phoneEnabled } from "./phone";

describe("parsePhoneMessage", () => {
  it("maps valid pen messages to commands", () => {
    expect(parsePhoneMessage({ type: "pen", action: "down", x: 0.25, y: 0.75 })).toEqual({
      type: "pen",
      action: "down",
      x: 0.25,
      y: 0.75,
    });
    expect(parsePhoneMessage({ type: "pen", action: "move", x: 0, y: 1 })).toEqual({
      type: "pen",
      action: "move",
      x: 0,
      y: 1,
    });
    expect(parsePhoneMessage({ type: "pen", action: "up" })).toEqual({
      type: "pen",
      action: "up",
    });
  });

  it("maps clear and brain messages", () => {
    expect(parsePhoneMessage({ type: "clear" })).toEqual({ type: "clear" });
    expect(parsePhoneMessage({ type: "brain" })).toEqual({ type: "brain" });
  });

  it("clamps out-of-range coordinates into the pad", () => {
    expect(parsePhoneMessage({ type: "pen", action: "move", x: -0.5, y: 1.5 })).toEqual({
      type: "pen",
      action: "move",
      x: 0,
      y: 1,
    });
  });

  it("rejects malformed coordinates instead of producing NaN strokes", () => {
    expect(parsePhoneMessage({ type: "pen", action: "down", x: NaN, y: 0.5 })).toBeNull();
    expect(parsePhoneMessage({ type: "pen", action: "down", x: Infinity, y: 0.5 })).toBeNull();
    expect(parsePhoneMessage({ type: "pen", action: "down", x: "0.5", y: 0.5 })).toBeNull();
    expect(parsePhoneMessage({ type: "pen", action: "down", y: 0.5 })).toBeNull();
  });

  it("ignores anything else, never throws", () => {
    expect(parsePhoneMessage(null)).toBeNull();
    expect(parsePhoneMessage(undefined)).toBeNull();
    expect(parsePhoneMessage("pen")).toBeNull();
    expect(parsePhoneMessage(42)).toBeNull();
    expect(parsePhoneMessage({})).toBeNull();
    expect(parsePhoneMessage({ type: "pen" })).toBeNull();
    expect(parsePhoneMessage({ type: "pen", action: "hover", x: 0.5, y: 0.5 })).toBeNull();
    expect(parsePhoneMessage({ type: "zoom", action: "down", x: 0.5, y: 0.5 })).toBeNull();
    // a message that is an array should not slip through the object check
    expect(parsePhoneMessage(["pen", "down"])).toBeNull();
  });
});

describe("phoneEnabled", () => {
  it("only arms on the wall or when forced", () => {
    expect(phoneEnabled("")).toBe(false);
    expect(phoneEnabled("?quality=high")).toBe(false);
    expect(phoneEnabled("?ftMsgUrl=ws%3A%2F%2Flocalhost%3A8089%2Fout")).toBe(true);
    expect(phoneEnabled("?ftmsg=1")).toBe(true);
    expect(phoneEnabled("?ftmsg=0")).toBe(false);
  });
});
