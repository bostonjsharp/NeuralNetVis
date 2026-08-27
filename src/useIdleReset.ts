import { useEffect, useRef } from "react";

/** "phone-activity" is dispatched while phone commands are arriving, so
 *  phone visitors don't get idle-reset mid-drawing. */
const ACTIVITY_EVENTS = ["pointerdown", "pointermove", "keydown", "phone-activity"] as const;

/**
 * Kiosk walk-away detection (ported from LifeOfLLM): while `active`, any
 * pointer/key activity keeps a timer alive; `timeoutMs` of silence fires
 * `onIdle` once. The caller's state change (returning to attract and
 * flipping `active` off) is what stops/starts the watch — the hook never
 * re-arms itself after firing. footron's own experience lifetime remains
 * the outer net.
 */
export function useIdleReset(active: boolean, timeoutMs: number, onIdle: () => void) {
  // Latest-callback ref so a new onIdle identity each render doesn't
  // tear down and re-arm the timer (which would itself reset the clock).
  const onIdleRef = useRef(onIdle);
  onIdleRef.current = onIdle;

  useEffect(() => {
    if (!active) return;
    let timer = window.setTimeout(() => onIdleRef.current(), timeoutMs);
    const reset = () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => onIdleRef.current(), timeoutMs);
    };
    for (const e of ACTIVITY_EVENTS) window.addEventListener(e, reset);
    return () => {
      window.clearTimeout(timer);
      for (const e of ACTIVITY_EVENTS) window.removeEventListener(e, reset);
    };
  }, [active, timeoutMs]);
}
