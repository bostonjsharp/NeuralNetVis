/** How long a lost WebGL context may stay lost before we give up on the
 *  driver restoring it and reboot the page instead. */
export const RESTORE_DEADLINE_MS = 10_000;

/**
 * Kiosk insurance for GPU resets. Three.js already prevents default on
 * `webglcontextlost` and re-initializes on `webglcontextrestored` — but a
 * driver that never restores leaves the wall black until someone walks over
 * with a keyboard. This watches for a loss that outlives the deadline and
 * reloads the page: a fresh boot into attract mode, which for an unattended
 * wall beats any amount of in-place cleverness.
 */
export function installContextLossRecovery(
  canvas: HTMLCanvasElement,
  reload: () => void = () => window.location.reload()
): () => void {
  let deadline = 0;
  const onLost = () => {
    window.clearTimeout(deadline);
    deadline = window.setTimeout(reload, RESTORE_DEADLINE_MS);
  };
  const onRestored = () => window.clearTimeout(deadline);
  canvas.addEventListener("webglcontextlost", onLost);
  canvas.addEventListener("webglcontextrestored", onRestored);
  return () => {
    window.clearTimeout(deadline);
    canvas.removeEventListener("webglcontextlost", onLost);
    canvas.removeEventListener("webglcontextrestored", onRestored);
  };
}
