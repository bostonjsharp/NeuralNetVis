/** The footron wall's exact canvas. Everything is authored at this size. */
export const STAGE_WIDTH = 2736;
export const STAGE_HEIGHT = 1216;

/** Silence before an interactive session snaps back to the attract loop. */
export const IDLE_MS = 60_000;

/** Attract mode fires a sample digit through the network this often. */
export const ATTRACT_FIRE_INTERVAL_MS = 7_000;

/** Pause after the last stroke before a drawing auto-fires. Long enough to
 *  lift the pen (open hand / mouse up) and start another stroke. */
export const AUTO_FIRE_DELAY_MS = 1400;

/** Drawing pad backing resolution (downsampled to 28×28 for the network). */
export const DRAW_PAD_SIZE = 280;
