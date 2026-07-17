import type { BrainVariant } from "../nn/variants";

/**
 * Zone headers matching the scene's left→right composition, plus the
 * weight-color legend. Fixed screen positions — the camera choreography
 * always preserves the input-left / output-right orientation. The middle
 * label tells the truth about whichever brain is on the wall.
 */
export default function Labels({ variant }: { variant: BrainVariant }) {
  const hidden = variant.net.shape.slice(1, -1);
  const middleName =
    hidden.length === 0 ? "STRAIGHT THROUGH" : hidden.length === 1 ? "HIDDEN LAYER" : "HIDDEN LAYERS";
  const middleSub =
    hidden.length === 0
      ? "no hidden layers — pixels vote directly"
      : `${hidden.join(" + ")} neurons finding patterns`;
  return (
    <>
      <div className="zone-label zone-label--input">
        <span className="zone-label__name">INPUT</span>
        <span className="zone-label__sub">your drawing — 784 pixels</span>
      </div>
      <div className="zone-label zone-label--hidden">
        <span className="zone-label__name">{middleName}</span>
        <span className="zone-label__sub">{middleSub}</span>
      </div>
      <div className="zone-label zone-label--output">
        <span className="zone-label__name">OUTPUT</span>
        <span className="zone-label__sub">10 digits — brightest wins</span>
      </div>
      <div className="hud-legend">
        <span className="hud-legend__warm">━ excites</span>
        <span className="hud-legend__cool">━ inhibits</span>
      </div>
    </>
  );
}
