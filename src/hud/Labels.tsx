/**
 * Zone headers matching the scene's left→right composition, plus the
 * weight-color legend. Fixed screen positions — the camera choreography
 * always preserves the input-left / output-right orientation.
 */
export default function Labels() {
  return (
    <>
      <div className="zone-label zone-label--input">
        <span className="zone-label__name">INPUT</span>
        <span className="zone-label__sub">your drawing — 784 pixels</span>
      </div>
      <div className="zone-label zone-label--hidden">
        <span className="zone-label__name">HIDDEN LAYERS</span>
        <span className="zone-label__sub">16 + 16 neurons finding patterns</span>
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
