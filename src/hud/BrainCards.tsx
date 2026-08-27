import { VARIANTS, type BrainVariant } from "../nn/variants";

interface BrainCardsProps {
  activeId: string;
  /** Locked while the cinematic plays or the network is reshaping. */
  disabled: boolean;
  /** Read-only display (attract mode) — no buttons, just the active brain. */
  readOnly?: boolean;
  onSelect: (id: string) => void;
}

/**
 * The brain picker: one card per trained variant, with an honest test
 * accuracy and a mini architecture glyph. Click/tap is the primary input;
 * the phone panel's "switch brain" button cycles them instead.
 */
export default function BrainCards({ activeId, disabled, readOnly, onSelect }: BrainCardsProps) {
  return (
    <div className={`brain-cards${readOnly ? " brain-cards--readonly" : ""}`}>
      {!readOnly && <div className="brain-cards__title">Pick a brain:</div>}
      <div className="brain-cards__row">
        {VARIANTS.map((variant) => (
          <button
            key={variant.id}
            className={`brain-card${variant.id === activeId ? " brain-card--active" : ""}`}
            disabled={disabled || readOnly}
            onClick={() => onSelect(variant.id)}
            aria-label={`switch to the ${variant.label} network`}
            title={variant.tagline}
          >
            <ShapeGlyph variant={variant} />
            <span className="brain-card__name">{variant.label}</span>
            <span className="brain-card__acc">
              {(variant.testAccuracy * 100).toFixed(1)}% accurate
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

/** Mini architecture diagram: one dotted column per layer, height ∝ width. */
function ShapeGlyph({ variant }: { variant: BrainVariant }) {
  const { shape } = variant.net;
  const W = 64;
  const H = 40;
  // Input rendered as a fixed tall column; hidden/output scale by neuron count
  const columns = shape.map((n, i) => ({
    x: 6 + (i * (W - 12)) / Math.max(1, shape.length - 1),
    height: i === 0 ? H - 8 : Math.min(H - 8, 6 + (n / 32) * (H - 14)),
    dots: i === 0 ? 7 : Math.max(2, Math.min(7, Math.round(n / 5))),
  }));
  return (
    <svg
      className="brain-card__glyph"
      viewBox={`0 0 ${W} ${H}`}
      role="presentation"
      aria-hidden="true"
    >
      {columns.map((col, i) => (
        <g key={i}>
          {Array.from({ length: col.dots }, (_, d) => {
            const y =
              H / 2 + (col.dots === 1 ? 0 : (d / (col.dots - 1) - 0.5) * col.height);
            return <circle key={d} cx={col.x} cy={y} r={i === 0 ? 1.4 : 2.1} />;
          })}
        </g>
      ))}
    </svg>
  );
}
