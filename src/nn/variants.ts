import weightsClassic from "../assets/weights-classic.json";
import weightsLinear from "../assets/weights-linear.json";
import weightsTiny from "../assets/weights-tiny.json";
import weightsWide from "../assets/weights-wide.json";
import { loadNet, type Net, type WeightsJson } from "./weights";

/**
 * The registry of brains the exhibit can swap between. All weights are
 * statically imported (~220KB total) — the kiosk loads once from a local
 * server, and a runtime fetch is one more thing that can fail unattended.
 */

export interface BrainVariant {
  id: string;
  /** Short display name on the brain card. */
  label: string;
  /** One-line plain-language description of the architecture. */
  tagline: string;
  net: Net;
  testAccuracy: number;
  paramCount: number;
  /** Stage-0 pruning override — sparse nets can afford richer fans. */
  inputTopK?: number;
}

const paramCount = (net: Net): number =>
  net.layers.reduce((sum, layer) => sum + layer.W.length + layer.b.length, 0);

const make = (
  json: WeightsJson,
  tagline: string,
  inputTopK?: number
): BrainVariant => {
  const net = loadNet(json);
  return {
    id: json.id ?? "classic",
    label: json.label ?? "Classic",
    tagline,
    net,
    testAccuracy: json.testAccuracy,
    paramCount: paramCount(net),
    inputTopK,
  };
};

export const VARIANTS: BrainVariant[] = [
  make(
    weightsLinear as WeightsJson,
    "No hidden layers — every pixel votes straight for a digit.",
    // Only 10 destination neurons at stage 0 — richer fans keep it legible
    64
  ),
  make(weightsTiny as WeightsJson, "One small hidden layer — 8 neurons to find every idea."),
  make(weightsClassic as WeightsJson, "Two hidden layers of 16 — the wall's original brain."),
  make(weightsWide as WeightsJson, "Twice the neurons per layer — sees finer detail."),
];

export const DEFAULT_VARIANT_ID = "classic";

export function getVariant(id: string): BrainVariant {
  const variant = VARIANTS.find((v) => v.id === id);
  if (!variant) throw new Error(`unknown brain variant "${id}"`);
  return variant;
}

/** Cycle order for the gesture / attract paths. */
export function nextVariantId(id: string): string {
  const index = VARIANTS.findIndex((v) => v.id === id);
  return VARIANTS[(index + 1) % VARIANTS.length].id;
}
