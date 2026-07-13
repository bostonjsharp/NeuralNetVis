import { useEffect, useRef } from "react";
import samplesJson from "../assets/samples.json";
import { decodeSamplePixels, type SamplesJson } from "../nn/weights";

const { samples } = samplesJson as SamplesJson;

/** One representative (most confident) sample per digit — the training
 *  script writes each class's samples confidence-first. */
const representatives = Array.from({ length: 10 }, (_, digit) =>
  samples.findIndex((s) => s.label === digit)
);

interface SampleStripProps {
  disabled: boolean;
  onPick: (sampleIndex: number) => void;
}

/** "Try an example" row: real MNIST digits as tappable thumbnails. */
export default function SampleStrip({ disabled, onPick }: SampleStripProps) {
  return (
    <div className="sample-strip">
      {representatives.map((sampleIndex, digit) => (
        <button
          key={digit}
          className="sample-strip__button"
          disabled={disabled}
          onClick={() => onPick(sampleIndex)}
          aria-label={`try a handwritten ${digit}`}
        >
          <Thumbnail pixels={decodeSamplePixels(samples[sampleIndex].pixels)} />
        </button>
      ))}
    </div>
  );
}

function Thumbnail({ pixels }: { pixels: Float32Array }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    const image = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = pixels[i];
      image.data[i * 4] = 30 + 195 * v;
      image.data[i * 4 + 1] = 36 + 214 * v;
      image.data[i * 4 + 2] = 60 + 195 * v;
      image.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }, [pixels]);
  return <canvas ref={canvasRef} width={28} height={28} className="sample-strip__thumb" />;
}
