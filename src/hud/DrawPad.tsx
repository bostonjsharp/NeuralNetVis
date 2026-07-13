import { useEffect, useRef } from "react";
import { DRAW_PAD_SIZE } from "../app/constants";

interface DrawPadProps {
  disabled: boolean;
  /** Incrementing this clears the pad. */
  resetKey: number;
  onStrokeStart: () => void;
  /** Fired throttled during a stroke with the pad's grayscale [0,1] pixels. */
  onDraw: (pixels: Float32Array, width: number, height: number) => void;
  onStrokeEnd: (pixels: Float32Array, width: number, height: number) => void;
}

/**
 * Freehand digit pad. The backing canvas is DRAW_PAD_SIZE² and downsampled
 * to 28×28 by the caller; strokes are thick and round-capped so a finger-
 * or mouse-drawn digit lands in MNIST's stroke-width ballpark.
 */
export default function DrawPad({
  disabled,
  resetKey,
  onStrokeStart,
  onDraw,
  onStrokeEnd,
}: DrawPadProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawingRef = useRef(false);
  const lastRef = useRef<{ x: number; y: number } | null>(null);
  const lastEmitRef = useRef(0);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, DRAW_PAD_SIZE, DRAW_PAD_SIZE);
  }, [resetKey]);

  const readPixels = (): Float32Array => {
    const ctx = canvasRef.current?.getContext("2d");
    const out = new Float32Array(DRAW_PAD_SIZE * DRAW_PAD_SIZE);
    if (!ctx) return out;
    const image = ctx.getImageData(0, 0, DRAW_PAD_SIZE, DRAW_PAD_SIZE).data;
    for (let i = 0; i < out.length; i++) out[i] = image[i * 4] / 255;
    return out;
  };

  const toPadCoords = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * DRAW_PAD_SIZE,
      y: ((e.clientY - rect.top) / rect.height) * DRAW_PAD_SIZE,
    };
  };

  const handleDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (disabled) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    drawingRef.current = true;
    lastRef.current = toPadCoords(e);
    onStrokeStart();
  };

  const handleMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current || disabled) return;
    const ctx = canvasRef.current?.getContext("2d");
    const last = lastRef.current;
    if (!ctx || !last) return;
    const point = toPadCoords(e);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    lastRef.current = point;
    const now = performance.now();
    if (now - lastEmitRef.current > 50) {
      lastEmitRef.current = now;
      onDraw(readPixels(), DRAW_PAD_SIZE, DRAW_PAD_SIZE);
    }
  };

  const handleUp = () => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    lastRef.current = null;
    onStrokeEnd(readPixels(), DRAW_PAD_SIZE, DRAW_PAD_SIZE);
  };

  return (
    <canvas
      ref={canvasRef}
      className={`draw-pad${disabled ? " draw-pad--disabled" : ""}`}
      width={DRAW_PAD_SIZE}
      height={DRAW_PAD_SIZE}
      onPointerDown={handleDown}
      onPointerMove={handleMove}
      onPointerUp={handleUp}
      onPointerLeave={handleUp}
      aria-label="drawing pad"
    />
  );
}
