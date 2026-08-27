import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
} from "react";
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
 * Imperative pen interface so the phone controller drives the exact same
 * drawing path as the mouse. Coordinates are in pad space (0..DRAW_PAD_SIZE).
 */
export interface DrawPadHandle {
  penDown(x: number, y: number): void;
  penMove(x: number, y: number): void;
  penUp(): void;
}

/**
 * Freehand digit pad. The backing canvas is DRAW_PAD_SIZE² and downsampled
 * to 28×28 by the caller; strokes are thick and round-capped so a finger-
 * or mouse-drawn digit lands in MNIST's stroke-width ballpark.
 */
const DrawPad = forwardRef<DrawPadHandle, DrawPadProps>(function DrawPad(
  { disabled, resetKey, onStrokeStart, onDraw, onStrokeEnd },
  ref
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawingRef = useRef(false);
  const lastRef = useRef<{ x: number; y: number } | null>(null);
  const lastEmitRef = useRef(0);
  const disabledRef = useRef(disabled);
  disabledRef.current = disabled;

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

  const penDown = (x: number, y: number) => {
    if (disabledRef.current) return;
    drawingRef.current = true;
    lastRef.current = { x, y };
    onStrokeStart();
  };

  const penMove = (x: number, y: number) => {
    if (!drawingRef.current || disabledRef.current) return;
    const ctx = canvasRef.current?.getContext("2d");
    const last = lastRef.current;
    if (!ctx || !last) return;
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(x, y);
    ctx.stroke();
    lastRef.current = { x, y };
    const now = performance.now();
    if (now - lastEmitRef.current > 50) {
      lastEmitRef.current = now;
      onDraw(readPixels(), DRAW_PAD_SIZE, DRAW_PAD_SIZE);
    }
  };

  const penUp = () => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    lastRef.current = null;
    onStrokeEnd(readPixels(), DRAW_PAD_SIZE, DRAW_PAD_SIZE);
  };

  useImperativeHandle(ref, () => ({
    penDown,
    penMove,
    penUp,
  }));

  const toPadCoords = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * DRAW_PAD_SIZE,
      y: ((e.clientY - rect.top) / rect.height) * DRAW_PAD_SIZE,
    };
  };

  return (
    <div className="draw-pad-wrap">
      <canvas
        ref={canvasRef}
        className={`draw-pad${disabled ? " draw-pad--disabled" : ""}`}
        width={DRAW_PAD_SIZE}
        height={DRAW_PAD_SIZE}
        onPointerDown={(e) => {
          if (disabled) return;
          e.currentTarget.setPointerCapture(e.pointerId);
          const p = toPadCoords(e);
          penDown(p.x, p.y);
        }}
        onPointerMove={(e) => {
          const p = toPadCoords(e);
          penMove(p.x, p.y);
        }}
        onPointerUp={penUp}
        onPointerLeave={penUp}
        aria-label="drawing pad"
      />
    </div>
  );
});

export default DrawPad;
