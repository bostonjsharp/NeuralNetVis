import { useEffect, useRef } from "react";
import { RAISE_LINE, WRIST, type HandPose } from "../gesture/handGesture";
import type { HandDebugFrame } from "../gesture/handTracker";

/** MediaPipe hand skeleton edges, for drawing. */
const BONES: readonly [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const POSE_COLOR: Record<HandPose, string> = {
  fist: "#ffa03d",
  open: "#4dbfff",
  unknown: "#8d97b4",
};

export interface DebugData {
  video: HTMLVideoElement | null;
  frame: HandDebugFrame | null;
}

const VIEW_W = 560;
const VIEW_H = 315;

/**
 * Dev-only overlay (G key / ?debug): the mirrored camera feed with hand
 * skeletons, the raise line, and a live readout of what the tracker
 * classified. Draws imperatively at camera rate — no React re-renders.
 */
export default function DebugPanel({ dataRef }: { dataRef: React.RefObject<DebugData> }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const statusRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const { video, frame } = dataRef.current;
      const canvas = canvasRef.current;
      const status = statusRef.current;
      if (!canvas || !status) return;
      if (!video) {
        status.textContent = "waiting for camera…";
        return;
      }
      if (video.parentElement !== containerRef.current) {
        video.className = "debug-panel__video";
        containerRef.current?.prepend(video);
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, VIEW_W, VIEW_H);
      // Everything mirrors to match the selfie-view video underneath
      const mx = (x: number) => (1 - x) * VIEW_W;

      // Raise line
      ctx.strokeStyle = "rgba(77, 191, 255, 0.6)";
      ctx.setLineDash([8, 6]);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, RAISE_LINE * VIEW_H);
      ctx.lineTo(VIEW_W, RAISE_LINE * VIEW_H);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(77, 191, 255, 0.8)";
      ctx.font = "13px Outfit, sans-serif";
      ctx.fillText("raise line — hold a hand above this", 10, RAISE_LINE * VIEW_H - 6);

      if (!frame) return;

      // Forearms from the body-pose model — the long-range ✕ signal
      if (frame.forearms) {
        const f = frame.forearms;
        ctx.strokeStyle = f.crossed ? "#7dff9a" : "rgba(255, 214, 90, 0.85)";
        ctx.lineWidth = 4;
        for (const [elbow, wristPoint] of [
          [f.leftElbow, f.leftWrist],
          [f.rightElbow, f.rightWrist],
        ] as const) {
          ctx.beginPath();
          ctx.moveTo(mx(elbow.x), elbow.y * VIEW_H);
          ctx.lineTo(mx(wristPoint.x), wristPoint.y * VIEW_H);
          ctx.stroke();
        }
      }

      for (let i = 0; i < frame.hands.length; i++) {
        const hand = frame.hands[i];
        const color = hand.closeEnough ? POSE_COLOR[hand.rawPose] : "#e5484d";
        ctx.strokeStyle = color;
        ctx.lineWidth = i === frame.primaryIndex ? 3.5 : 2;
        for (const [a, b] of BONES) {
          ctx.beginPath();
          ctx.moveTo(mx(hand.landmarks[a].x), hand.landmarks[a].y * VIEW_H);
          ctx.lineTo(mx(hand.landmarks[b].x), hand.landmarks[b].y * VIEW_H);
          ctx.stroke();
        }
        ctx.fillStyle = color;
        for (const lm of hand.landmarks) {
          ctx.beginPath();
          ctx.arc(mx(lm.x), lm.y * VIEW_H, 3, 0, Math.PI * 2);
          ctx.fill();
        }
        const wrist = hand.landmarks[WRIST];
        const penTag = i === frame.primaryIndex ? "PEN · " : "";
        ctx.fillText(
          `${penTag}${hand.closeEnough ? hand.rawPose : "too far"} · span ${hand.span.toFixed(3)}`,
          mx(wrist.x) + 8,
          wrist.y * VIEW_H + 16
        );
      }
      // Wrist-to-wrist line for the ✕ gesture
      if (frame.hands.length >= 2) {
        const [a, b] = [frame.hands[0].landmarks[WRIST], frame.hands[1].landmarks[WRIST]];
        ctx.strokeStyle = frame.crossed ? "#7dff9a" : "rgba(141, 151, 180, 0.6)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(mx(a.x), a.y * VIEW_H);
        ctx.lineTo(mx(b.x), b.y * VIEW_H);
        ctx.stroke();
      }

      const s = frame.state;
      status.textContent = [
        `hands: ${frame.hands.length}`,
        `✕: ${frame.crossed}${frame.forearms ? ` (arms ${frame.forearms.crossed ? "crossed" : "apart"})` : " (no body pose)"}`,
        s
          ? `emitted: ${s.present ? `${s.pose}${s.raised ? " · raised" : ""} @ ${s.x.toFixed(2)},${s.y.toFixed(2)}` : "absent"}`
          : "emitted: —",
      ].join("   |   ");
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [dataRef]);

  return (
    <div className="debug-panel">
      <div ref={containerRef} className="debug-panel__view">
        <canvas ref={canvasRef} width={VIEW_W} height={VIEW_H} className="debug-panel__overlay" />
      </div>
      <div ref={statusRef} className="debug-panel__status">
        waiting for camera…
      </div>
    </div>
  );
}
