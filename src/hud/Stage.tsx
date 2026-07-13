import { useEffect, useState, type ReactNode } from "react";
import { STAGE_HEIGHT, STAGE_WIDTH } from "../app/constants";

/**
 * Scale factor that fits the fixed 2736×1216 wall canvas inside a viewport,
 * preserving aspect ratio (letterboxed on the shorter axis). Degenerate
 * viewports (jsdom reports 0×0) fall back to 1 so tests and headless
 * environments render the stage at natural size instead of collapsing it.
 */
export function stageScale(vw: number, vh: number): number {
  if (vw <= 0 || vh <= 0) return 1;
  return Math.min(vw / STAGE_WIDTH, vh / STAGE_HEIGHT);
}

/**
 * The single fixed-size stage everything renders into (ported from
 * LifeOfLLM). The wall itself is exactly 2736×1216; any other viewport
 * sees the same composition scaled to fit, so layout is authored once
 * against the wall's real canvas.
 */
export default function Stage({ children }: { children: ReactNode }) {
  const [scale, setScale] = useState(() =>
    typeof window === "undefined" ? 1 : stageScale(window.innerWidth, window.innerHeight)
  );

  useEffect(() => {
    const onResize = () => setScale(stageScale(window.innerWidth, window.innerHeight));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return (
    <div className="stage-viewport">
      <div className="stage" style={{ transform: `scale(${scale})` }}>
        {children}
      </div>
    </div>
  );
}
