import { useEffect, useRef } from 'react';

/**
 * Runs `step` once per interval (scaled by speed) while active. `step` returns false to stop.
 * The latest `step` closure is always used, so callers need not memoise it.
 */
export function useInterval(active: boolean, intervalMs: number, speed: number, step: () => boolean | void, onStop?: () => void) {
  const fn = useRef(step);
  fn.current = step;
  const stop = useRef(onStop);
  stop.current = onStop;
  useEffect(() => {
    if (!active) return;
    let raf = 0, acc = 0, last = performance.now();
    const tick = (now: number) => {
      acc += Math.min(250, now - last) * speed;
      last = now;
      let guard = 0;
      while (acc >= intervalMs && guard++ < 50) {
        acc -= intervalMs;
        if (fn.current() === false) { stop.current?.(); return; }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active, intervalMs, speed]);
}
