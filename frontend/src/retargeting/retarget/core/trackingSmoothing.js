// A common response time for live bones/root. The input filter handles sensor noise.
export const TRACKING_HALF_LIFE_SEC = 0.007;

// `smoothing`: false = snap, true = default half-life, or a half-life in seconds.
export function trackingAlpha(deltaSeconds = 1 / 60, smoothing = true) {
  const halfLife = smoothing === true ? TRACKING_HALF_LIFE_SEC : smoothing;
  if (!halfLife) return 1;
  if (!Number.isFinite(deltaSeconds) || deltaSeconds <= 0) return 0;
  return 1 - Math.exp(-Math.LN2 * Math.min(deltaSeconds, 0.1) / halfLife);
}
