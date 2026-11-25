// Web-only display smoothing for the realtime score. The backend's displayed_score is a
// per-frame EMA (alpha 0.2, kept for Desktop Demo/research parity), so its speed depends on
// the evaluation rate. Here the RAW FinalScore is smoothed in wall-clock seconds instead,
// and the shown number only changes every `holdSec`. Records keep using RAW means.
export const SCORE_TIME_CONSTANT_SEC = 1.0;
export const SCORE_DISPLAY_HOLD_SEC = 0.5;
// Pauses or tracking gaps must not snap the score to one new frame.
export const SCORE_MAX_STEP_SEC = 0.25;

export function createScoreDisplay({timeConstantSec = SCORE_TIME_CONSTANT_SEC, holdSec = SCORE_DISPLAY_HOLD_SEC, maxStepSec = SCORE_MAX_STEP_SEC} = {}) {
  let smoothed = null, lastTime = null, shown = null, shownAt = null;
  return {
    // raw: RAW FinalScore 0..1 (null/NaN keeps the previous value). nowSec: monotonic seconds.
    // Returns the integer score to show, or null before the first valid score.
    update(raw, nowSec) {
      const dt = lastTime === null ? 0 : Math.min(maxStepSec, Math.max(0, nowSec - lastTime));
      lastTime = nowSec;
      if (!Number.isFinite(raw)) return shown;
      const target = raw * 100;
      if (smoothed === null) smoothed = target;
      else smoothed += (1 - Math.exp(-dt / timeConstantSec)) * (target - smoothed);
      if (shown === null || nowSec - shownAt >= holdSec) {
        shown = Math.round(smoothed);
        shownAt = nowSec;
      }
      return shown;
    },
    reset() { smoothed = lastTime = shown = shownAt = null; },
    get smoothed() { return smoothed; }
  };
}
