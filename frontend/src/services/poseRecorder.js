// Developer tool: record the webcam landmark stream (no video) to tune the live avatar offline.
// Enable with ?recordPose in the URL; Shift+R downloads the recording as JSON.
const MAX_FRAMES = 30 * 180; // ~3 minutes at 30 fps
const round = v => Math.round(v * 1e5) / 1e5;
const compact = frame => frame && {
  timestamp_ms: frame.timestamp_ms, image_width: frame.image_width, image_height: frame.image_height,
  image_landmarks: frame.image_landmarks?.map(p => p && {x: round(p.x), y: round(p.y), z: round(p.z),
    visibility: round(p.visibility ?? 1), presence: round(p.presence ?? 1)}) ?? null,
  world_landmarks: frame.world_landmarks?.map(p => p && p.map(round)) ?? null,
  world_confidence: frame.world_confidence?.map(round) ?? null
};

export const poseRecordingEnabled = () => {
  try { return new URLSearchParams(window.location.search).has('recordPose'); } catch { return false; }
};

export function createPoseRecorder(calibrationFrame) {
  const frames = [];
  let lastTimestamp = -Infinity;
  return {
    push(frame) {
      if (!Number.isFinite(frame?.timestamp_ms) || frame.timestamp_ms <= lastTimestamp) return;
      lastTimestamp = frame.timestamp_ms;
      frames.push(compact(frame));
      if (frames.length > MAX_FRAMES) frames.shift();
    },
    download() {
      const blob = new Blob([JSON.stringify({version: 1, calibration: compact(calibrationFrame), frames})], {type: 'application/json'});
      const link = Object.assign(document.createElement('a'), {
        href: URL.createObjectURL(blob), download: `pose-recording-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
      });
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
      return frames.length;
    }
  };
}
