// Only retain the exact frames accepted by the readiness API, before countdown.
const JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const CHECKS = ['ready', 'full_body_visible', 'arms_horizontal', 'arms_extended', 'arms_spread'];

export function createCalibrationCapture({minSamples = 3, maxSamples = 5, maxGapMs = 750} = {}) {
  let samples = [];
  const reset = () => { samples = []; };
  function accept(frame, readiness) {
    const image = frame?.image_landmarks, world = frame?.world_landmarks;
    if (!CHECKS.every(key => readiness?.[key] === true) || !Number.isFinite(frame?.timestamp_ms) ||
        image?.length !== 33 || world?.length !== 33 || !JOINTS.every(i =>
          image[i]?.visibility >= 0.5 && [image[i].x, image[i].y, image[i].z].every(Number.isFinite) &&
          world[i]?.length === 3 && world[i].every(Number.isFinite))) {
      reset(); return null;
    }
    const last = samples.at(-1);
    if (last && frame.timestamp_ms <= last.timestamp_ms) return null;
    if (last && frame.timestamp_ms - last.timestamp_ms > maxGapMs) reset();
    // Copies prevent camera updates or the countdown pose from changing calibration.
    samples.push({...frame, image_landmarks: image.map(p => p ? {...p} : null),
      world_landmarks: world.map(p => p ? [...p] : null)});
    samples = samples.slice(-maxSamples);
    if (samples.length < minSamples) return null;
    const torso = Math.hypot(...[0, 1, 2].map(axis =>
      (world[11][axis] + world[12][axis] - world[23][axis] - world[24][axis]) / 2));
    if (torso < 0.05) { reset(); return null; }
    const averaged = {...frame, sample_count: samples.length,
      image_landmarks: image.map(p => p ? {...p} : null), world_landmarks: world.map(p => p ? [...p] : null)};
    for (const i of JOINTS) {
      averaged.world_landmarks[i] = [0, 1, 2].map(axis =>
        samples.reduce((sum, sample) => sum + sample.world_landmarks[i][axis], 0) / samples.length);
      for (const key of ['x', 'y', 'z', 'visibility']) {
        averaged.image_landmarks[i][key] = samples.reduce((sum, sample) => sum + sample.image_landmarks[i][key], 0) / samples.length;
      }
      if (samples.some(sample => Math.hypot(...sample.world_landmarks[i].map((v, axis) =>
        v - averaged.world_landmarks[i][axis])) > torso * 0.15 ||
        Math.hypot(sample.image_landmarks[i].x - averaged.image_landmarks[i].x,
          sample.image_landmarks[i].y - averaged.image_landmarks[i].y) > 0.03)) {
        samples = samples.slice(-1); return null;
      }
    }
    return averaged;
  }
  return {accept, reset};
}

export function calibrationPose(adapter, frame) {
  if (!frame?.image_landmarks || !frame?.world_landmarks) return null;
  const pose = adapter.convert(frame.image_landmarks, frame.world_landmarks, frame.timestamp_ms);
  // Normalized image x and y use different units; the avatar needs the ratio to scale motion.
  if (pose && frame.image_width > 0 && frame.image_height > 0) {
    pose.metadata.imageAspect = frame.image_width / frame.image_height;
  }
  return pose;
}
