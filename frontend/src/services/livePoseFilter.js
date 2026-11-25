// Visual-only adaptive low-pass filter. Never modify the raw frame used for scoring.
const PARTS = {body:[11,12,23,24], leftArm:[13,15], rightArm:[14,16], leftLeg:[25,27], rightLeg:[26,28]};
const MAX_AGE_MS = 250;
const HIPS = [23, 24];
const lowPassAlpha = (cutoff, dt) => 1 - Math.exp(-2 * Math.PI * cutoff * dt);

// Render-time prediction: the camera delivers ~30 Hz, the avatar renders at 60+ Hz.
// Extrapolating the filtered landmarks along their (smoothed) velocity fills the gap
// between sensor frames without the delay of a render-side smoothing filter.
// One-Euro parameters: minCutoffHz sets the smoothing at rest, beta how quickly it opens up with speed.
export const FILTER = {
  world: {minCutoffHz: .7, beta: 12},
  // MediaPipe depth is far noisier than x/y; filter the world z axis on its own.
  worldDepth: {minCutoffHz: .6, beta: 9},
  // Hip landmarks are the least reliable and anchor the waist/pelvis: their depth noise reads
  // as a wobbling, leaning body, and the pelvis moves slower than the limbs, so filter it harder.
  hipDepth: {minCutoffHz: .4, beta: 4},
  image: {minCutoffHz: .7, beta: 6}
};

export const PREDICTION = {
  velocityCutoffHz: 3,   // smoothing of the extrapolation velocity
  leadSec: 0,            // extra look-ahead beyond the newest frame's age
  maxAheadSec: .05       // never extrapolate further than this
};

// One-Euro style adaptive filter with an extrapolation velocity of its output.
function vectorFilter({minCutoffHz: minCutoff, beta}, prediction) {
  let raw, value, velocity = [0,0,0], outputVelocity = [0,0,0], timestamp;
  return {
    update(next, time) {
      if (!value || time - timestamp > MAX_AGE_MS) {
        raw = [...next]; value = [...next]; velocity = [0,0,0]; outputVelocity = [0,0,0]; timestamp = time;
        return [...value];
      }
      const dt = Math.max(0.001, (time - timestamp) / 1000);
      const derivativeAlpha = lowPassAlpha(1, dt);
      velocity = next.map((v,i) => velocity[i] + derivativeAlpha * ((v - raw[i]) / dt - velocity[i]));
      const alpha = lowPassAlpha(minCutoff + beta * Math.hypot(...velocity), dt);
      const previous = value;
      value = next.map((v,i) => value[i] + alpha * (v - value[i]));
      const velocityAlpha = lowPassAlpha(prediction.velocityCutoffHz, dt);
      outputVelocity = value.map((v,i) => outputVelocity[i] + velocityAlpha * ((v - previous[i]) / dt - outputVelocity[i]));
      raw = [...next]; timestamp = time;
      return [...value];
    },
    predict(time) {
      const ahead = Math.min(Math.max(0, (time - timestamp) / 1000 + prediction.leadSec), prediction.maxAheadSec);
      return value.map((v,i) => v + outputVelocity[i] * ahead);
    }
  };
}

// World landmarks: x/y and depth (z) filtered separately, recombined.
function worldFilter(filter, prediction, hip) {
  const depthOptions = (hip ? filter.hipDepth : null) ?? filter.worldDepth ?? filter.world;
  const plane = vectorFilter(filter.world, prediction), depth = vectorFilter(depthOptions, prediction);
  const join = ([x, y], [z]) => [x, y, z];
  return {
    update: (next, time) => join(plane.update([next[0], next[1]], time), depth.update([next[2]], time)),
    predict: time => join(plane.predict(time), depth.predict(time))
  };
}

export function createLivePoseFilter(seed, {prediction = PREDICTION, filter = FILTER} = {}) {
  let lastTimestamp = -Infinity, bodyTracked = false;
  const active = Object.fromEntries(Object.keys(PARTS).map(key => [key,false]));
  const worldFilters = new Map(), imageFilters = new Map();
  const image = seed?.image_landmarks?.map(p => p ? {...p} : null) || Array(33).fill(null);
  const world = seed?.world_landmarks?.map(p => p ? [...p] : null) || Array(33).fill(null);
  const confidence = (frame,i) => {
    const p = frame.image_landmarks?.[i], w = frame.world_landmarks?.[i];
    if (!p || ![p.x,p.y,p.z].every(Number.isFinite) || w?.length !== 3 || !w.every(Number.isFinite)) return 0;
    return Math.min(p.visibility ?? 0, p.presence ?? 1, frame.world_confidence?.[i] ?? 1);
  };
  return {
    canAnimate: now => bodyTracked && now - lastTimestamp <= MAX_AGE_MS,
    // Landmarks extrapolated to `now`; held (untracked) limbs keep their last value.
    predict(now) {
      if (!(bodyTracked && now - lastTimestamp <= MAX_AGE_MS)) return null;
      const predictedImage = image.map(p => p ? {...p} : null), predictedWorld = world.map(p => p ? [...p] : null);
      for (const [part, indices] of Object.entries(PARTS)) {
        if (!active[part]) continue;
        for (const i of indices) {
          if (!worldFilters.has(i)) continue;
          predictedWorld[i] = worldFilters.get(i).predict(now);
          const [x, y, z] = imageFilters.get(i).predict(now);
          predictedImage[i] = {...image[i], x, y, z};
        }
      }
      return {image_landmarks: predictedImage, world_landmarks: predictedWorld, timestamp_ms: now, parts: {...active}};
    },
    update(frame, now) {
      if (!Number.isFinite(frame?.timestamp_ms) || now - frame.timestamp_ms > MAX_AGE_MS) {
        bodyTracked = false; return null;
      }
      if (frame.timestamp_ms <= lastTimestamp) return null;
      lastTimestamp = frame.timestamp_ms;
      for (const [part, indices] of Object.entries(PARTS)) {
        // Hysteresis prevents rapid enable/disable when confidence hovers near the threshold.
        active[part] = indices.every(i => confidence(frame,i) >= (active[part] ? 0.45 : 0.65));
      }
      bodyTracked = active.body;
      if (!bodyTracked) return null;
      for (const [part, indices] of Object.entries(PARTS)) {
        if (!active[part]) continue;
        for (const i of indices) {
          if (!worldFilters.has(i)) {
            worldFilters.set(i, worldFilter(filter,prediction,HIPS.includes(i))); imageFilters.set(i, vectorFilter(filter.image,prediction));
          }
          world[i] = worldFilters.get(i).update(frame.world_landmarks[i],frame.timestamp_ms);
          const p = frame.image_landmarks[i];
          const xyz = imageFilters.get(i).update([p.x,p.y,p.z],frame.timestamp_ms);
          image[i] = {...p,x:xyz[0],y:xyz[1],z:xyz[2]};
        }
      }
      // A held limb needs a prior valid position (normally supplied by T-pose calibration).
      if (!Object.values(PARTS).flat().every(i => image[i] && world[i])) return null;
      return {image_landmarks:image.map(p=>p?{...p}:null),world_landmarks:world.map(p=>p?[...p]:null),
        timestamp_ms:lastTimestamp,parts:{...active}};
    }
  };
}
