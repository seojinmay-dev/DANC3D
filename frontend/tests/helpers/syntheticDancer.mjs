// One synthetic 3D dance rendered both as an AI Hub reference sequence and as a
// MediaPipe webcam stream, so the reference and live avatar paths can be compared.
// Canonical space: meters, Y up, floor at Y = 0, the dancer's left at +X, facing +Z (the camera).
import * as THREE from 'three';

export const BODY = {shoulderHalf: .18, hipHalf: .1, torso: .5, upperArm: .28, foreArm: .26, thigh: .44, shin: .42, ankle: .08};
const HIP_HEIGHT = BODY.thigh + BODY.shin + BODY.ankle;
const deg = THREE.MathUtils.degToRad;
const smooth = (t, a, b) => THREE.MathUtils.smoothstep(t, a, b);
const pulse = (t, a, b) => smooth(t, a, a + .3) * (1 - smooth(t, b - .3, b)); // 0 → 1 → 0 inside [a, b]
const wave = (t, period) => Math.sin(2 * Math.PI * t / period);

// Named segments let the report attribute differences to a kind of movement.
export const SEGMENTS = [
  {name: 'T포즈 유지', start: 0, end: 1},
  {name: '팔 동작', start: 1, end: 3.5},
  {name: '스쿼트', start: 3.5, end: 5.5},
  {name: '무릎 들기', start: 5.5, end: 7.5},
  {name: '몸 회전', start: 7.5, end: 9.5},
  {name: '체중이동+앞뒤 이동', start: 9.5, end: 11.5},
  {name: '팔 앞으로 뻗기', start: 11.5, end: 13},
  {name: '느린 팔 동작', start: 13, end: 17}
];
export const DURATION = 17;

function limbDirection(side, frame, elevation, forward) {
  // elevation: 0 = horizontal T-pose, -90° = hanging; forward: swing towards the front.
  const out = frame.right.clone().multiplyScalar(side * Math.cos(elevation) * Math.cos(forward))
    .addScaledVector(frame.front, Math.cos(elevation) * Math.sin(forward))
    .addScaledVector(frame.up, Math.sin(elevation));
  return out.normalize();
}

function bend(direction, frame, angle) {
  // Bend the forearm towards the front (or up when the upper arm points forwards).
  let axis = frame.front.clone().addScaledVector(direction, -frame.front.dot(direction));
  if (axis.lengthSq() < 1e-4) axis = frame.up.clone().addScaledVector(direction, -frame.up.dot(direction));
  return direction.clone().multiplyScalar(Math.cos(angle)).addScaledVector(axis.normalize(), Math.sin(angle)).normalize();
}

// Full body pose at time t (seconds).
export function poseAt(t) {
  const yaw = deg(30) * pulse(t, 7.5, 9.5) * wave(t - 7.5, 2);
  const frame = {
    right: new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw)), // dancer's left (+X) when yaw = 0
    up: new THREE.Vector3(0, 1, 0),
    front: new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw))
  };
  // Root translation: weight shift sideways and walking towards / away from the camera.
  const shift = .12 * pulse(t, 9.5, 11.5) * wave(t - 9.5, 1);
  const travel = .35 * pulse(t, 9.5, 11.5);
  // Squat: ankles stay under the hips, so the hips drop.
  const squat = deg(80) * pulse(t, 3.5, 5.5) * Math.abs(wave(t - 3.5, 2));
  const lift = side => deg(70) * pulse(t, 5.5, 7.5) * Math.max(0, side * wave(t - 5.5, 1));

  const floorHip = new THREE.Vector3(0, 0, travel); // hip center projected on the floor (before shift)
  const hipCenter = floorHip.clone().add(new THREE.Vector3(shift, 0, 0));
  const legReach = BODY.thigh + BODY.shin;
  const sideLean = Math.asin(THREE.MathUtils.clamp(-shift / legReach, -1, 1));
  hipCenter.y = BODY.ankle + legReach * Math.cos(squat / 2) * Math.cos(sideLean);

  const joints = {};
  for (const [side, name] of [[1, 'left'], [-1, 'right']]) {
    const hip = hipCenter.clone().addScaledVector(frame.right, side * BODY.hipHalf);
    const thighPitch = squat / 2 + lift(side), shinPitch = -squat / 2;
    const leanDown = frame.up.clone().negate().applyAxisAngle(frame.front, sideLean);
    const legDirection = pitch => leanDown.clone().applyAxisAngle(frame.right, -pitch).normalize();
    joints[name + 'Hip'] = hip;
    joints[name + 'Knee'] = hip.clone().addScaledVector(legDirection(thighPitch), BODY.thigh);
    joints[name + 'Ankle'] = joints[name + 'Knee'].clone().addScaledVector(legDirection(shinPitch), BODY.shin);

    // Arms: a T-pose that waves, lowers, rises and reaches forwards.
    const armsActive = pulse(t, 1, 3.5), reach = pulse(t, 11.5, 13), yawArms = pulse(t, 7.5, 9.5);
    const slow = pulse(t, 13, 17);
    const elevation = armsActive * deg(55) * wave(t - 1 + (side === 1 ? 0 : .3), 1.25)
      - reach * deg(10) + yawArms * deg(30) * wave(t, .8) + slow * deg(40) * wave(t - 13, 4);
    const forward = armsActive * deg(35) * (1 + wave(t - 1, 2.5)) / 2 + reach * deg(80);
    const elbow = armsActive * deg(70) * (1 + wave(t - 1.2, .9)) / 2 + yawArms * deg(40);
    const shoulder = hipCenter.clone().addScaledVector(frame.up, BODY.torso).addScaledVector(frame.right, side * BODY.shoulderHalf);
    const upperDirection = limbDirection(side, frame, elevation, forward);
    joints[name + 'Shoulder'] = shoulder;
    joints[name + 'Elbow'] = shoulder.clone().addScaledVector(upperDirection, BODY.upperArm);
    joints[name + 'Wrist'] = joints[name + 'Elbow'].clone().addScaledVector(bend(upperDirection, frame, elbow), BODY.foreArm);
  }
  return joints;
}

export function tPose() {
  return poseAt(0);
}

// ---------- AI Hub reference format ----------
export function aiHubSequence({fps = 30, joints = poseAt, duration = DURATION} = {}) {
  const frames = [];
  for (let i = 0; i <= Math.round(duration * fps); i++) {
    const t = i / fps, pose = joints(t);
    frames.push({frameIndex: i, timestamp: t, rootScreenPosition: [.5, .5],
      joints: Object.fromEntries(Object.entries(pose).map(([k, v]) => [k, v.toArray()]))});
  }
  return {fps, frames};
}

// ---------- MediaPipe webcam format ----------
const MEDIAPIPE_INDEX = {
  leftShoulder: 11, rightShoulder: 12, leftElbow: 13, rightElbow: 14, leftWrist: 15, rightWrist: 16,
  leftHip: 23, rightHip: 24, leftKnee: 25, rightKnee: 26, leftAnkle: 27, rightAnkle: 28
};
// Pinhole webcam. `forward` is the viewing direction; image x follows forward × up.
export function createCamera({position = new THREE.Vector3(0, 1, 3.2), forward = new THREE.Vector3(0, 0, -1),
  focal = 1.1, width = 640, height = 480} = {}) {
  const up = new THREE.Vector3(0, 1, 0);
  const right = new THREE.Vector3().crossVectors(forward, up).normalize();
  return {position, forward: forward.clone().normalize(), right, up, focal, width, height};
}
export const CAMERA = createCamera();

// Normalized image coordinates like MediaPipe.
export function project(point, camera = CAMERA) {
  const relative = point.clone().sub(camera.position), depth = relative.dot(camera.forward);
  return {
    x: .5 + camera.focal * relative.dot(camera.right) / depth / (camera.width / camera.height),
    y: .5 - camera.focal * relative.dot(camera.up) / depth
  };
}

// MediaPipe-like landmark noise: depth (z) much worse than x/y, hips worse than other joints,
// and a temporally correlated wobble on top of frame-to-frame jitter.
// `shoulderDepthBias` (m): MediaPipe often misjudges the torso's depth, so the whole torso reads
// as leaning (positive = shoulders further from the camera than they are).
export function createNoiseModel({noise = .004, random = Math.random, depthScale = 3, hipScale = 2, wobbleSec = .15,
  shoulderDepthBias = 0} = {}) {
  const gauss = () => Math.sqrt(-2 * Math.log(random() + 1e-12)) * Math.cos(2 * Math.PI * random());
  const wobble = new Map();
  let lastTime;
  const scaleOf = index => (index === 23 || index === 24 ? hipScale : 1);
  return {
    advance(timeSec) {
      const dt = lastTime === undefined ? 0 : timeSec - lastTime; lastTime = timeSec;
      const keep = Math.exp(-dt / wobbleSec), add = Math.sqrt(1 - keep * keep);
      for (const [key, value] of wobble) wobble.set(key, value * keep + add * gauss());
    },
    sample(kind, index, axis) {
      const key = `${kind}${index}${axis}`;
      if (!wobble.has(key)) wobble.set(key, gauss());
      const sigma = noise * (kind === 'world' ? 2 : 1) * scaleOf(index) * (kind === 'world' && axis === 2 ? depthScale : 1);
      const bias = kind === 'world' && axis === 2 && (index === 11 || index === 12) ? shoulderDepthBias : 0;
      return bias + sigma * (gauss() + wobble.get(key)) / Math.SQRT2;
    }
  };
}

export function mediaPipeFrame(timestampMs, joints, {noise = 0, random = Math.random, camera = CAMERA, noiseModel} = {}) {
  const gauss = () => Math.sqrt(-2 * Math.log(random() + 1e-12)) * Math.cos(2 * Math.PI * random());
  noiseModel?.advance(timestampMs / 1000);
  const hipCenter = joints.leftHip.clone().add(joints.rightHip).multiplyScalar(.5);
  const image = Array.from({length: 33}, () => ({x: .5, y: .5, z: 0, visibility: .99, presence: .99}));
  const world = Array.from({length: 33}, () => [0, 0, 0]);
  for (const [name, index] of Object.entries(MEDIAPIPE_INDEX)) {
    const p = joints[name], screen = project(p, camera), local = p.clone().sub(hipCenter);
    // MediaPipe world: hip-centered meters in camera axes, y down, -z towards the camera.
    const worldNoise = axis => noiseModel ? noiseModel.sample('world', index, axis) : noise * 2 * gauss();
    const imageNoise = axis => noiseModel ? noiseModel.sample('image', index, axis) : noise * gauss();
    world[index] = [local.dot(camera.right), -local.dot(camera.up), local.dot(camera.forward)].map((v, axis) => v + worldNoise(axis));
    image[index] = {x: screen.x + imageNoise(0), y: screen.y + imageNoise(1), z: 0, visibility: .99, presence: .99};
  }
  return {timestamp_ms: timestampMs, image_width: camera.width, image_height: camera.height,
    image_landmarks: image, world_landmarks: world, pose_detected: true};
}

// A user T-pose facing the camera, with the dancer's proportions, standing at `hipCenter`'s floor spot.
export function tPoseFacing(camera, joints) {
  const length = (a, b) => joints[a].distanceTo(joints[b]);
  const right = camera.right.clone().negate(); // the user's right appears on the image left
  const up = new THREE.Vector3(0, 1, 0), down = up.clone().negate(), left = right.clone().negate();
  const hip = joints.leftHip.clone().add(joints.rightHip).multiplyScalar(.5);
  const legs = (length('leftHip', 'leftKnee') + length('leftKnee', 'leftAnkle') + length('rightHip', 'rightKnee') + length('rightKnee', 'rightAnkle')) / 2;
  const ankleHeight = Math.min(joints.leftAnkle.y, joints.rightAnkle.y);
  hip.y = ankleHeight + legs;
  const shoulder = hip.clone().addScaledVector(up, hip.distanceTo(joints.leftShoulder.clone().add(joints.rightShoulder).multiplyScalar(.5)));
  const out = {
    leftHip: hip.clone().addScaledVector(left, length('leftHip', 'rightHip') / 2),
    rightHip: hip.clone().addScaledVector(right, length('leftHip', 'rightHip') / 2),
    leftShoulder: shoulder.clone().addScaledVector(left, length('leftShoulder', 'rightShoulder') / 2),
    rightShoulder: shoulder.clone().addScaledVector(right, length('leftShoulder', 'rightShoulder') / 2)
  };
  const chain = (a, b, c, direction) => {
    out[b] = out[a].clone().addScaledVector(direction, length(a, b));
    out[c] = out[b].clone().addScaledVector(direction, length(b, c));
  };
  chain('leftShoulder', 'leftElbow', 'leftWrist', left);
  chain('rightShoulder', 'rightElbow', 'rightWrist', right);
  chain('leftHip', 'leftKnee', 'leftAnkle', down);
  chain('rightHip', 'rightKnee', 'rightAnkle', down);
  return out;
}

export function seededRandom(seed = 1) {
  let state = seed;
  return () => (state = (state * 16807) % 2147483647) / 2147483647;
}
