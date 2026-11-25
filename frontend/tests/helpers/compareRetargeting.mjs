// Drive the reference (AI Hub bake) and live (webcam) avatar paths with the same dance
// (synthetic, or a real AI Hub sequence re-rendered as a webcam stream) and measure how far
// apart the two characters are, and how late the live one is.
import * as THREE from 'three';
import {loadRig} from './rig.mjs';
import {CAMERA, DURATION, SEGMENTS, aiHubSequence, createNoiseModel, mediaPipeFrame, poseAt, seededRandom} from './syntheticDancer.mjs';
import {createFullBodyRetargeter} from '../../src/retargeting/retarget/fullBodyRetargeter.js';
import {bakeReferenceAnimation} from '../../src/retargeting/referenceAnimation.js';
import {createLiveAvatarDriver} from '../../src/retargeting/liveAvatarDriver.js';
import {alignToDominantFacing} from '../../src/retargeting/pose/adapters/aiHubPoseAdapter.js';

export const BONES = {
  '팔(상완)': [['LeftArm', 'LeftForeArm'], ['RightArm', 'RightForeArm']],
  '팔(전완)': [['LeftForeArm', 'LeftHand'], ['RightForeArm', 'RightHand']],
  '허벅지': [['LeftUpLeg', 'LeftLeg'], ['RightUpLeg', 'RightLeg']],
  '정강이': [['LeftLeg', 'LeftFoot'], ['RightLeg', 'RightFoot']]
};
const PAIRS = Object.values(BONES).flat();
// Mirrored contract: a character bone is driven by the dancer's opposite-side segment.
const SOURCE_SEGMENTS = {
  LeftArm: ['rightShoulder', 'rightElbow'], LeftForeArm: ['rightElbow', 'rightWrist'],
  RightArm: ['leftShoulder', 'leftElbow'], RightForeArm: ['leftElbow', 'leftWrist'],
  LeftUpLeg: ['rightHip', 'rightKnee'], LeftLeg: ['rightKnee', 'rightAnkle'],
  RightUpLeg: ['leftHip', 'leftKnee'], RightLeg: ['leftKnee', 'leftAnkle']
};

// Ground truth: the dancer's own segment directions, faced front and mirrored like the avatars.
function truthSampler(motion) {
  const rotation = THREE.MathUtils.degToRad(alignToDominantFacing({frames: motion.data.frames}).facingRotationDeg);
  const quaternion = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), rotation);
  return time => {
    const joints = motion.poseAt(time);
    return PAIRS.map(([bone]) => {
      const [from, to] = SOURCE_SEGMENTS[bone];
      const d = joints[to].clone().sub(joints[from]).applyQuaternion(quaternion).normalize();
      d.x *= -1;
      return d;
    });
  };
}
const RENDER_HZ = 60;

function sampler(character) {
  const nodes = PAIRS.map(([a, b]) => [a, b].map(name => character.model.getObjectByName('mixamorig' + name)));
  const hips = character.model.getObjectByName('mixamorigHips');
  const position = o => o.getWorldPosition(new THREE.Vector3());
  return () => {
    character.model.updateMatrixWorld(true);
    return {
      directions: nodes.map(([a, b]) => position(b).sub(position(a)).normalize()),
      hips: position(hips),
      // Waist: hips → chest axis, and the hip line (for tilt).
      chest: position(character.model.getObjectByName('mixamorigNeck')).sub(position(hips)).normalize(),
      hipLine: position(character.bones.rightUpLeg).sub(position(character.bones.leftUpLeg)).normalize(),
      // Whole-body lean: hips relative to the midpoint of the feet.
      bodyAxis: position(hips).sub(position(character.model.getObjectByName('mixamorigLeftFoot'))
        .add(position(character.model.getObjectByName('mixamorigRightFoot'))).multiplyScalar(.5)).normalize(),
      // Body yaw from the hip line (character left hip → right hip), projected on the floor.
      rootYaw: (() => {
        const across = position(character.bones.rightUpLeg).sub(position(character.bones.leftUpLeg));
        return Math.atan2(across.z, across.x);
      })()
    };
  };
}

function legLength(character) {
  character.model.updateMatrixWorld(true);
  const p = name => character.model.getObjectByName('mixamorig' + name).getWorldPosition(new THREE.Vector3());
  return (p('LeftUpLeg').distanceTo(p('LeftFoot')) + p('RightUpLeg').distanceTo(p('RightFoot'))) / 2;
}

const quiet = async fn => {
  const {log, warn} = console; console.log = console.warn = () => {};
  try { return await fn(); } finally { console.log = log; console.warn = warn; }
};

// Reference character sampled on a fine clock (stateless playback).
async function referenceTrack(motion, step, grounding = true) {
  const character = loadRig();
  const retargeter = createFullBodyRetargeter(character, {pelvisPositionMode: 'world', smoothing: false});
  const playback = await quiet(() => bakeReferenceAnimation({data: motion.data, character, retargeter, grounding, yieldWork: async () => {}}));
  const sample = sampler(character), samples = [];
  for (let i = 0; i * step <= motion.duration; i++) { playback.apply(i * step); samples.push(sample()); }
  return {samples, legLength: legLength(character)};
}

// Live character driven exactly like DanceAvatar: 30 Hz sensor frames, 60 Hz render loop.
async function liveTrack(motion, {noise = 0, realisticNoise = false, shoulderDepthBias = 0, sensorHz = 30, processingDelayMs = 0, calibrationArmDrop = 0, seed = 1,
  smoothing = true, prediction, filter, grounding = true, body}) {
  const random = seededRandom(seed);
  const noiseModel = realisticNoise || shoulderDepthBias ? createNoiseModel({noise: realisticNoise ? noise || .004 : 0, random, shoulderDepthBias}) : undefined;
  const character = loadRig();
  const camera = motion.camera;
  const calibrationJoints = motion.calibrationJoints();
  if (calibrationArmDrop) {
    // An imperfect user T-pose: arms a little below horizontal.
    for (const side of ['left', 'right']) {
      const shoulder = calibrationJoints[side + 'Shoulder'];
      for (const joint of ['Elbow', 'Wrist']) {
        calibrationJoints[side + joint].sub(shoulder)
          .applyAxisAngle(camera.forward, (side === 'left' ? 1 : -1) * calibrationArmDrop).add(shoulder);
      }
    }
  }
  // The calibration frame carries the same systematic bias as the live stream.
  const calibrationNoise = shoulderDepthBias ? createNoiseModel({noise: 0, random, shoulderDepthBias}) : undefined;
  const driver = await quiet(() => createLiveAvatarDriver(character, mediaPipeFrame(0, calibrationJoints, {camera, noiseModel: calibrationNoise}), {smoothing, prediction, filter, grounding, body}));
  if (!driver) throw Error('live calibration failed');
  const sample = sampler(character), samples = [];
  let latest = null, nextSensor = 0, previousRender;
  for (let n = 0; n * 1000 / RENDER_HZ <= motion.duration * 1000; n++) {
    const now = n * 1000 / RENDER_HZ;
    // A frame captured at `capture` becomes available after the detector's processing delay.
    while (nextSensor * 1000 / sensorHz + processingDelayMs <= now) {
      const capture = nextSensor * 1000 / sensorHz;
      latest = mediaPipeFrame(capture, motion.poseAt(capture / 1000), {noise, random, camera, noiseModel});
      nextSensor++;
    }
    driver.update(latest, now, previousRender === undefined ? 1 / RENDER_HZ : (now - previousRender) / 1000);
    previousRender = now;
    samples.push({time: now / 1000, ...sample()});
  }
  return {samples, legLength: legLength(character)};
}

const angle = (a, b) => THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(a.dot(b), -1, 1)));

export const SYNTHETIC_MOTION = {
  data: aiHubSequence(), duration: DURATION, segments: SEGMENTS, poseAt, camera: CAMERA,
  calibrationJoints: () => poseAt(0),
  latencyWindows: {fast: [1.3, 3.5], slow: [13.5, 16.5]}, holdWindow: [.3, 1]
};

export async function compareRetargeting({motion = SYNTHETIC_MOTION, ...options} = {}) {
  const STEP = .002;
  const reference = await referenceTrack(motion, STEP, options.referenceGrounding ?? true);
  const live = await liveTrack(motion, options);
  const truth = truthSampler(motion);
  const referenceAt = time => reference.samples[Math.max(0, Math.min(reference.samples.length - 1, Math.round(time / STEP)))];
  const origin = {ref: referenceAt(.5).hips, live: live.samples.find(s => s.time >= .5).hips};

  // Latency: the time shift that best aligns the live limbs with the reference.
  const bestLagIn = (start, end) => {
    const moving = live.samples.filter(s => s.time > start && s.time < end);
    const errorAtLag = lag => moving.reduce((sum, s) => sum + PAIRS.reduce((acc, _, i) =>
      acc + angle(s.directions[i], referenceAt(s.time - lag).directions[i]), 0), 0) / moving.length / PAIRS.length;
    let lag = 0, error = Infinity;
    for (let candidate = 0; candidate <= .3; candidate += .002) {
      const e = errorAtLag(candidate);
      if (e < error) { error = e; lag = candidate; }
    }
    return {lag, error, errorAtZero: errorAtLag(0)};
  };
  const fast = bestLagIn(...motion.latencyWindows.fast), slow = bestLagIn(...motion.latencyWindows.slow);
  const bestLag = fast.lag, bestError = fast.error, errorAtLag = () => fast.errorAtZero;
  const segmentStats = lag => motion.segments.map(segment => {
    // Skip the first 0.3 s of each segment so the comparison is not dominated by the previous one.
    const samples = live.samples.filter(s => s.time >= segment.start + .3 && s.time <= segment.end);
    const stats = {};
    const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
    for (const [group, pairs] of Object.entries(BONES)) {
      const indices = pairs.map(pair => PAIRS.indexOf(pair));
      const errors = samples.flatMap(s => indices.map(i => angle(s.directions[i], referenceAt(s.time - lag).directions[i])));
      const truths = samples.map(s => truth(s.time));
      stats[group] = {
        mean: mean(errors), max: Math.max(...errors),
        // Each avatar against the dancer's real segment directions.
        referenceVsTruth: mean(samples.flatMap((s, k) => indices.map(i => angle(referenceAt(s.time).directions[i], truths[k][i])))),
        liveVsTruth: mean(samples.flatMap((s, k) => indices.map(i => angle(s.directions[i], truths[k][i]))))
      };
    }
    // Root: position relative to each character's own start, in each character's leg lengths.
    const rootErrors = samples.map(s => {
      const r = referenceAt(s.time - lag);
      const a = r.hips.clone().sub(origin.ref).divideScalar(reference.legLength);
      const b = s.hips.clone().sub(origin.live).divideScalar(live.legLength);
      return {horizontal: Math.hypot(a.x - b.x, a.z - b.z), lateral: Math.abs(a.x - b.x), vertical: Math.abs(a.y - b.y),
        yaw: Math.abs(THREE.MathUtils.radToDeg(Math.atan2(Math.sin(r.rootYaw - s.rootYaw), Math.cos(r.rootYaw - s.rootYaw))))};
    });
    const maxOf = key => Math.max(...rootErrors.map(e => e[key]));
    const meanOf = key => rootErrors.reduce((sum, e) => sum + e[key], 0) / rootErrors.length;
    return {name: segment.name, bones: stats, root: {horizontal: maxOf('horizontal'), lateral: maxOf('lateral'),
      vertical: maxOf('vertical'), yaw: maxOf('yaw'), meanYaw: meanOf('yaw'), meanHorizontal: meanOf('horizontal')}};
  });

  // Jitter: mean angular speed of the live limbs while the dancer holds still (deg/s).
  const hold = live.samples.filter(s => s.time > motion.holdWindow[0] && s.time < motion.holdWindow[1]);
  let jitter = 0;
  for (let i = 1; i < hold.length; i++) {
    jitter += PAIRS.reduce((acc, _, b) => acc + angle(hold[i].directions[b], hold[i - 1].directions[b]), 0) /
      PAIRS.length / (hold[i].time - hold[i - 1].time);
  }
  jitter /= hold.length - 1;
  const jitterByBone = Object.fromEntries(Object.entries(BONES).map(([group, pairs]) => {
    const indices = pairs.map(pair => PAIRS.indexOf(pair));
    let sum = 0;
    for (let i = 1; i < hold.length; i++) {
      sum += indices.reduce((acc, b) => acc + angle(hold[i].directions[b], hold[i - 1].directions[b]), 0) /
        indices.length / (hold[i].time - hold[i - 1].time);
    }
    return [group, +(sum / (hold.length - 1)).toFixed(1)];
  }));
  // Waist while holding still: chest-axis lean from vertical and hip-line tilt (degrees).
  const deg = v => THREE.MathUtils.radToDeg(v);
  const lean = hold.map(s => deg(Math.acos(THREE.MathUtils.clamp(s.chest.y, -1, 1))));
  const tilt = hold.map(s => Math.abs(deg(Math.asin(THREE.MathUtils.clamp(s.hipLine.y, -1, 1)))));
  const stats = values => ({mean: values.reduce((a, b) => a + b, 0) / values.length, max: Math.max(...values)});
  const bodyLean = hold.map(s => deg(Math.acos(THREE.MathUtils.clamp(s.bodyAxis.y, -1, 1))));
  // Signed forward(+)/backward(-) lean of the whole body over the whole run.
  const pitch = live.samples.map(s => deg(Math.atan2(s.bodyAxis.z, s.bodyAxis.y)));
  const referencePitch = reference.samples.map(s => deg(Math.atan2(s.bodyAxis.z, s.bodyAxis.y)));
  const waist = {lean: stats(lean), hipTilt: stats(tilt), bodyLean: stats(bodyLean),
    pitchMean: pitch.reduce((a, b) => a + b, 0) / pitch.length,
    referencePitchMean: referencePitch.reduce((a, b) => a + b, 0) / referencePitch.length};
  return {waist, trace: {reference, live, referenceAt, origin}, jitterDegPerSec: jitter, jitterByBone, segments: segmentStats(0), alignedSegments: segmentStats(bestLag), latencyMs: bestLag * 1000, slowLatencyMs: slow.lag * 1000, errorAtZeroLag: errorAtLag(0), errorAtBestLag: bestError};
}
