import * as THREE from 'three';
import {findCharacterMotionRoot} from './retarget/core/characterRoot.js';
import {DEFAULT_IMAGE_ASPECT} from './retarget/pelvis/pelvisRetargeter.js';
import {solveTwoBoneIK} from './twoBoneIK.js';
import {
  getMirroredLeg, findNaturalFloor, translateRoot, rootYaw, fitPelvisToReach,
  createSoleCorners, lowestCornerY, liftAboveFloor, approach
} from './groundingCommon.js';

const SIDES = ['left', 'right'];
// Heights/speeds are in screen leg lengths (per second for speeds).
const LIFT_HEIGHT = .075, CONTACT_HEIGHT = .045;
const RELEASE_SPEED = .55, CONTACT_SPEED = .2, CONTACT_HOLD_SEC = .12, ANKLE_SMOOTHING_SEC = .08;

const midpoint = (a, b) => a.clone().add(b).multiplyScalar(.5);

// Screen feet remain stationary during a squat; MediaPipe's hip-relative world Y does not.
// Distances use image-height units (x is rescaled by the aspect ratio).
export function createLiveFootContacts(calibration) {
  const aspect = calibration.metadata?.imageAspect ?? DEFAULT_IMAGE_ASPECT;
  const point = (pose, name) => {
    const p = pose.screenJoints?.[name];
    return p && Number.isFinite(p.x) && Number.isFinite(p.y) ? new THREE.Vector2(p.x * aspect, p.y) : null;
  };
  const torsoOf = pose => {
    const [ls, rs, lh, rh] = ['leftShoulder', 'rightShoulder', 'leftHip', 'rightHip'].map(name => point(pose, name));
    return ls && rs && lh && rh ? midpoint(ls, rs).distanceTo(midpoint(lh, rh)) : null;
  };
  const calibrationLeg = Math.max(.05, ...SIDES.map(side =>
    point(calibration, side + 'Hip').distanceTo(point(calibration, side + 'Ankle'))));
  const calibrationTorso = torsoOf(calibration);
  // The floor line and body size adapt as the user moves around; tracking gaps keep them.
  let floor = Math.max(...SIDES.map(side => point(calibration, side + 'Ankle').y));
  let sizeRatio = 1;
  const states = Object.fromEntries(SIDES.map(side => [side, {}]));
  const clearState = state => Object.assign(state, {contact: false, lifted: false, candidate: 0, smoothed: null});
  let lastTime;
  function reset() {
    lastTime = undefined;
    Object.values(states).forEach(clearState);
  }
  reset();

  function observe(pose, time) {
    if (!Number.isFinite(time) || (lastTime !== undefined && time <= lastTime)) return;
    const dt = lastTime === undefined ? 0 : (time - lastTime) / 1000;
    if (dt > .25) reset();
    lastTime = time;

    // The body grows on screen when the user walks towards the camera.
    const torso = torsoOf(pose);
    if (calibrationTorso && torso) {
      sizeRatio = approach(sizeRatio, THREE.MathUtils.clamp(torso / calibrationTorso, .5, 2), dt, .5);
    }
    const legSize = calibrationLeg * sizeRatio;

    let floorCandidate = -Infinity;
    for (const side of SIDES) {
      const state = states[side], ankle = point(pose, side + 'Ankle');
      const valid = ankle && pose.metadata.trackingParts?.[side + 'Leg'] !== false &&
        [side + 'Hip', side + 'Knee', side + 'Ankle'].every(name => (pose.metadata.visibility?.[name] ?? 1) >= .45);
      if (!valid) { clearState(state); continue; }
      // Frame-to-frame landmark jitter would read as foot speed; measure motion on a
      // lightly smoothed ankle instead. Lifting is still detected on the raw height.
      const previous = state.smoothed;
      state.smoothed = previous ? previous.clone().lerp(ankle, 1 - Math.exp(-dt / ANKLE_SMOOTHING_SEC)) : ankle;
      const speed = previous && dt > 0 ? state.smoothed.distanceTo(previous) / legSize / dt : Infinity;
      const height = (floor - ankle.y) / legSize;
      const smoothedHeight = (floor - state.smoothed.y) / legSize;
      state.lifted = height > LIFT_HEIGHT;
      if (state.lifted || speed > RELEASE_SPEED) { state.contact = false; state.candidate = 0; }
      if (!state.contact) {
        state.candidate = smoothedHeight < CONTACT_HEIGHT && speed < CONTACT_SPEED ? state.candidate + Math.min(dt, .1) : 0;
        if (state.candidate >= CONTACT_HOLD_SEC) state.contact = true;
      }
      if (speed < CONTACT_SPEED) floorCandidate = Math.max(floorCandidate, state.smoothed.y);
    }
    // Stationary feet define the floor: follow quickly downwards (feet cannot be below it),
    // slowly upwards (a briefly hovering foot is not the floor).
    if (floorCandidate > -Infinity) floor = approach(floor, floorCandidate, dt, floorCandidate > floor ? .1 : .6);
  }

  return {
    observe, reset,
    isContact: side => states[side].contact,
    // Both feet clearly off the floor: a jump.
    isAirborne: () => SIDES.every(side => states[side].lifted),
    isFresh: now => lastTime !== undefined && now - lastTime <= 250
  };
}

// A display-only post-process: restore FK before each update so IK never feeds its
// corrections back into the temporal filter or the next frame's bone rotations.
//  1. Height: the lowest sole rests on the floor unless both feet are lifted (jump),
//     so screen-space hip noise or camera distance cannot make the avatar float or sink.
//  2. Planted feet are anchored horizontally (leashed to the FK foot) and flattened with IK.
//  3. No sole corner passes through the floor.
export function createLiveGrounding(character, calibration) {
  const contacts = createLiveFootContacts(calibration);
  const root = findCharacterMotionRoot(character);
  character.model.updateMatrixWorld(true);
  const initialRotation = root.getWorldQuaternion(new THREE.Quaternion());
  const feet = SIDES.map(side => ({side, ...getMirroredLeg(character, side), weight: 0, anchor: null}));
  const legLength = (feet[0].length + feet[1].length) / 2;
  const floor = findNaturalFloor(character, feet, legLength);
  for (const state of feet) {
    const sole = state.ankle.clone(); sole.y = floor;
    state.soleLocal = state.foot.worldToLocal(sole);
    state.forward = state.toePosition.clone().sub(state.ankle);
    state.forward.y = 0; state.forward.normalize();
    state.soleCorners = createSoleCorners(state, floor, legLength * .05);
  }
  const nodes = feet.flatMap(state => [state.upper, state.lower, state.foot]);
  const rawPosition = root.position.clone(), rawRotations = nodes.map(node => node.quaternion.clone());
  const pelvisShift = new THREE.Vector3();
  let groundWeight = 1;

  function beforeRetarget() {
    root.position.copy(rawPosition);
    nodes.forEach((node, i) => node.quaternion.copy(rawRotations[i]));
    character.model.updateMatrixWorld(true);
  }

  function apply(now, dt) {
    rawPosition.copy(root.position);
    rawRotations.forEach((rotation, i) => rotation.copy(nodes[i].quaternion));
    translateRoot(character, root, new THREE.Vector3(0, -floor, 0));
    const fresh = contacts.isFresh(now);

    groundWeight = approach(groundWeight, fresh && contacts.isAirborne() ? 0 : 1, dt, .05);
    translateRoot(character, root, new THREE.Vector3(0, -Math.min(...feet.map(lowestCornerY)) * groundWeight, 0));

    const yaw = rootYaw(root, initialRotation);
    const goals = [];
    for (const state of feet) {
      const active = fresh && contacts.isContact(state.side);
      // Eased both ways so a lifting or landing foot never pops.
      state.weight = approach(state.weight, active ? 1 : 0, dt, active ? .06 : .04);
      if (!active && state.weight < .01) { state.weight = 0; state.anchor = null; continue; }
      const sole = state.foot.localToWorld(state.soleLocal.clone());
      if (!state.anchor) { state.anchor = sole.clone(); state.anchor.y = 0; }
      if (active) {
        // Leash: tracking noise cannot slide a planted foot, but a real step drags the anchor along.
        const offset = state.anchor.clone().sub(sole); offset.y = 0;
        offset.clampLength(0, legLength * .1);
        state.anchor.copy(sole).add(offset).setY(0);
      }
      // The screen contact releases only once the ankle is clearly up. Before that, let a foot
      // whose FK sole is already rising leave the floor instead of pinning the thigh down.
      const weight = state.weight * (1 - THREE.MathUtils.smoothstep(lowestCornerY(state), legLength * .02, legLength * .06));
      if (weight < .01) continue;
      const flatRotation = state.restRotation.clone().premultiply(yaw);
      const rotation = state.foot.getWorldQuaternion(new THREE.Quaternion()).slerp(flatRotation, weight);
      const offset = state.soleLocal.clone().multiply(state.scale).applyQuaternion(rotation);
      const target = state.foot.getWorldPosition(new THREE.Vector3()).lerp(state.anchor.clone().sub(offset), weight);
      goals.push({state, target, rotation, pole: state.forward.clone().applyQuaternion(yaw),
        hip: state.upper.getWorldPosition(new THREE.Vector3()), length: state.length});
    }
    const shift = fitPelvisToReach(goals,
      {iterations: 6, maxDrop: legLength * .25, maxRise: legLength * .05, maxHorizontal: legLength * .05});
    pelvisShift.lerp(shift, 1 - Math.exp(-Math.max(0, dt) / .05));
    translateRoot(character, root, pelvisShift);
    for (const {state, target, rotation, pole} of goals) solveTwoBoneIK({
      upper: state.upper, lower: state.lower, foot: state.foot, target, pole, footRotation: rotation});
    for (const state of feet) liftAboveFloor(state, 0, state.forward.clone().applyQuaternion(yaw), solveTwoBoneIK);
    character.model.updateMatrixWorld(true);
  }

  return {
    observe: contacts.observe, beforeRetarget, apply,
    loseTracking() {
      contacts.reset();
      for (const state of feet) { state.weight = 0; state.anchor = null; }
      pelvisShift.set(0, 0, 0);
      groundWeight = 1;
    },
    getDebugState: () => feet.map(state => ({side: state.side, weight: state.weight,
      sole: state.foot.localToWorld(state.soleLocal.clone()).toArray()}))
  };
}
