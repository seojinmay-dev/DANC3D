import * as THREE from 'three';
import {createCharacterBodyFrame} from '../core/boneRetarget.js';
import {trackingAlpha} from '../core/trackingSmoothing.js';
import {
  characterPelvisQuaternion, frameQuaternion, inWorld, localDelta, mirrorLocal, pelvisFrame, softDeadZone, calibrationUp
} from '../core/frameDelta.js';

const SPINE_BONES = ['mixamorigSpine', 'mixamorigSpine1', 'mixamorigSpine2'];
// Cumulative share of the torso rotation reached at each spine bone.
const SPINE_SHARE = [.35, .7, 1];
const IDENTITY = new THREE.Quaternion();

// In the character pelvis basis +Z points backwards. Limit only that component
// of chest-up; keep sideways lean and leave forward bends untouched.
export function limitBackwardBend(rotation, maxDegrees = Infinity) {
  if (!Number.isFinite(maxDegrees)) return rotation;
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(rotation);
  const pitch = Math.atan2(up.z, up.y);
  if (pitch <= 0) return rotation;
  const limit = THREE.MathUtils.degToRad(Math.max(.1, maxDegrees));
  const knee = limit * .5;
  if (pitch <= knee) return rotation;
  const limited = knee + (limit - knee) * Math.tanh((pitch - knee) / (limit - knee));
  return new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), limited - pitch)
    .multiply(rotation);
}

// Torso rotation relative to the pelvis (bending forward, leaning sideways, twisting),
// distributed along the spine so the chest ends up at the user's torso orientation.
// `deadZoneDeg` ignores small, noisy torso rotations; `halfLife` (seconds) overrides `smoothing`.
export function createSpineRetargeter(character, {smoothing = true, rootDelta, deadZoneDeg = 0, halfLife, maxBackwardDeg = Infinity} = {}) {
  const hips = character.model.getObjectByName('mixamorigHips');
  const bones = SPINE_BONES.map(name => character.model.getObjectByName(name));
  if (!hips || bones.some(bone => !bone)) throw new Error('Spine bone을 찾지 못했습니다.');
  const rest = bones.map(bone => bone.quaternion.clone());
  const targets = rest.map(q => q.clone());

  // Character torso basis expressed in its pelvis basis, at rest.
  const characterPelvis = characterPelvisQuaternion(character);
  const characterTorsoInPelvis = characterPelvis.clone().invert()
    .multiply(frameQuaternion(createCharacterBodyFrame(character)));

  let worldUp, calibrationTorsoInPelvis, pelvisLocalRotation = new THREE.Quaternion();
  let calibrated = false;

  const torsoInPelvis = pose => {
    const pelvis = pelvisFrame(pose, worldUp);
    if (!pelvis || !pose.bodyFrame) return null;
    return frameQuaternion(pelvis.matrix).invert().multiply(frameQuaternion(pose.bodyFrame.matrix));
  };

  function calibrate(pose) {
    if (!pose?.bodyFrame) return false;
    worldUp = calibrationUp(pose);
    calibrationTorsoInPelvis = torsoInPelvis(pose);
    if (!calibrationTorsoInPelvis) return false;
    bones.forEach((bone, i) => { bone.quaternion.copy(rest[i]); targets[i].copy(rest[i]); });
    pelvisLocalRotation.identity();
    calibrated = true;
    return true;
  }

  function setPose(pose) {
    if (!calibrated) return;
    const current = torsoInPelvis(pose);
    if (!current) return;
    // User torso change (torso-local, mirrored) → character pelvis axes.
    const change = softDeadZone(mirrorLocal(localDelta(calibrationTorsoInPelvis, current)), THREE.MathUtils.degToRad(deadZoneDeg));
    pelvisLocalRotation = limitBackwardBend(inWorld(characterTorsoInPelvis, change), maxBackwardDeg);
  }

  function update(deltaSeconds) {
    if (!calibrated) return;
    character.model.updateMatrixWorld(true);
    // The pelvis basis follows the root rotation applied this frame.
    const pelvis = rootDelta().multiply(characterPelvis);
    const rotation = inWorld(pelvis, pelvisLocalRotation);
    let restWorld = hips.getWorldQuaternion(new THREE.Quaternion());
    let parentDesired = restWorld.clone();
    const alpha = trackingAlpha(deltaSeconds, smoothing && halfLife !== undefined ? halfLife : smoothing);
    bones.forEach((bone, i) => {
      restWorld = restWorld.clone().multiply(rest[i]);
      const desired = IDENTITY.clone().slerp(rotation, SPINE_SHARE[i]).multiply(restWorld);
      targets[i].copy(parentDesired.clone().invert().multiply(desired));
      parentDesired = desired;
      bone.quaternion.slerp(targets[i], alpha);
    });
    character.model.updateMatrixWorld(true);
  }

  return {calibrate, setPose, update};
}
