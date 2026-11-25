import * as THREE from 'three';
import {getDirection3D} from './retargetMath3D.js';
import {createOrthonormalFrame} from './bodyFrames.js';

// Direction retargeting: at calibration the user's bone direction (in their body frame)
// is paired with the character's rest direction. Each frame, the user's rotation away
// from that neutral direction is applied to the character's rest direction.

export function createCharacterBodyFrame(character) {
  const hips = character.model.getObjectByName('mixamorigHips');
  character.model.updateMatrixWorld(true);
  const leftShoulder = character.bones.leftArm.getWorldPosition(new THREE.Vector3());
  const rightShoulder = character.bones.rightArm.getWorldPosition(new THREE.Vector3());
  const hipPosition = hips.getWorldPosition(new THREE.Vector3());
  const shoulderCenter = new THREE.Vector3().addVectors(leftShoulder, rightShoulder).multiplyScalar(0.5);
  const right = new THREE.Vector3().subVectors(rightShoulder, leftShoulder);
  const up = new THREE.Vector3().subVectors(shoulderCenter, hipPosition);
  return createOrthonormalFrame(right, up).matrix;
}

const toParentSpace = (bone, worldDirection) => worldDirection.clone()
  .applyQuaternion(bone.parent.getWorldQuaternion(new THREE.Quaternion()).invert())
  .normalize();

export function createDirectionState(character, config, characterBodyFrame) {
  character.model.updateMatrixWorld(true);
  const bonePosition = config.bone.getWorldPosition(new THREE.Vector3());
  const childPosition = config.childBone.getWorldPosition(new THREE.Vector3());
  const directionWorld = new THREE.Vector3().subVectors(childPosition, bonePosition).normalize();
  return {
    restQuaternion: config.bone.quaternion.clone(),
    targetQuaternion: config.bone.quaternion.clone(),
    restDirectionParent: toParentSpace(config.bone, directionWorld),
    restDirectionBody: directionWorld.clone().transformDirection(characterBodyFrame.clone().invert()).normalize(),
    neutralUserDirection: new THREE.Vector3(),
    hasNeutral: false
  };
}

export function getUserDirectionLocal(config, pose, bodyFrame) {
  const direction = getDirection3D(pose?.joints?.[config.fromJoint], pose?.joints?.[config.toJoint]);
  if (!direction) return null;
  const local = direction.transformDirection(bodyFrame.clone().invert()).normalize();
  // The webcam view is mirrored: user left drives character right.
  if (config.mirrorX) local.x *= -1;
  return local.normalize();
}

export function calibrateDirectionBone(config, state, pose, bodyFrame) {
  const neutral = getUserDirectionLocal(config, pose, bodyFrame);
  if (!neutral) return false;
  config.bone.quaternion.copy(state.restQuaternion);
  state.targetQuaternion.copy(state.restQuaternion);
  state.neutralUserDirection.copy(neutral);
  state.hasNeutral = true;
  return true;
}

export function updateDirectionBoneTarget(character, config, state, pose, bodyFrame, characterBodyFrame) {
  if (!state.hasNeutral) return;
  const currentDirection = getUserDirectionLocal(config, pose, bodyFrame);
  if (!currentDirection) return;

  const userDelta = new THREE.Quaternion().setFromUnitVectors(state.neutralUserDirection, currentDirection);
  const targetBodyDirection = state.restDirectionBody.clone().applyQuaternion(userDelta).normalize();

  if (config.preserveBodyFrontDepth && config.depthConstraintActive) {
    // Keep the swing, but carry over only the calibration-relative front/back depth.
    // At neutral this returns exactly to the Mixamo rest depth.
    targetBodyDirection.z = state.restDirectionBody.z + (currentDirection.z - state.neutralUserDirection.z);
    targetBodyDirection.normalize();
  }

  const targetWorldDirection = targetBodyDirection.transformDirection(characterBodyFrame).normalize();
  character.model.updateMatrixWorld(true);
  const targetDirectionParent = toParentSpace(config.bone, targetWorldDirection);
  state.targetQuaternion
    .setFromUnitVectors(state.restDirectionParent, targetDirectionParent)
    .multiply(state.restQuaternion);
}

export function applyDirectionBone(config, state, alpha) {
  config.bone.quaternion.slerp(state.targetQuaternion, alpha);
}
