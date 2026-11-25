import * as THREE from 'three';
import {createUpperBodyFrame} from '../core/bodyFrames.js';
import {
  createCharacterBodyFrame, createDirectionState, getUserDirectionLocal,
  calibrateDirectionBone, updateDirectionBoneTarget, applyDirectionBone
} from '../core/boneRetarget.js';
import {trackingAlpha} from '../core/trackingSmoothing.js';

const trackingPart = config => config.fromJoint.startsWith('left') ? 'leftArm' : 'rightArm';
const isTracked = (pose, config) => pose?.metadata?.trackingParts?.[trackingPart(config)] !== false;

export function createArmRetargeter(character, {smoothing = true} = {}) {
  const leftHand = character.model.getObjectByName('mixamorigLeftHand');
  const rightHand = character.model.getObjectByName('mixamorigRightHand');
  if (!leftHand || !rightHand) throw new Error('손 bone을 찾지 못했습니다.');

  // Mirrored: the user's left arm drives the character's right arm.
  const upperArms = {
    userLeft: {bone: character.bones.rightArm, childBone: character.bones.rightForeArm,
      fromJoint: 'leftShoulder', toJoint: 'leftElbow', mirrorX: true},
    userRight: {bone: character.bones.leftArm, childBone: character.bones.leftForeArm,
      fromJoint: 'rightShoulder', toJoint: 'rightElbow', mirrorX: true}
  };
  const foreArms = {
    userLeft: {bone: character.bones.rightForeArm, childBone: rightHand,
      fromJoint: 'leftElbow', toJoint: 'leftWrist', mirrorX: true, preserveBodyFrontDepth: true},
    userRight: {bone: character.bones.leftForeArm, childBone: leftHand,
      fromJoint: 'rightElbow', toJoint: 'rightWrist', mirrorX: true, preserveBodyFrontDepth: true}
  };
  const all = [...Object.values(upperArms), ...Object.values(foreArms)];

  const restCharacterBodyFrame = createCharacterBodyFrame(character);
  const states = new Map(all.map(config => [config, createDirectionState(character, config, restCharacterBodyFrame)]));

  let calibrated = false;
  let latestPose, latestFrame, latestCharacterFrame;

  // Depth correction is only meaningful when the elbow is bent enough to define an arm plane.
  function isArmPlaneStable(upper, fore, pose, frame) {
    const upperDirection = getUserDirectionLocal(upper, pose, frame);
    const foreDirection = getUserDirectionLocal(fore, pose, frame);
    if (!upperDirection || !foreDirection) return false;
    return new THREE.Vector3().crossVectors(upperDirection, foreDirection).lengthSq() >= 0.0025;
  }

  function calibrate(pose) {
    const frame = createUpperBodyFrame(pose);
    if (!frame) return false;
    for (const config of all) {
      if (!calibrateDirectionBone(config, states.get(config), pose, frame)) return false;
    }
    calibrated = true;
    return true;
  }

  function setPose(pose) {
    if (!calibrated) return;
    const frame = createUpperBodyFrame(pose);
    if (!frame) return;
    const characterFrame = createCharacterBodyFrame(character);
    latestPose = pose; latestFrame = frame; latestCharacterFrame = characterFrame;

    for (const side of Object.keys(foreArms)) {
      foreArms[side].depthConstraintActive = isArmPlaneStable(upperArms[side], foreArms[side], pose, frame);
    }
    // An untracked limb keeps its previous target.
    for (const config of all) {
      if (isTracked(pose, config)) {
        updateDirectionBoneTarget(character, config, states.get(config), pose, frame, characterFrame);
      }
    }
  }

  function update(deltaSeconds) {
    if (!calibrated) return;
    const alpha = trackingAlpha(deltaSeconds, smoothing);
    for (const config of Object.values(upperArms)) applyDirectionBone(config, states.get(config), alpha);
    character.model.updateMatrixWorld(true);

    for (const config of Object.values(foreArms)) {
      // Re-solve against this render's upper-arm rotation, not the previous render's parent.
      if (smoothing && isTracked(latestPose, config)) {
        updateDirectionBoneTarget(character, config, states.get(config), latestPose, latestFrame, latestCharacterFrame);
      }
      applyDirectionBone(config, states.get(config), alpha);
    }
    character.model.updateMatrixWorld(true);
  }

  return {
    calibrate,
    setPose,
    update,
    getDebugState: () => Object.fromEntries(Object.entries(foreArms).map(([side, config]) =>
      [side, {targetQuaternion: states.get(config).targetQuaternion.toArray()}]))
  };
}
