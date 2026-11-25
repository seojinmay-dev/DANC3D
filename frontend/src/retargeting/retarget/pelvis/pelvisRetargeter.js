import * as THREE from 'three';
import {findCharacterMotionRoot} from '../core/characterRoot.js';
import {trackingAlpha} from '../core/trackingSmoothing.js';

// Moves the character's motion root (the ancestor holding Hips and the skinned meshes).
//  - 'screen' (webcam): hip center movement on screen, scaled by character/user screen leg length.
//    Its height only matters in the air: live grounding rests the feet on the floor.
//  - 'world' (reference): 3D hip displacement scaled by character/dancer leg length.
// Matches the webcam's requested 640×480 when the frame size is unknown.
export const DEFAULT_IMAGE_ASPECT = 4 / 3;

// Mean hip→knee→ankle length on screen, in image-height units.
export function screenLegLengthOf(screenJoints, imageAspect) {
  const point = name => screenJoints?.[name] && new THREE.Vector2(screenJoints[name].x * imageAspect, screenJoints[name].y);
  let total = 0;
  for (const side of ['left', 'right']) {
    const hip = point(`${side}Hip`), knee = point(`${side}Knee`), ankle = point(`${side}Ankle`);
    if (!hip || !knee || !ankle) return NaN;
    total += hip.distanceTo(knee) + knee.distanceTo(ankle);
  }
  return total / 2;
}

export function createPelvisRetargeter(character, {positionMode = 'screen', smoothing = true, rootDelta = () => new THREE.Quaternion()} = {}) {
  const leftFoot = character.model.getObjectByName('mixamorigLeftFoot');
  const rightFoot = character.model.getObjectByName('mixamorigRightFoot');
  if (!leftFoot || !rightFoot) throw new Error('Pelvis constraint용 Foot bone을 찾지 못했습니다.');

  const motionRoot = findCharacterMotionRoot(character);
  const rootRestPosition = motionRoot.position.clone();
  // Hip displacement in world space: target and smoothed value.
  const targetDisplacement = new THREE.Vector3(), displacement = new THREE.Vector3();

  character.model.updateMatrixWorld(true);
  const worldPosition = object => object.getWorldPosition(new THREE.Vector3());
  const rootRestWorldPosition = worldPosition(motionRoot);
  // Thigh + shin, the same leg length the leg IK and the foot anchors scale by.
  const chain = (upLeg, leg, foot) => worldPosition(upLeg).distanceTo(worldPosition(leg)) + worldPosition(leg).distanceTo(worldPosition(foot));
  const characterLegLength = Math.max((
    chain(character.bones.leftUpLeg, character.bones.leftLeg, leftFoot) +
    chain(character.bones.rightUpLeg, character.bones.rightLeg, rightFoot)
  ) * 0.5, 0.001);
  // The root pivots at the character origin (the floor); rotations must pivot at the hips.
  const hips = character.model.getObjectByName('mixamorigHips');
  const hipsOffset = worldPosition(hips).sub(rootRestWorldPosition);
  const neutralHipCenter = new THREE.Vector2();
  const neutralWorldHip = new THREE.Vector3();
  let worldMovementScale = 1;
  // Screen mode: image x/y are normalized by width/height separately.
  let imageAspect = DEFAULT_IMAGE_ASPECT, screenMovementScale = 1;
  let calibrated = false;

  function calibrate(pose) {
    const center = pose?.root?.screenPosition ?? null;
    if (positionMode === 'world' ? !pose?.root?.position : !center) {
      console.error('Pelvis calibration 실패: hip landmarks 없음');
      return false;
    }
    if (center) neutralHipCenter.copy(center);

    const joints = pose.joints;
    if (positionMode === 'world') {
      neutralWorldHip.copy(pose.root.position);
      const sourceLegLength = (
        joints.leftHip.distanceTo(joints.leftKnee) + joints.leftKnee.distanceTo(joints.leftAnkle) +
        joints.rightHip.distanceTo(joints.rightKnee) + joints.rightKnee.distanceTo(joints.rightAnkle)
      ) * 0.5;
      if (!Number.isFinite(sourceLegLength) || sourceLegLength <= 0.000001) return false;
      // Preserve motion relative to the dancer's size, in character world units.
      worldMovementScale = characterLegLength / sourceLegLength;
    } else {
      imageAspect = pose.metadata?.imageAspect ?? DEFAULT_IMAGE_ASPECT;
      // Standing straight at calibration: the screen leg length matches the character's leg.
      const screenLegLength = screenLegLengthOf(pose.screenJoints, imageAspect);
      if (!Number.isFinite(screenLegLength) || screenLegLength <= 0.000001) return false;
      screenMovementScale = characterLegLength / screenLegLength;
    }

    motionRoot.position.copy(rootRestPosition);
    targetDisplacement.set(0, 0, 0); displacement.set(0, 0, 0);
    character.model.updateMatrixWorld(true);
    calibrated = true;
    return true;
  }

  function setPose(pose) {
    if (!calibrated) return;
    if (positionMode === 'world') {
      if (!pose?.root?.position) return;
      targetDisplacement.copy(pose.root.position).sub(neutralWorldHip).multiplyScalar(worldMovementScale);
      targetDisplacement.x *= -1; // Mirrored contract, matching the limbs.
    } else {
      const center = pose?.root?.screenPosition;
      if (!center) return;
      // Mirrored webcam: user moving left (+dx) moves the character to screen left (-X).
      // Image y grows downwards, Three.js Y grows upwards.
      targetDisplacement.set(
        -(center.x - neutralHipCenter.x) * imageAspect * screenMovementScale,
        -(center.y - neutralHipCenter.y) * screenMovementScale,
        0
      );
    }
  }

  function update(deltaSeconds) {
    if (!calibrated) return;
    displacement.lerp(targetDisplacement, trackingAlpha(deltaSeconds, positionMode !== 'world' && smoothing));
    // Keep the hips where the displacement puts them, whatever the root rotation.
    const pivot = hipsOffset.clone().sub(hipsOffset.clone().applyQuaternion(rootDelta()));
    const targetWorld = rootRestWorldPosition.clone().add(displacement).add(pivot);
    // Convert into parent-local space (the root's parent may be scaled/rotated).
    motionRoot.parent.updateWorldMatrix(true, false);
    motionRoot.position.copy(motionRoot.parent.worldToLocal(targetWorld));
    character.model.updateMatrixWorld(true);
  }

  return {
    calibrate,
    setPose,
    update,
    getDebugState: () => ({actualX: motionRoot.position.x, actualY: motionRoot.position.y})
  };
}
