import * as THREE from 'three';
import {createOrthonormalFrame} from '../retarget/core/bodyFrames.js';
import {midpoint3D} from '../retarget/core/retargetMath3D.js';

export const DANCE_SYNC_JOINTS = [
  'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
  'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
];

// Torso frame: right averages the shoulder and hip axes; up points hips → shoulders.
function createBodyFrame(joints) {
  const right = joints.rightShoulder.clone().sub(joints.leftShoulder).normalize()
    .add(joints.rightHip.clone().sub(joints.leftHip).normalize());
  const up = midpoint3D(joints.leftShoulder, joints.rightShoulder)
    .sub(midpoint3D(joints.leftHip, joints.rightHip));
  return createOrthonormalFrame(right, up);
}

// Source-independent pose consumed by the retargeter.
export function createDanceSyncPose({joints, screenJoints = {}, rootScreenPosition = null, timestamp = 0, metadata = {}}) {
  for (const name of DANCE_SYNC_JOINTS) {
    if (!(joints[name] instanceof THREE.Vector3)) {
      throw new Error(`DanceSyncPose joint가 없거나 Vector3가 아닙니다: ${name}`);
    }
  }
  const {leftHip, rightHip} = screenJoints;
  const screenPosition = rootScreenPosition?.clone()
    ?? (leftHip && rightHip ? leftHip.clone().add(rightHip).multiplyScalar(0.5) : null);
  return {
    timestamp,
    joints,
    screenJoints,
    root: {position: midpoint3D(joints.leftHip, joints.rightHip), screenPosition},
    bodyFrame: createBodyFrame(joints),
    metadata
  };
}
