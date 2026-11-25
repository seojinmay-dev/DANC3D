import * as THREE from 'three';
import {createDanceSyncPose} from './danceSyncPose.js';

// Source-independent T-pose synthesis: preserve measured proportions.
// The body axes default to the pose's own; pass `right`/`up` for an upright, front-facing T-pose.
// `ankleHeight` (along `up`) stands the T-pose straight on the floor instead of at the frame's hip height.
export function createSyntheticCalibrationPose(pose, {right: rightAxis, up: upAxis, ankleHeight} = {}) {
  if (!pose) return null;
  const source = pose.joints;
  const right = (rightAxis ?? pose.bodyFrame.right).clone().normalize(), left = right.clone().negate();
  const up = (upAxis ?? pose.bodyFrame.up).clone().normalize(), down = up.clone().negate();
  const length = (from, to) => source[from].distanceTo(source[to]);
  const hipCenter = source.leftHip.clone().add(source.rightHip).multiplyScalar(0.5);
  const torsoLength = hipCenter.distanceTo(source.leftShoulder.clone().add(source.rightShoulder).multiplyScalar(0.5));
  if (Number.isFinite(ankleHeight)) {
    const legLength = (length('leftHip', 'leftKnee') + length('leftKnee', 'leftAnkle') +
      length('rightHip', 'rightKnee') + length('rightKnee', 'rightAnkle')) / 2;
    hipCenter.addScaledVector(up, ankleHeight + legLength - hipCenter.dot(up));
  }
  const shoulderCenter = hipCenter.clone().addScaledVector(up, torsoLength);
  const hipWidth = length('leftHip', 'rightHip'), shoulderWidth = length('leftShoulder', 'rightShoulder');

  const joints = {
    leftHip: hipCenter.clone().addScaledVector(left, hipWidth * 0.5),
    rightHip: hipCenter.clone().addScaledVector(right, hipWidth * 0.5),
    leftShoulder: shoulderCenter.clone().addScaledVector(left, shoulderWidth * 0.5),
    rightShoulder: shoulderCenter.clone().addScaledVector(right, shoulderWidth * 0.5)
  };
  // Arms straight out along the body's right axis, legs straight down.
  const chain = (start, mid, end, direction) => {
    joints[mid] = joints[start].clone().addScaledVector(direction, length(start, mid));
    joints[end] = joints[mid].clone().addScaledVector(direction, length(mid, end));
  };
  chain('leftShoulder', 'leftElbow', 'leftWrist', left);
  chain('rightShoulder', 'rightElbow', 'rightWrist', right);
  chain('leftHip', 'leftKnee', 'leftAnkle', down);
  chain('rightHip', 'rightKnee', 'rightAnkle', down);

  return createDanceSyncPose({joints, rootScreenPosition: new THREE.Vector2(0.5, 0.5), timestamp: pose.timestamp});
}
