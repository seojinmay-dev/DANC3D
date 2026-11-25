import * as THREE from 'three';
import {DANCE_SYNC_JOINTS, createDanceSyncPose} from '../danceSyncPose.js';
import {createSyntheticCalibrationPose} from '../syntheticCalibration.js';

// AI Hub reference frames are fed as-is, so the reference avatar follows the same
// mirrored contract as the webcam: dancer left drives character right (a mirror image).
export function createAIHubPoseAdapter() {
  function convert(frame) {
    if (!frame?.joints) return null;
    const joints = {};
    for (const name of DANCE_SYNC_JOINTS) {
      if (!frame.joints[name]) return null;
      joints[name] = new THREE.Vector3().fromArray(frame.joints[name]);
    }
    return createDanceSyncPose({
      joints,
      rootScreenPosition: frame.rootScreenPosition ? new THREE.Vector2().fromArray(frame.rootScreenPosition) : null,
      timestamp: frame.timestamp ?? 0
    });
  }
  return {
    convert,
    // Upright T-pose facing the source front (+Z, dancer's right towards -X) with the frame's
    // proportions, standing straight with its ankles at the sequence's floor ankle height.
    createCalibrationPose: (frame, {ankleHeight} = {}) => createSyntheticCalibrationPose(convert(frame), {
      right: new THREE.Vector3(-1, 0, 0), up: new THREE.Vector3(0, 1, 0), ankleHeight
    })
  };
}

// Body yaw: angle of the dancer's left → right axis on the floor (source Y is up).
function bodyYaw(joints) {
  const dx = joints.rightShoulder[0] - joints.leftShoulder[0] + joints.rightHip[0] - joints.leftHip[0];
  const dz = joints.rightShoulder[2] - joints.leftShoulder[2] + joints.rightHip[2] - joints.leftHip[2];
  return Math.atan2(dz, dx);
}

// AI Hub world axes are arbitrary per capture ("camera extrinsic not reapplied"). The dancer
// faces the video camera most of the time, so rotate the sequence about the vertical axis until
// that dominant facing is the front (+Z). The avatar then faces the user on average instead of
// copying whatever direction the first frame happens to face.
// Ankle height of planted feet: a low percentile over the whole sequence (source Y is up).
export function floorAnkleHeight(data) {
  const heights = data.frames.flatMap(frame => [frame.joints.leftAnkle[1], frame.joints.rightAnkle[1]]).sort((a, b) => a - b);
  return heights[Math.floor((heights.length - 1) * .05)];
}

export function alignToDominantFacing(data) {
  const yaws = data.frames.map(frame => bodyYaw(frame.joints));
  const facing = Math.atan2(yaws.reduce((a, y) => a + Math.sin(y), 0), yaws.reduce((a, y) => a + Math.cos(y), 0));
  const rotation = facing - Math.PI; // front-facing: left → right axis points to -X
  const quaternion = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), rotation);
  const frames = data.frames.map(frame => ({
    ...frame,
    joints: Object.fromEntries(Object.entries(frame.joints).map(([name, value]) =>
      [name, new THREE.Vector3().fromArray(value).applyQuaternion(quaternion).toArray()]))
  }));
  return {...data, frames, facingRotationDeg: THREE.MathUtils.radToDeg(rotation)};
}
