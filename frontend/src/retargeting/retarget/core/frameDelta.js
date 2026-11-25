import * as THREE from 'three';
import {createOrthonormalFrame} from './bodyFrames.js';

// Rotations between body frames. Frames are right-handed bases (right, up, right × up).

export const frameQuaternion = matrix => new THREE.Quaternion().setFromRotationMatrix(matrix);

// User pelvis frame around gravity. Heading (yaw) comes from the torso's left → right axis
// (shoulders + hips: four landmarks, steadier than the two hips); the sideways tilt comes from
// the hip axis. Forward pitch cannot be measured at the pelvis; the spine takes it.
export function pelvisFrame(pose, worldUp) {
  const joints = pose?.joints;
  const torsoRight = pose?.bodyFrame?.right;
  if (!joints?.leftHip || !joints?.rightHip || !torsoRight) return null;
  const up = worldUp.clone().normalize();
  const hipAxis = joints.rightHip.clone().sub(joints.leftHip);
  const heading = torsoRight.clone().addScaledVector(up, -torsoRight.dot(up));
  if (heading.lengthSq() < 1e-10) return null;
  const hipHorizontal = hipAxis.clone().addScaledVector(up, -hipAxis.dot(up)).length();
  const right = heading.normalize().multiplyScalar(hipHorizontal).addScaledVector(up, hipAxis.dot(up));
  return createOrthonormalFrame(right, up);
}

// The webcam is mirrored (user left drives character right): reflect a rotation expressed
// in body-local axes across the local X (right) axis.
export const mirrorLocal = q => new THREE.Quaternion(q.x, -q.y, -q.z, q.w);

// Rotation taking frame `from` to frame `to`, expressed in `from`'s local axes.
export const localDelta = (from, to) => from.clone().invert().multiply(to);

// A rotation given in `basis`'s local axes, as a world-space rotation.
export const inWorld = (basis, local) => basis.clone().multiply(local).multiply(basis.clone().invert());

// Character pelvis basis from its hip joints and world up (rig rest or current pose).
export function characterPelvisQuaternion(character) {
  character.model.updateMatrixWorld(true);
  const position = bone => bone.getWorldPosition(new THREE.Vector3());
  const right = position(character.bones.rightUpLeg).sub(position(character.bones.leftUpLeg));
  return frameQuaternion(createOrthonormalFrame(right, new THREE.Vector3(0, 1, 0)).matrix);
}

// Soft dead zone on a rotation's angle: rotations below ~2·deadZone are progressively
// suppressed (landmark noise), larger ones pass with a constant offset. Smooth (C1).
export function softDeadZone(q, deadZone) {
  if (!(deadZone > 0)) return q.clone();
  const w = THREE.MathUtils.clamp(Math.abs(q.w), -1, 1);
  const angle = 2 * Math.acos(w);
  if (angle < 1e-9) return new THREE.Quaternion();
  const reduced = angle < 2 * deadZone ? angle * angle / (4 * deadZone) : angle - deadZone;
  const axis = new THREE.Vector3(q.x, q.y, q.z).multiplyScalar(Math.sign(q.w) || 1).normalize();
  return new THREE.Quaternion().setFromAxisAngle(axis, reduced);
}

// Split a rotation into twist about `axis` and the remaining swing (q = swing · twist).
export function swingTwist(q, axis) {
  const projection = axis.clone().multiplyScalar(new THREE.Vector3(q.x, q.y, q.z).dot(axis));
  const twist = new THREE.Quaternion(projection.x, projection.y, projection.z, q.w);
  if (twist.lengthSq() < 1e-12) twist.identity(); else twist.normalize();
  const swing = q.clone().multiply(twist.clone().invert());
  return {swing, twist};
}

// Gravity direction from the calibration T-pose: the user stands straight, so the average
// hip → ankle axis points down. The legs are long and straight, a far steadier vertical than
// the short torso, whose depth MediaPipe often misjudges (a torso read as leaning would tilt
// the whole avatar). Falls back to the torso axis.
export function calibrationUp(pose) {
  const j = pose?.joints;
  if (j?.leftHip && j.rightHip && j.leftAnkle && j.rightAnkle) {
    const up = j.leftHip.clone().sub(j.leftAnkle).add(j.rightHip.clone().sub(j.rightAnkle));
    if (up.lengthSq() > 1e-10) return up.normalize();
  }
  return pose?.bodyFrame?.up?.clone().normalize() ?? null;
}
