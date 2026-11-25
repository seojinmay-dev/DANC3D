import * as THREE from 'three';
import {findCharacterMotionRoot} from '../core/characterRoot.js';
import {trackingAlpha} from '../core/trackingSmoothing.js';
import {
  characterPelvisQuaternion, frameQuaternion, inWorld, localDelta, mirrorLocal, pelvisFrame, softDeadZone, swingTwist, calibrationUp
} from '../core/frameDelta.js';

// Rotates the character's motion root by the user's pelvis rotation since calibration:
// turning (yaw) and sideways hip tilt. Forward pitch is left to the spine.
// `tilt` scales the sideways hip tilt (0 = turning only); `tiltDeadZoneDeg` ignores small, noisy tilts.
export function createRootOrientationRetargeter(character, {smoothing = true, tilt = 1, tiltDeadZoneDeg = 0} = {}) {
  const root = findCharacterMotionRoot(character);
  const restQuaternion = root.quaternion.clone();
  const targetQuaternion = root.quaternion.clone();
  character.model.updateMatrixWorld(true);
  const restWorld = root.getWorldQuaternion(new THREE.Quaternion());
  const characterPelvis = characterPelvisQuaternion(character);
  let worldUp, calibrationPelvis;
  let calibrated = false;

  const userPelvis = pose => {
    const frame = pelvisFrame(pose, worldUp);
    return frame && frameQuaternion(frame.matrix);
  };

  function calibrate(pose) {
    // The calibration T-pose stands upright: its leg axis is the user's gravity direction,
    // which also absorbs a tilted webcam.
    if (!pose?.bodyFrame) return false;
    worldUp = calibrationUp(pose);
    calibrationPelvis = userPelvis(pose);
    if (!calibrationPelvis) return false;
    root.quaternion.copy(restQuaternion);
    targetQuaternion.copy(restQuaternion);
    character.model.updateMatrixWorld(true);
    calibrated = true;
    return true;
  }

  function setPose(pose) {
    if (!calibrated) return;
    const current = userPelvis(pose);
    if (!current) return;
    // Body-local delta: twist about local up = turning, swing = sideways tilt.
    const {swing, twist} = swingTwist(mirrorLocal(localDelta(calibrationPelvis, current)), new THREE.Vector3(0, 1, 0));
    const tilted = softDeadZone(new THREE.Quaternion().slerp(swing, tilt), THREE.MathUtils.degToRad(tiltDeadZoneDeg));
    const rotation = inWorld(characterPelvis, tilted.multiply(twist));
    const targetWorld = rotation.multiply(restWorld);
    targetQuaternion.copy(root.parent.getWorldQuaternion(new THREE.Quaternion()).invert().multiply(targetWorld));
  }

  function update(deltaSeconds) {
    if (!calibrated) return;
    root.quaternion.slerp(targetQuaternion, trackingAlpha(deltaSeconds, smoothing));
    character.model.updateMatrixWorld(true);
  }

  return {
    calibrate, setPose, update,
    worldUp: () => worldUp,
    // Root rotation applied since rest, in world space.
    rootDelta: () => root.getWorldQuaternion(new THREE.Quaternion()).multiply(restWorld.clone().invert())
  };
}
