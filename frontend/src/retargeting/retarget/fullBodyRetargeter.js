import {createArmRetargeter} from './arms/armRetargeter.js';
import {createLegRetargeter} from './legs/legRetargeter.js';
import {createPelvisRetargeter} from './pelvis/pelvisRetargeter.js';
import {createRootOrientationRetargeter} from './root/rootOrientationRetargeter.js';
import {createSpineRetargeter} from './spine/spineRetargeter.js';

// Torso noise handling, shared by the reference and live avatars so both stay comparable.
// Hip landmarks are the least stable: small sideways hip tilts are treated as noise.
export const BODY = {pelvisTilt: 1, pelvisTiltDeadZoneDeg: 4, spineDeadZoneDeg: 0, spineHalfLife: undefined};

export function createFullBodyRetargeter(character, {pelvisPositionMode = 'screen', smoothing = true, body: overrides = {}} = {}) {
  const body = {...BODY, ...overrides};
  const rootOrientation = createRootOrientationRetargeter(character, {smoothing,
    tilt: body.pelvisTilt, tiltDeadZoneDeg: body.pelvisTiltDeadZoneDeg});
  const rootDelta = () => rootOrientation.rootDelta();
  const pelvis = createPelvisRetargeter(character, {positionMode: pelvisPositionMode, smoothing, rootDelta});
  const spine = createSpineRetargeter(character, {smoothing, rootDelta,
    deadZoneDeg: body.spineDeadZoneDeg, halfLife: body.spineHalfLife,
    maxBackwardDeg: body.spineMaxBackwardDeg});
  const arms = createArmRetargeter(character, {smoothing});
  const legs = createLegRetargeter(character, {smoothing, rootDelta});

  let calibrated = false;
  let latestPose = null;

  function calibrate(pose) {
    const results = {
      arms: arms.calibrate(pose),
      legs: legs.calibrate(pose),
      pelvis: pelvis.calibrate(pose),
      spine: spine.calibrate(pose),
      rootOrientation: rootOrientation.calibrate(pose)
    };
    calibrated = Object.values(results).every(Boolean);
    if (!calibrated) console.warn('Retargeter calibration 실패:', results);
    return calibrated;
  }

  function setPose(pose) {
    if (calibrated) latestPose = pose;
  }

  // Parents before children; every sub-retargeter refreshes world matrices after its update.
  function update(deltaSeconds = 1/60) {
    if (!calibrated || !latestPose) return;
    const pose = latestPose;
    rootOrientation.setPose(pose);
    rootOrientation.update(deltaSeconds);
    pelvis.setPose(pose);
    pelvis.update(deltaSeconds);
    // The chest follows the torso, so the arm frame below already includes the spine.
    spine.setPose(pose);
    spine.update(deltaSeconds);
    arms.setPose(pose);
    arms.update(deltaSeconds);
    legs.setPose(pose);
    legs.update(deltaSeconds);
  }

  return {
    calibrate,
    setPose,
    update,
    getDebugState: () => ({arms: arms.getDebugState(), pelvis: pelvis.getDebugState()})
  };
}
