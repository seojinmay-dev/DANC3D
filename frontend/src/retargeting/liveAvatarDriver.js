import {createFullBodyRetargeter} from './retarget/fullBodyRetargeter.js';
import {createMediaPipePoseAdapter} from './pose/adapters/mediaPipePoseAdapter.js';
import {createLiveGrounding} from './liveGrounding.js';
import {calibrationPose} from '../services/avatarCalibration.js';
import {createLivePoseFilter} from '../services/livePoseFilter.js';

// Webcam estimates need gentler torso handling than recorded 3D reference data.
// Keep deliberate bends responsive while suppressing small pelvis/torso errors.
export const LIVE_BODY = {
  pelvisTilt: .85,
  pelvisTiltDeadZoneDeg: 4,
  spineDeadZoneDeg: 1,
  spineHalfLife: .015,
  spineMaxBackwardDeg: 8
};

// The whole webcam → avatar path, one call per rendered frame (`grounding: false` is for diagnostics):
// filter the newest camera frame, predict landmarks for the render time, retarget, ground.
export function createLiveAvatarDriver(character, calibrationFrame, {smoothing = true, prediction, filter: filterOptions, grounding: useGrounding = true, body} = {}) {
  const adapter = createMediaPipePoseAdapter();
  const calibration = calibrationPose(adapter, calibrationFrame);
  const retargeter = createFullBodyRetargeter(character, {
    pelvisPositionMode: 'screen', smoothing, body: {...LIVE_BODY, ...body}
  });
  if (!calibration || !retargeter.calibrate(calibration)) return null;
  const grounding = createLiveGrounding(character, calibration);
  const filter = createLivePoseFilter(calibrationFrame, {...(prediction && {prediction}), ...(filterOptions && {filter: filterOptions})});
  const toPose = sample => {
    const pose = adapter.convert(sample.image_landmarks, sample.world_landmarks, sample.timestamp_ms);
    pose.metadata.trackingParts = sample.parts;
    return pose;
  };
  return {
    retargeter,
    grounding,
    // `now` must share the camera frames' clock (performance.now()).
    update(frame, now, deltaSeconds) {
      const filtered = filter.update(frame, now);
      // Foot contacts use measured frames only; the limbs follow the prediction.
      if (filtered) grounding.observe(toPose(filtered), filtered.timestamp_ms);
      const predicted = filter.predict(now);
      if (predicted) retargeter.setPose(toPose(predicted));
      if (filter.canAnimate(now)) {
        if (useGrounding) grounding.beforeRetarget();
        retargeter.update(deltaSeconds);
        if (useGrounding) grounding.apply(now, deltaSeconds);
      } else grounding.loseTracking();
    }
  };
}
