import * as THREE from 'three';
import {createDanceSyncPose} from '../danceSyncPose.js';

const MEDIAPIPE_JOINTS = {
  leftShoulder: 11, rightShoulder: 12, leftElbow: 13, rightElbow: 14, leftWrist: 15, rightWrist: 16,
  leftHip: 23, rightHip: 24, leftKnee: 25, rightKnee: 26, leftAnkle: 27, rightAnkle: 28
};

// World landmarks may be {x,y,z,visibility} objects or [x,y,z] arrays (as sent by the API).
export function createMediaPipePoseAdapter() {
  function convert(landmarks, worldLandmarks, timestamp = performance.now()) {
    if (!landmarks || !worldLandmarks) return null;
    const joints = {}, screenJoints = {}, visibility = {};
    for (const [name, index] of Object.entries(MEDIAPIPE_JOINTS)) {
      const world = worldLandmarks[index], screen = landmarks[index];
      if (!world || !screen) return null;
      joints[name] = Array.isArray(world)
        ? new THREE.Vector3().fromArray(world)
        : new THREE.Vector3(world.x, world.y, world.z);
      screenJoints[name] = new THREE.Vector2(screen.x, screen.y);
      visibility[name] = Math.min(world.visibility ?? 1, screen.visibility ?? 1);
    }
    return createDanceSyncPose({joints, screenJoints, timestamp, metadata: {visibility}});
  }
  return {convert};
}
