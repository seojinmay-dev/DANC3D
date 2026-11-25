// A real AI Hub reference sequence as a "user" dancing it in front of a webcam.
import {readFileSync} from 'node:fs';
import * as THREE from 'three';
import {createCamera, tPoseFacing} from './syntheticDancer.mjs';

const yawOf = j => Math.atan2(j.rightShoulder[2] - j.leftShoulder[2], j.rightShoulder[0] - j.leftShoulder[0]);

export function loadReferenceMotion(id, {windowSec = 10} = {}) {
  const data = JSON.parse(readFileSync(new URL(`../../public/references/${id}.json`, import.meta.url)));
  const times = data.frames.map((f, i) => f.timestamp ?? i / data.fps);
  const duration = times.at(-1);
  const poseAt = t => {
    const i = Math.min(Math.max(0, Math.floor(t * data.fps)), data.frames.length - 2);
    const alpha = THREE.MathUtils.clamp((t - times[i]) / (times[i + 1] - times[i]), 0, 1);
    const a = data.frames[i].joints, b = data.frames[i + 1].joints;
    return Object.fromEntries(Object.keys(a).map(k => [k, new THREE.Vector3().fromArray(a[k]).lerp(new THREE.Vector3().fromArray(b[k]), alpha)]));
  };
  // The dancer mostly faces the video camera: put the webcam in front of the dominant facing.
  const yaws = data.frames.map(f => yawOf(f.joints));
  const facing = Math.atan2(yaws.reduce((a, y) => a + Math.sin(y), 0), yaws.reduce((a, y) => a + Math.cos(y), 0));
  const front = new THREE.Vector3(Math.sin(facing), 0, -Math.cos(facing));
  const hips = data.frames.map(f => new THREE.Vector3().fromArray(f.joints.leftHip).add(new THREE.Vector3().fromArray(f.joints.rightHip)).multiplyScalar(.5));
  const center = hips.reduce((a, h) => a.add(h), new THREE.Vector3()).divideScalar(hips.length); center.y = 0;
  const camera = createCamera({position: center.clone().addScaledVector(front, 3.6).setY(1), forward: front.clone().negate()});
  const segments = [];
  for (let start = 0; start < duration; start += windowSec) {
    segments.push({name: `${start}–${Math.min(duration, start + windowSec).toFixed(0)}초`, start, end: Math.min(duration, start + windowSec)});
  }
  return {
    id, data, duration, poseAt, camera, segments,
    facingOffsetDeg: THREE.MathUtils.radToDeg(Math.atan2(Math.sin(yaws[0] - facing), Math.cos(yaws[0] - facing))),
    // The user calibrates facing the webcam where the dance starts.
    calibrationJoints: () => tPoseFacing(camera, poseAt(0)),
    latencyWindows: {fast: [1, duration - 1], slow: [1, duration - 1]},
    holdWindow: [0, 0]
  };
}
