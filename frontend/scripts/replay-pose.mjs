// Replay a webcam landmark recording (?recordPose, Shift+R) through the live avatar path
// on the real rig and report waist stability and jitter for one or more settings.
// Usage: node scripts/replay-pose.mjs recording.json ['[["label", {driver options}], ...]']
import {readFileSync} from 'node:fs';
import * as THREE from 'three';
import {loadRig} from '../tests/helpers/rig.mjs';
import {createLiveAvatarDriver} from '../src/retargeting/liveAvatarDriver.js';

const [file, configsJson] = process.argv.slice(2);
if (!file) { console.error('usage: node scripts/replay-pose.mjs recording.json [configs]'); process.exit(1); }
const recording = JSON.parse(readFileSync(file));
const configs = configsJson ? JSON.parse(configsJson) : [['현재 설정', {}]];
const deg = THREE.MathUtils.radToDeg;
console.log = () => {};

for (const [label, options] of configs) {
  const character = loadRig();
  const driver = createLiveAvatarDriver(character, recording.calibration, options);
  if (!driver) { process.stdout.write(`${label}: 캘리브레이션 실패\n`); continue; }
  const bone = name => character.model.getObjectByName('mixamorig' + name);
  const position = o => o.getWorldPosition(new THREE.Vector3());
  const limbs = [['LeftArm', 'LeftForeArm'], ['RightArm', 'RightForeArm'], ['LeftUpLeg', 'LeftLeg'], ['RightUpLeg', 'RightLeg']];
  const start = recording.frames[0].timestamp_ms, end = recording.frames.at(-1).timestamp_ms;
  let index = 0, previous, lean = [], tilt = [], speed = [];
  for (let now = start; now <= end; now += 1000 / 60) {
    while (index + 1 < recording.frames.length && recording.frames[index + 1].timestamp_ms <= now) index++;
    driver.update(recording.frames[index], now, 1 / 60);
    character.model.updateMatrixWorld(true);
    const chest = position(bone('Neck')).sub(position(bone('Hips'))).normalize();
    const hipLine = position(character.bones.rightUpLeg).sub(position(character.bones.leftUpLeg)).normalize();
    lean.push(deg(Math.acos(THREE.MathUtils.clamp(chest.y, -1, 1))));
    tilt.push(Math.abs(deg(Math.asin(THREE.MathUtils.clamp(hipLine.y, -1, 1)))));
    const directions = limbs.map(([a, b]) => position(bone(b)).sub(position(bone(a))).normalize());
    if (previous) speed.push(directions.reduce((sum, d, i) => sum + deg(d.angleTo(previous[i])), 0) / directions.length * 60);
    previous = directions;
  }
  const sorted = values => [...values].sort((a, b) => a - b);
  const p = (values, q) => sorted(values)[Math.floor((values.length - 1) * q)];
  const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
  process.stdout.write(`${label}: 허리 기울기 평균 ${mean(lean).toFixed(1)}° (90% ${p(lean, .9).toFixed(1)}°) | ` +
    `골반 기울기 평균 ${mean(tilt).toFixed(1)}° (90% ${p(tilt, .9).toFixed(1)}°) | ` +
    `팔다리 각속도 중앙값 ${p(speed, .5).toFixed(0)}°/s (낮을수록 덜 떨림; 정지 구간이 많을수록 의미 있음)\n`);
}
