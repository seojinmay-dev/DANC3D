// Same dance → reference avatar and live avatar should match.
// One synthetic 3D dance drives both paths (AI Hub bake / webcam filter → predict → retarget → ground).
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync, readdirSync} from 'node:fs';
import {compareRetargeting} from './helpers/compareRetargeting.mjs';
import {alignToDominantFacing} from '../src/retargeting/pose/adapters/aiHubPoseAdapter.js';

const clean = await compareRetargeting();
const noisy = await compareRetargeting({noise: .004});
const segment = (result, name) => result.segments.find(s => s.name === name);

function report(t, result) {
  for (const s of result.segments) {
    t.diagnostic(`${s.name}: ` + Object.entries(s.bones).map(([bone, e]) =>
      `${bone} ${e.mean.toFixed(1)}°/${e.max.toFixed(0)}°`).join(', ') +
      ` | 루트 수평 ${s.root.horizontal.toFixed(3)} 수직 ${s.root.vertical.toFixed(3)} yaw ${s.root.yaw.toFixed(1)}°`);
  }
}

test('limbs of both avatars agree for the same dance (mean / max degrees per segment)', t => {
  report(t, clean);
  const limits = {
    'T포즈 유지': {'허벅지': [3.5, 4], '정강이': [3, 4]},
    '팔 동작': {'팔(상완)': [4.5, 14], '팔(전완)': [4.5, 15]},
    '스쿼트': {'허벅지': [4, 16], '정강이': [3.5, 13]},
    '무릎 들기': {'허벅지': [5.5, 28], '정강이': [3, 8]},
    '몸 회전': {'팔(상완)': [4.5, 12], '팔(전완)': [2, 6], '허벅지': [3, 6], '정강이': [3, 7]},
    '체중이동+앞뒤 이동': {'허벅지': [5, 9], '정강이': [6, 12]},
    '팔 앞으로 뻗기': {'팔(상완)': [2, 12], '팔(전완)': [1, 8]},
    '느린 팔 동작': {'팔(상완)': [2.5, 5], '팔(전완)': [1, 3]}
  };
  for (const [name, bones] of Object.entries(limits)) {
    for (const [bone, [mean, max]] of Object.entries(bones)) {
      const e = segment(clean, name).bones[bone];
      assert.ok(e.mean <= mean && e.max <= max, `${name} ${bone}: ${e.mean.toFixed(1)}°/${e.max.toFixed(0)}° > ${mean}°/${max}°`);
    }
  }
});

test('both avatars move their body the same way (root, in leg lengths)', () => {
  for (const s of clean.segments.filter(s => s.name !== '체중이동+앞뒤 이동')) {
    assert.ok(s.root.horizontal < .02, `${s.name} horizontal ${s.root.horizontal}`);
  }
  for (const s of clean.segments) assert.ok(s.root.vertical < .06, `${s.name} vertical ${s.root.vertical}`);
  assert.ok(segment(clean, '몸 회전').root.yaw < 13, `${segment(clean, '몸 회전').root.yaw}`);
});

test('live avatar latency stays low without excessive jitter', t => {
  t.diagnostic(`지연: 빠른 동작 ${clean.latencyMs.toFixed(0)}ms, 느린 동작 ${clean.slowLatencyMs.toFixed(0)}ms ` +
    `(카메라 촬영·MediaPipe 추론 시간 제외) | 노이즈 시 정지 떨림 ${noisy.jitterDegPerSec.toFixed(1)}°/s`);
  assert.ok(clean.latencyMs <= 15, `fast latency ${clean.latencyMs}ms`);
  assert.ok(clean.slowLatencyMs <= 40, `slow latency ${clean.slowLatencyMs}ms`);
  assert.ok(noisy.jitterDegPerSec <= 60, `jitter ${noisy.jitterDegPerSec}°/s`);
  assert.ok(noisy.errorAtZeroLag <= 4.2, `noisy mean error ${noisy.errorAtZeroLag}°`);
});

test('live avatar holds still under MediaPipe-like noise (no leaning waist, no wobbling hips)', async t => {
  // Depth 3× and hips 2× noisier than other landmarks, plus temporally correlated wobble.
  const result = await compareRetargeting({noise: .004, realisticNoise: true});
  const {lean, hipTilt} = result.waist;
  t.diagnostic(`정지 시 허리 기울기 평균 ${lean.mean.toFixed(1)}° / 최대 ${lean.max.toFixed(1)}°, ` +
    `골반 기울기 평균 ${hipTilt.mean.toFixed(1)}° / 최대 ${hipTilt.max.toFixed(1)}°, 떨림 ${result.jitterDegPerSec.toFixed(0)}°/s`);
  assert.ok(lean.mean < 1.5 && lean.max < 4, `lean ${lean.mean}/${lean.max}`);
  assert.ok(hipTilt.mean < 1.2 && hipTilt.max < 6, `hip tilt ${hipTilt.mean}/${hipTilt.max}`);
  assert.ok(result.jitterDegPerSec < 100, `jitter ${result.jitterDegPerSec}`);
});

test('a torso depth misjudged by MediaPipe does not tilt the whole avatar', async t => {
  // Gravity comes from the calibration legs, not the (often misjudged) torso.
  for (const bias of [.06, -.06]) {
    const result = await compareRetargeting({shoulderDepthBias: bias});
    t.diagnostic(`어깨 깊이 편향 ${bias}m: 정지 시 몸 전체 기울기 ${result.waist.bodyLean.mean.toFixed(1)}°`);
    assert.ok(result.waist.bodyLean.mean < 1.5, `bias ${bias}: ${result.waist.bodyLean.mean}°`);
    assert.ok(Math.abs(result.waist.pitchMean - result.waist.referencePitchMean) < 1.5, `pitch ${result.waist.pitchMean}`);
  }
});

test('an imperfect user T-pose shifts the arms by about the calibration error', async t => {
  const drop = 10 * Math.PI / 180;
  const result = await compareRetargeting({calibrationArmDrop: drop});
  const arms = segment(result, '느린 팔 동작').bones['팔(상완)'].mean;
  t.diagnostic(`T포즈에서 팔이 10° 처지면 상완 평균 차이 ${arms.toFixed(1)}°`);
  assert.ok(arms > 5 && arms < 15, `${arms}`);
});

// ----- Known gaps (todo: reported, do not fail the suite) -----

test('live avatar follows forward/backward travel like the reference', {todo: '웹캠 경로는 앞뒤(Z) 이동을 추정하지 않음'}, () => {
  assert.ok(segment(clean, '체중이동+앞뒤 이동').root.horizontal < .05);
});

test('live avatar squats as deep as the reference (leg IK, planted feet stay flat)', () => {
  const squat = segment(clean, '스쿼트');
  assert.ok(squat.root.vertical < .06, `${squat.root.vertical}`);
  assert.ok(squat.bones['허벅지'].mean < 4, `${squat.bones['허벅지'].mean}`);
});

test('reference avatars face the front for most of each song, whatever the first frame faces', () => {
  const directory = new URL('../public/references/', import.meta.url);
  const yawOf = j => Math.atan2(j.rightShoulder[2] - j.leftShoulder[2] + j.rightHip[2] - j.leftHip[2],
    j.rightShoulder[0] - j.leftShoulder[0] + j.rightHip[0] - j.leftHip[0]);
  const facingAway = {};
  for (const file of readdirSync(directory).filter(f => f.endsWith('.json'))) {
    const {frames} = alignToDominantFacing(JSON.parse(readFileSync(new URL(file, directory))));
    // Aligned front: the left → right axis points to -X (yaw ±180°).
    const away = frames.filter(f => Math.abs(Math.atan2(Math.sin(yawOf(f.joints) - Math.PI), Math.cos(yawOf(f.joints) - Math.PI))) > Math.PI / 2).length / frames.length;
    if (away > .15) facingAway[file] = Math.round(away * 100) + '%';
  }
  assert.deepEqual(facingAway, {});
});
