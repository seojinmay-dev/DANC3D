import * as THREE from 'three';
import {createAIHubPoseAdapter, alignToDominantFacing, floorAnkleHeight} from './pose/adapters/aiHubPoseAdapter.js';
import {findCharacterMotionRoot} from './retarget/core/characterRoot.js';
import {createReferenceGrounding} from './referenceGrounding.js';

// Stateless sampling: pause, reset, skipped renders and speed use the same content clock.
export function createTransformPlayback({times, nodes, rotations, motionRoot, positions}) {
  const a = new THREE.Quaternion(), b = new THREE.Quaternion();
  const left = new THREE.Vector3(), right = new THREE.Vector3();
  return {
    apply(time) {
      const t = Math.max(times[0], Math.min(Number.isFinite(time) ? time : 0, times.at(-1)));
      let lo = 0, hi = times.length - 1;
      while (lo < hi) {
        const mid = Math.ceil((lo + hi) / 2);
        if (times[mid] <= t) lo = mid; else hi = mid - 1;
      }
      const next = Math.min(lo + 1, times.length - 1);
      const alpha = next === lo ? 0 : (t - times[lo]) / (times[next] - times[lo]);
      for (let n = 0; n < nodes.length; n++) {
        a.fromArray(rotations, (lo * nodes.length + n) * 4);
        b.fromArray(rotations, (next * nodes.length + n) * 4);
        nodes[n].quaternion.slerpQuaternions(a, b, alpha);
      }
      left.fromArray(positions, lo * 3); right.fromArray(positions, next * 3);
      motionRoot.position.lerpVectors(left, right, alpha);
      motionRoot.updateWorldMatrix(true, true);
    }
  };
}

// Bake AI Hub frames once without tracking filters; interpolate rotations at playback time.
// `grounding: false` skips foot contacts (diagnostics only).
export async function bakeReferenceAnimation({data: sourceData, character, retargeter, signal,
  yieldWork = () => new Promise(resolve => setTimeout(resolve, 0)), grounding: useGrounding = true}) {
  if (!sourceData?.frames?.length) throw Error('기준 안무 3D 프레임이 없습니다.');
  let data = sourceData;
  const times = data.frames.map((frame, i) => frame.timestamp ?? i / data.fps);
  if (!times.every((t, i) => Number.isFinite(t) && t >= 0 && (!i || t > times[i - 1]))) {
    throw Error('기준 안무 시간 정보가 올바르지 않습니다.');
  }
  const checkCancelled = () => { if (signal?.aborted) throw new DOMException('Aborted', 'AbortError'); };
  checkCancelled();
  data = alignToDominantFacing(data);
  const adapter = createAIHubPoseAdapter();
  const calibration = adapter.createCalibrationPose(data.frames[0], {ankleHeight: floorAnkleHeight(data)});
  if (!retargeter.calibrate(calibration)) throw Error('기준 안무 보정에 실패했습니다.');
  const grounding = createReferenceGrounding(character, data, {calibration});
  const motionRoot = findCharacterMotionRoot(character);
  const nodes = [motionRoot];
  character.model.traverse(node => { if (node.isBone && node !== motionRoot) nodes.push(node); });
  const rotations = new Float32Array(times.length * nodes.length * 4);
  const positions = new Float32Array(times.length * 3);
  let chunkStarted = performance.now();
  for (let i = 0; i < data.frames.length; i++) {
    checkCancelled();
    const pose = adapter.convert(data.frames[i]);
    if (!pose) throw Error(`기준 안무 ${i} 프레임의 관절 좌표가 없습니다.`);
    // Resolve child targets again after parent rotations reach this frame's pose.
    retargeter.setPose(pose); retargeter.update();
    retargeter.setPose(pose); retargeter.update();
    nodes.forEach((node, n) => node.quaternion.toArray(rotations, (i * nodes.length + n) * 4));
    motionRoot.position.toArray(positions, i * 3);
    if (performance.now() - chunkStarted > 8) {
      await yieldWork(); checkCancelled(); chunkStarted = performance.now();
    }
  }
  const base = createTransformPlayback({times, nodes, rotations, motionRoot, positions});
  const playback = {
    apply(time) { base.apply(time); if (useGrounding) grounding.apply(Number.isFinite(time) ? time : 0); },
    getGroundingState: () => grounding.getDebugState()
  };
  playback.apply(0);
  return playback;
}
