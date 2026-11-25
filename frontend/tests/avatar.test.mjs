import {loadRig} from './helpers/rig.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import * as THREE from 'three';
import {createCalibrationCapture, calibrationPose} from '../src/services/avatarCalibration.js';
import {createMediaPipePoseAdapter} from '../src/retargeting/pose/adapters/mediaPipePoseAdapter.js';
import {createTransformPlayback, bakeReferenceAnimation} from '../src/retargeting/referenceAnimation.js';
import {createFullBodyRetargeter} from '../src/retargeting/retarget/fullBodyRetargeter.js';
import {createContentClock} from '../src/services/contentClock.js';

const accepted = {ready:true, full_body_visible:true, arms_horizontal:true, arms_extended:true, arms_spread:true};
function frame(timestamp, offset = 0) {
  const image = Array.from({length:33}, () => ({x:0.5, y:0.5, z:0, visibility:1}));
  const world = Array.from({length:33}, () => [0, 0, 0]);
  for (const [i, x, y] of [[11,-.2,-.5],[12,.2,-.5],[13,-.5,-.5],[14,.5,-.5],
    [15,-.8,-.5],[16,.8,-.5],[23,-.1,0],[24,.1,0],[25,-.1,.4],[26,.1,.4],[27,-.1,.8],[28,.1,.8]]) {
    world[i] = [x + offset, y, 0]; image[i] = {x:.5+x*.4+offset, y:.5+y*.4, z:0, visibility:1};
  }
  return {timestamp_ms:timestamp, image_landmarks:image, world_landmarks:world};
}

test('calibration averages distinct accepted T frames and survives later camera changes', () => {
  const capture = createCalibrationCapture();
  assert.equal(capture.accept(frame(0), accepted), null);
  assert.equal(capture.accept(frame(0), accepted), null);
  assert.equal(capture.accept(frame(200,.001), accepted), null);
  const last = frame(400,.002), saved = capture.accept(last, accepted);
  assert.equal(saved.sample_count, 3);
  assert.ok(Math.abs(saved.world_landmarks[11][0] - (-.199)) < 1e-10);
  last.world_landmarks[11][0] = 999;
  const pose = calibrationPose(createMediaPipePoseAdapter(), saved);
  assert.ok(Math.abs(pose.joints.leftShoulder.x - (-.199)) < 1e-10);
  assert.equal(calibrationPose(createMediaPipePoseAdapter(), null), null);
});

test('failed readiness, gaps, missing world and moving poses cannot complete calibration', () => {
  const capture = createCalibrationCapture();
  capture.accept(frame(0), accepted); capture.accept(frame(200), accepted);
  assert.equal(capture.accept(frame(400), {...accepted, ready:false}), null);
  assert.equal(capture.accept(frame(600), accepted), null);
  assert.equal(capture.accept(frame(1600), accepted), null);
  assert.equal(capture.accept({...frame(1800), world_landmarks:null}, accepted), null);
  capture.accept(frame(2000), accepted); capture.accept(frame(2200), accepted);
  assert.equal(capture.accept(frame(2400,.3), accepted), null);
  capture.reset(); assert.equal(capture.accept(frame(2600), accepted), null);
});

test('reference interpolates rotations/root at content time, independent of render history and speed', () => {
  const root = new THREE.Group(), bone = new THREE.Bone(); root.add(bone);
  const end = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1), Math.PI/2);
  const playback = createTransformPlayback({times:[0,1], nodes:[bone], motionRoot:root,
    rotations:new Float32Array([0,0,0,1,...end.toArray()]), positions:new Float32Array([0,0,0,2,0,0])});
  playback.apply(.5);
  assert.ok(bone.quaternion.angleTo(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),Math.PI/4)) < 1e-3);
  assert.equal(root.position.x,1);
  const snapshot = bone.quaternion.toArray();
  playback.apply(1); playback.apply(0); playback.apply(.5);
  assert.deepEqual(bone.quaternion.toArray(), snapshot);
  for (const speed of [.5,.75,1,1.25,1.5]) {
    let now = 0; const clock = createContentClock(speed,2,()=>now); clock.start(); now=400;
    playback.apply(clock.read()); assert.ok(Math.abs(root.position.x-.8*speed) < 1e-8);
    clock.pause(); now+=1000; playback.apply(clock.read()); assert.ok(Math.abs(root.position.x-.8*speed) < 1e-8);
    clock.reset(); playback.apply(clock.read()); assert.equal(root.position.x,0);
  }
  playback.apply(99); assert.equal(root.position.x,2);
});

test('actual reference and GLB rig bake to finite poses; reset reproduces same bones and root', async t => {
  const data = JSON.parse(readFileSync(new URL('../public/references/033_D00_001_F.json',import.meta.url)));
  const character = loadRig();
  const log = console.log; console.log = ()=>{};
  let playback;
  try {
    const retargeter = createFullBodyRetargeter(character,{pelvisPositionMode:'world',smoothing:false});
    playback = await bakeReferenceAnimation({data,character,retargeter});
  } finally { console.log=log; }
  const snapshot = ()=>{const values=[];character.model.traverse(node=>values.push(...node.position.toArray(),...node.quaternion.toArray()));return values;};
  playback.apply(10.123); const first=snapshot(); assert.ok(first.every(Number.isFinite));
  playback.apply(90); playback.apply(0); playback.apply(10.123); assert.deepEqual(snapshot(),first);
  playback.apply(data.frames.at(-1).timestamp);assert.ok(snapshot().every(Number.isFinite));
  t.diagnostic(`Baked ${data.frames.length} AI Hub frames using the actual character skeleton`);
});

test('reference bake rejects cancellation and invalid timeline', async () => {
  const controller = new AbortController(); controller.abort();
  await assert.rejects(bakeReferenceAnimation({data:{frames:[{timestamp:0}]},signal:controller.signal}), {name:'AbortError'});
  await assert.rejects(bakeReferenceAnimation({data:{frames:[{timestamp:1},{timestamp:0}]}}), /시간 정보/);
});

test('actual user rig uses elapsed time and holds an occluded arm target', t => {
  const snapshots=[];
  const log=console.log;console.log=()=>{};
  try {
    for(const fps of [30,60,120]) {
      const character=loadRig(), adapter=createMediaPipePoseAdapter();
      const retargeter=createFullBodyRetargeter(character);
      assert.ok(retargeter.calibrate(calibrationPose(adapter,frame(0))));
      const moved=frame(100,.05);
      moved.world_landmarks[13]=[-.15,-.8,0];moved.world_landmarks[15]=[-.15,-1.1,0];
      const pose=calibrationPose(adapter,moved);
      retargeter.setPose(pose);
      for(let i=0;i<fps/5;i++)retargeter.update(1/fps);
      snapshots.push({root:retargeter.getDebugState().pelvis.actualX,
        arm:character.bones.rightArm.quaternion.clone(),forearm:character.bones.rightForeArm.quaternion.clone(),
        leg:character.bones.rightUpLeg.quaternion.clone(),knee:character.bones.rightLeg.quaternion.clone()});
      const before=retargeter.getDebugState().arms.userLeft.targetQuaternion;
      const occluded=calibrationPose(adapter,frame(200));
      occluded.metadata.trackingParts={leftArm:false};
      retargeter.setPose(occluded);retargeter.update(1/fps);
      assert.deepEqual(retargeter.getDebugState().arms.userLeft.targetQuaternion,before);
    }
  } finally {console.log=log;}
  for(const snapshot of snapshots.slice(1)) {
    assert.ok(Math.abs(snapshot.root-snapshots[0].root)<1e-6);
    assert.ok(snapshot.arm.angleTo(snapshots[0].arm)<1e-4);
    assert.ok(snapshot.leg.angleTo(snapshots[0].leg)<1e-4);
    assert.ok(snapshot.forearm.angleTo(snapshots[0].forearm)<.015);
    assert.ok(snapshot.knee.angleTo(snapshots[0].knee)<1e-4);
  }
  t.diagnostic('Actual GLB root, upper arm and thigh agree at 30/60/120Hz after 200ms');
});

test('webcam root motion is scaled by the screen leg length and image aspect ratio', () => {
  const log=console.log;console.log=()=>{};
  try {
    const character=loadRig(), adapter=createMediaPipePoseAdapter();
    const retargeter=createFullBodyRetargeter(character,{smoothing:false});
    const calibration={...frame(0),image_width:1280,image_height:720};
    assert.ok(retargeter.calibrate(calibrationPose(adapter,calibration)));
    character.model.updateMatrixWorld(true);
    const legOf=side=>{const p=n=>character.model.getObjectByName('mixamorig'+n).getWorldPosition(new THREE.Vector3());
      return p(side+'UpLeg').distanceTo(p(side+'Foot'));};
    const characterLeg=(legOf('Left')+legOf('Right'))/2;
    // frame(): screen leg = 0.8 world units * 0.4 = 0.32 image heights; shift the image by 0.1 widths.
    const moved=frame(100);for(const p of moved.image_landmarks)p.x+=.1;
    retargeter.setPose(calibrationPose(adapter,moved));retargeter.update(1/60);
    const expected=-.1*(1280/720)*characterLeg/.32;
    const actual=retargeter.getDebugState().pelvis.actualX;
    assert.ok(Math.abs(actual-expected)<Math.abs(expected)*.02,`${actual} vs ${expected}`);
  } finally {console.log=log;}
});
