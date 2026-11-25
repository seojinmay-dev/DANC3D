import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {limitBackwardBend} from '../src/retargeting/retarget/spine/spineRetargeter.js';
import {createFullBodyRetargeter} from '../src/retargeting/retarget/fullBodyRetargeter.js';
import {createDanceSyncPose} from '../src/retargeting/pose/danceSyncPose.js';
import {tPose} from './helpers/syntheticDancer.mjs';
import {loadRig} from './helpers/rig.mjs';
const deg=THREE.MathUtils.radToDeg;
test('backward limit is smooth and bounded; forward bends and sideways lean are preserved',()=>{
  const x=new THREE.Vector3(1,0,0);
  for(const angle of [-70,-30,0,3,10,35,70]) {
    const q=new THREE.Quaternion().setFromAxisAngle(x,THREE.MathUtils.degToRad(angle));
    const result=limitBackwardBend(q,8);
    const up=new THREE.Vector3(0,1,0).applyQuaternion(result);
    const pitch=deg(Math.atan2(up.z,up.y));
    if(angle<=4) assert.ok(Math.abs(pitch-angle)<1e-8);
    else assert.ok(pitch<=8 && pitch>0);
    assert.ok(limitBackwardBend(q).angleTo(q)<1e-7);
  }
  const side=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),.4);
  assert.ok(limitBackwardBend(side,8).angleTo(side)<1e-7);
});
test('actual character limits anatomical backward lean and preserves forward bend',async()=>{
  async function chestPitch(offset,limit) {
    const character=await loadRig();
    const r=createFullBodyRetargeter(character,{pelvisPositionMode:'world',smoothing:false,
      body:{spineMaxBackwardDeg:limit}});
    r.calibrate(createDanceSyncPose({joints:tPose()}));
    const joints=tPose();
    for(const side of ['left','right']) for(const part of ['Shoulder','Elbow','Wrist']) joints[side+part].z+=offset;
    r.setPose(createDanceSyncPose({joints}));r.update(1/60);
    const hips=character.model.getObjectByName('mixamorigHips').getWorldPosition(new THREE.Vector3());
    const shoulders=character.bones.leftArm.getWorldPosition(new THREE.Vector3())
      .add(character.bones.rightArm.getWorldPosition(new THREE.Vector3())).multiplyScalar(.5).sub(hips);
    return deg(Math.atan2(shoulders.z,shoulders.y));
  }
  const backward=await chestPitch(-.3,8),unlimited=await chestPitch(-.3,Infinity);
  assert.ok(unlimited < -12, `unlimited ${unlimited}`);
  assert.ok(backward >= -8.1 && backward < 0, `limited ${backward}`);
  assert.ok(Math.abs(await chestPitch(.3,8)-await chestPitch(.3,Infinity))<1e-6);
});
