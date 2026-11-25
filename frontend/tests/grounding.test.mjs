import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import * as THREE from 'three';
import {analyzeFootContacts,sampleFootContact} from '../src/retargeting/footContacts.js';
import {solveTwoBoneIK} from '../src/retargeting/twoBoneIK.js';
import {bakeReferenceAnimation} from '../src/retargeting/referenceAnimation.js';
import {createFullBodyRetargeter} from '../src/retargeting/retarget/fullBodyRetargeter.js';
import {loadRig} from './helpers/rig.mjs';

function sequence(kind='standing') {
  const frames=Array.from({length:90},(_,i)=>{
    const time=i/30,joints={};
    for(const [side,sign]of [['left',-1],['right',1]]) {
      joints[side+'Hip']=[sign*.1,1,0];joints[side+'Knee']=[sign*.1,.55,0];joints[side+'Ankle']=[sign*.1,.1,0];
      joints[side+'Shoulder']=[sign*.2,1.5,0];joints[side+'Elbow']=[sign*.5,1.5,0];joints[side+'Wrist']=[sign*.8,1.5,0];
    }
    if(kind==='step'&&time>=.8&&time<=1.2)joints.leftAnkle[1]+=.25;
    if(kind==='jump'&&time>=.8&&time<=1.2)for(const p of Object.values(joints))p[1]+=.3;
    if(kind==='noise')for(const side of ['left','right'])joints[side+'Ankle'][0]+=.001*Math.sin(i);
    return {timestamp:time,frameIndex:i,joints,rootScreenPosition:[.5,.5]};
  });
  return {fps:30,frames};
}

async function bake(data,{geometry=false}={}) {
  const character=loadRig({geometry}),log=console.log;console.log=()=>{};
  try {
    const retargeter=createFullBodyRetargeter(character,{pelvisPositionMode:'world',smoothing:false});
    return {character,playback:await bakeReferenceAnimation({data,character,retargeter})};
  } finally {console.log=log;}
}

test('source contacts distinguish a planted foot, a lifted foot and a jump',()=>{
  const step=analyzeFootContacts(sequence('step'));
  assert.equal(sampleFootContact(step,'left',1),null);
  assert.equal(sampleFootContact(step,'right',1).weight,1);
  const jump=analyzeFootContacts(sequence('jump'));
  assert.equal(sampleFootContact(jump,'left',1),null);assert.equal(sampleFootContact(jump,'right',1),null);
  assert.ok(sampleFootContact(jump,'left',1.6).weight>.99);
  const noise=analyzeFootContacts(sequence('noise'));
  assert.equal(noise.feet.left.segments.length,1);assert.equal(noise.feet.right.segments.length,1);
});

test('contact fades around lift/landing and does not invent a floor for an always-raised foot',()=>{
  const data=sequence('standing');
  for(const frame of data.frames)frame.joints.leftAnkle[1]+=.4;
  const plan=analyzeFootContacts(data);assert.equal(plan.feet.left.segments.length,0);
  assert.equal(sampleFootContact(plan,'right',1).weight,1);
  const step=analyzeFootContacts(sequence('step'));
  const segment=step.feet.left.segments[0];
  assert.equal(sampleFootContact(step,'left',segment.end).weight,0);
  const last=step.feet.left.segments.at(-1);
  assert.equal(sampleFootContact(step,'left',last.start).weight,0);
  assert.ok(sampleFootContact(step,'left',last.start+.04).weight<1);
});

test('two-bone IK reaches a world-space goal under a scaled/rotated parent without stretching bones',()=>{
  const parent=new THREE.Group();parent.position.set(1,2,3);parent.scale.setScalar(2);parent.rotation.y=.7;
  const upper=new THREE.Bone(),lower=new THREE.Bone(),foot=new THREE.Bone();
  upper.position.set(0,1,0);lower.position.set(0,-.45,0);foot.position.set(0,-.45,0);
  parent.add(upper);upper.add(lower);lower.add(foot);parent.updateMatrixWorld(true);
  const hip=upper.getWorldPosition(new THREE.Vector3());
  const goal=hip.clone().add(new THREE.Vector3(.2,-1.5,.3));
  const rotation=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),.2);
  solveTwoBoneIK({upper,lower,foot,target:goal,pole:new THREE.Vector3(0,0,1),footRotation:rotation});
  assert.ok(foot.getWorldPosition(new THREE.Vector3()).distanceTo(goal)<1e-6);
  assert.ok(foot.getWorldQuaternion(new THREE.Quaternion()).angleTo(rotation)<1e-6);
  assert.deepEqual(lower.position.toArray(),[0,-.45,0]);assert.deepEqual(foot.position.toArray(),[0,-.45,0]);
  solveTwoBoneIK({upper,lower,foot,target:hip.clone().addScalar(10),pole:new THREE.Vector3(0,0,1)});
  assert.ok(foot.getWorldPosition(new THREE.Vector3()).distanceTo(hip)<1.80001);
  assert.ok(foot.quaternion.toArray().every(Number.isFinite));
});

test('actual rig locks planted soles despite source jitter, without accumulating corrections on seek',async()=>{
  const {playback,character}=await bake(sequence('noise'));
  let anchor;
  for(const time of [0,.3,.6333,1,1.5,2.2]) {
    playback.apply(time);const state=playback.getGroundingState();
    for(const foot of Object.values(state.feet)) {
      assert.equal(foot.weight,1);assert.ok(Math.abs(foot.sole[1])<1e-4);assert.ok(foot.error<1e-4);
    }
    anchor??=state.feet.left.sole;
    assert.ok(new THREE.Vector3().fromArray(state.feet.left.sole).distanceTo(new THREE.Vector3().fromArray(anchor))<1e-4);
  }
  const snapshot=()=>{const result=[];character.model.traverse(node=>result.push(...node.quaternion.toArray(),...node.position.toArray()));return result;};
  playback.apply(.6333);const before=snapshot();playback.apply(2);playback.apply(0);playback.apply(.6333);
  assert.deepEqual(snapshot(),before);
});

test('airborne feet are released and keep jump height; landing reestablishes contact',async()=>{
  const {playback}=await bake(sequence('jump'));
  playback.apply(1);const jump=playback.getGroundingState();
  for(const foot of Object.values(jump.feet)) {
    assert.equal(foot.weight,0);assert.equal(foot.anchor,null);assert.ok(foot.sole[1]>.2);
  }
  playback.apply(1.8);
  for(const foot of Object.values(playback.getGroundingState().feet)) {
    assert.equal(foot.weight,1);assert.ok(foot.error<1e-4);
  }
});

test('actual reference: planted sole error remains below 1cm at frames and between frames',async t=>{
  const data=JSON.parse(readFileSync(new URL('../public/references/033_D00_001_F.json',import.meta.url)));
  const {playback}=await bake(data);
  let planted=0,maxError=0;
  for(let i=0;i<data.frames.length;i++) {
    const time=data.frames[i].timestamp;
    for(const sample of [time,time+1/60]) {
      playback.apply(sample);
      for(const foot of Object.values(playback.getGroundingState().feet)) {
        assert.ok(foot.sole.every(Number.isFinite));
        if(foot.weight>.999){planted++;maxError=Math.max(maxError,foot.error);}
      }
    }
  }
  assert.ok(planted>1000);assert.ok(maxError<.01,`max planted sole error ${maxError}m`);
  t.diagnostic(`${planted} fully planted samples; maximum sole error ${(maxError*1000).toFixed(3)}mm (rig markers)`);
});

test('actual skinned shoe vertices of a planted foot stay within 1cm of the floor',async()=>{
  const data=JSON.parse(readFileSync(new URL('../public/references/033_D00_001_F.json',import.meta.url)));
  const {character,playback}=await bake(data,{geometry:true});
  const meshes=[];character.model.traverse(node=>{if(node.isSkinnedMesh)meshes.push(node);});
  const samples={};
  for(const sourceSide of ['left','right']) {
    const characterSide=sourceSide==='left'?'Right':'Left';
    samples[sourceSide]=meshes.map(mesh=>{
      const joints=new Set(mesh.skeleton.bones.map((bone,i)=>({bone,i})).filter(({bone})=>
        bone.name.startsWith('mixamorig'+characterSide+'Foot')||bone.name.startsWith('mixamorig'+characterSide+'Toe')).map(({i})=>i));
      const ids=mesh.geometry.getAttribute('skinIndex'),weights=mesh.geometry.getAttribute('skinWeight'),vertices=[];
      for(let i=0;i<ids.count;i++) {
        let influence=0;for(let j=0;j<4;j++)if(joints.has(ids.getComponent(i,j)))influence+=weights.getComponent(i,j);
        if(influence>.8)vertices.push(i);
      }
      return {mesh,vertices};
    });
  }
  let checked=0;
  for(const time of [0,10,20,40]) {
    playback.apply(time);const state=playback.getGroundingState();
    for(const side of ['left','right'])if(state.feet[side].weight>.999) {
      const heights=samples[side].flatMap(({mesh,vertices})=>vertices.map(i=>
        mesh.getVertexPosition(i,new THREE.Vector3()).applyMatrix4(mesh.matrixWorld).y));
      assert.ok(heights.length>100);assert.ok(Math.abs(Math.min(...heights))<.01);checked++;
    }
  }
  assert.ok(checked>=4);
});
