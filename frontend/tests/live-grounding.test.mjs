import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {createLiveFootContacts,createLiveGrounding} from '../src/retargeting/liveGrounding.js';
import {loadRig} from './helpers/rig.mjs';
import {findCharacterMotionRoot} from '../src/retargeting/retarget/core/characterRoot.js';
function pose() {
  const screenJoints={},joints={};
  for(const side of ['left','right']) for(const [part,y] of [['Hip',.4],['Knee',.6],['Ankle',.8]]) {
    screenJoints[side+part]=new THREE.Vector2(side==='left'?.4:.6,y);
    joints[side+part]=new THREE.Vector3(0,y,0);
  }
  return {screenJoints,joints,metadata:{visibility:{},trackingParts:{}}};
}
test('contact keeps squats, releases lifts, jumps, occlusion and stale samples',()=>{
  const p=pose(), c=createLiveFootContacts(p);
  for(let t=0;t<=300;t+=20)c.observe(p,t);
  assert.equal(c.isContact('left'),true);
  p.joints.leftAnkle.y-=.3;p.screenJoints.leftHip.y+=.2;c.observe(p,320);
  assert.equal(c.isContact('left'),true);
  p.screenJoints.leftAnkle.y-=.12;c.observe(p,340);
  assert.equal(c.isContact('left'),false);assert.equal(c.isContact('right'),true);
  p.screenJoints.rightAnkle.y-=.12;c.observe(p,360);assert.equal(c.isContact('right'),false);
  p.screenJoints.leftAnkle.y=.8;p.screenJoints.rightAnkle.y=.8;
  for(let t=380;t<=700;t+=20)c.observe(p,t);
  assert.equal(c.isContact('left'),true);
  p.metadata.trackingParts.leftLeg=false;c.observe(p,720);assert.equal(c.isContact('left'),false);
  assert.equal(c.isFresh(1000),false);c.observe(p,1100);assert.equal(c.isContact('right'),false);
});
test('rig contact stays on floor without accumulating IK or mutating source',async()=>{
  const character=await loadRig(),p=pose(),snapshot=JSON.stringify(p);
  const g=createLiveGrounding(character,p);
  for(let t=0;t<=400;t+=20)g.observe(p,t);
  const nodes=Object.values(character.bones),positions=nodes.map(b=>b.position.toArray());
  for(let i=0;i<120;i++){g.beforeRetarget();g.apply(400,1/60);}
  const first=g.getDebugState();
  for(let i=0;i<120;i++){g.beforeRetarget();g.apply(400,1/60);}
  const second=g.getDebugState();
  for(let i=0;i<2;i++) {
    assert.ok(Math.abs(second[i].sole[1])<.01);
    assert.ok(new THREE.Vector3().fromArray(first[i].sole).distanceTo(new THREE.Vector3().fromArray(second[i].sole))<1e-7);
  }
  assert.deepEqual(nodes.map(b=>b.position.toArray()),positions);assert.equal(JSON.stringify(p),snapshot);
  g.loseTracking();g.beforeRetarget();g.apply(1000,1/60);assert.ok(g.getDebugState().every(f=>f.weight===0));
});

test('contact survives landmark jitter and follows a floor line that moves with the user',()=>{
  const p=pose(), c=createLiveFootContacts(p);
  let seed=3;const jitter=()=>((seed=(seed*16807)%2147483647)/2147483647-.5)*.008;
  for(let t=0;t<=600;t+=33){
    for(const side of ['left','right'])p.screenJoints[side+'Ankle'].y=.8+jitter();
    c.observe(p,t);
  }
  assert.equal(c.isContact('left'),true);assert.equal(c.isContact('right'),true);
  // Walking towards the camera moves the feet down the image; contact must hold.
  for(let t=633,y=.8;t<=3000;t+=33,y=Math.min(.86,y+.0015)){
    for(const side of ['left','right'])p.screenJoints[side+'Ankle'].y=y;
    c.observe(p,t);assert.equal(c.isContact('left'),true);
  }
  assert.equal(c.isAirborne(),false);
  p.screenJoints.leftAnkle.y-=.12;c.observe(p,3033);
  assert.equal(c.isContact('left'),false);assert.equal(c.isAirborne(),false);
  p.screenJoints.rightAnkle.y-=.12;c.observe(p,3066);assert.equal(c.isAirborne(),true);
});

test('lowest sole rests on the floor however the screen hip moves; only a jump lifts the body',()=>{
  const character=loadRig(),p=pose(),root=findCharacterMotionRoot(character);
  const g=createLiveGrounding(character,p);
  const lowest=()=>Math.min(...g.getDebugState().map(f=>f.sole[1]));
  // Like the retargeter, set an absolute root height before grounding.
  const base=root.position.y;
  const frame=(t,rise)=>{g.observe(p,t);g.beforeRetarget();root.position.y=base+rise;g.apply(t,1/60);};
  // Screen-space float (e.g. stepping back from the camera) must not lift the feet.
  for(let t=0;t<=500;t+=16)frame(t,.2);
  assert.ok(Math.abs(lowest())<.005,`float ${lowest()}`);
  // Sinking is not allowed either: the feet never pass through the floor.
  for(let t=516;t<=900;t+=16)frame(t,-.2);
  assert.ok(Math.abs(lowest())<.005,`sink ${lowest()}`);
  // Both feet lifted on screen: the body follows the screen height into the air.
  for(const side of ['left','right'])p.screenJoints[side+'Ankle'].y-=.15;
  for(let t=916;t<=1100;t+=16)frame(t,.2);
  assert.ok(lowest()>.05,`jump ${lowest()}`);
  // Releasing a contact is eased instead of popping.
  for(const side of ['left','right'])p.screenJoints[side+'Ankle'].y+=.15;
  for(let t=1116;t<=1600;t+=16)frame(t,0);
  assert.ok(g.getDebugState().every(f=>f.weight>.9));
  p.screenJoints.leftAnkle.y-=.15;frame(1616,0);
  const released=g.getDebugState().find(f=>f.side==='left').weight;
  assert.ok(released>0&&released<.9,`release ${released}`);
});
