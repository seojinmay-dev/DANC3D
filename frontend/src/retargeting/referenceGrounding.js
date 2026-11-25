import * as THREE from 'three';
import {findCharacterMotionRoot} from './retarget/core/characterRoot.js';
import {analyzeFootContacts,sampleFootContact} from './footContacts.js';
import {solveTwoBoneIK} from './twoBoneIK.js';
import {getMirroredLeg,findNaturalFloor,translateRoot,rootYaw,fitPelvisToReach,createSoleCorners,liftAboveFloor} from './groundingCommon.js';

export function createReferenceGrounding(character,data,{calibration,floorY=0}={}) {
  const plan=analyzeFootContacts(data);
  const root=findCharacterMotionRoot(character);
  character.model.updateMatrixWorld(true);
  const restRootRotation=root.getWorldQuaternion(new THREE.Quaternion());
  const feet={};
  for(const side of ['left','right'])feet[side]={...getMirroredLeg(character,side),weight:0,anchor:null};
  const characterLegLength=(feet.left.hip.distanceTo(feet.left.ankle)+feet.right.hip.distanceTo(feet.right.ankle))/2;
  const naturalFloor=findNaturalFloor(character,Object.values(feet),characterLegLength);
  const baselineOffset=floorY-naturalFloor;
  // Anchors use the same mapping as the leg IK: a source ankle position relative to the
  // calibration hip, mirrored, scaled by character / dancer leg length, from the character's hip.
  const sourceLeg=['left','right'].reduce((sum,side)=>sum+
    calibration.joints[side+'Hip'].distanceTo(calibration.joints[side+'Knee'])+
    calibration.joints[side+'Knee'].distanceTo(calibration.joints[side+'Ankle']),0)/2;
  const movementScale=(feet.left.length+feet.right.length)/2/sourceLeg;
  for(const state of Object.values(feet)) {
    const sole=state.ankle.clone();sole.y=naturalFloor;
    state.soleLocal=state.foot.worldToLocal(sole.clone());
    state.forward=state.toePosition.clone().sub(state.ankle);state.forward.y=0;state.forward.normalize();
    state.soleCorners=createSoleCorners(state,naturalFloor,characterLegLength*.05);
  }
  let rootCorrection=0;
  const rootShift=new THREE.Vector3();
  function apply(time) {
    // The base playback restores every bone/root first: this correction never accumulates on seek/pause.
    translateRoot(character,root,new THREE.Vector3(0,baselineOffset,0));
    const t=THREE.MathUtils.clamp(time,plan.times[0],plan.times.at(-1));
    const yaw=rootYaw(root,restRootRotation);
    const goals=[];
    for(const [side,state] of Object.entries(feet)) {
      const contact=sampleFootContact(plan,side,t);
      state.weight=contact?.weight??0;state.anchor=null;
      if(state.weight<=0)continue;
      const sourceHip=calibration.joints[side+'Hip'];
      const anchor=state.hip.clone().add(new THREE.Vector3(
        -(contact.anchor[0]-sourceHip.x)*movementScale,0,(contact.anchor[2]-sourceHip.z)*movementScale));anchor.y=floorY;
      state.anchor=anchor;
      // Allow the shoe to pivot with body yaw while keeping its sole on the plane.
      const flatRotation=state.restRotation.clone().premultiply(yaw);
      const offset=state.soleLocal.clone().multiply(state.scale).applyQuaternion(flatRotation);
      const ankleTarget=anchor.clone().sub(offset);
      const current=state.foot.getWorldPosition(new THREE.Vector3());
      const target=current.clone().lerp(ankleTarget,state.weight);
      const rotation=state.foot.getWorldQuaternion(new THREE.Quaternion()).slerp(flatRotation,state.weight);
      const hip=state.upper.getWorldPosition(new THREE.Vector3());
      goals.push({state,target,rotation,pole:state.forward.clone().applyQuaternion(yaw),
        hip,length:state.length});
    }
    // Project the pelvis into both legs' reach. A small horizontal correction avoids dropping
    // the whole body to ankle height when a wide stance is just beyond one leg's reach.
    rootShift.copy(fitPelvisToReach(goals,{iterations:8,maxDrop:characterLegLength*.5,
      maxRise:characterLegLength*.1,maxHorizontal:characterLegLength*.2}));
    rootCorrection=rootShift.y;
    if(rootShift.lengthSq())translateRoot(character,root,rootShift);
    for(const {state,target,rotation,pole} of goals) {
      solveTwoBoneIK({upper:state.upper,lower:state.lower,foot:state.foot,target,pole,footRotation:rotation});
    }
    // A moving/unlocked foot may slide or lift, but must not pass through the floor.
    for(const state of Object.values(feet))liftAboveFloor(state,floorY,state.forward.clone().applyQuaternion(yaw),solveTwoBoneIK);
    character.model.updateMatrixWorld(true);
  }
  return {
    apply,
    getDebugState() {
      return {floorY,baselineOffset,rootCorrection,rootShift:rootShift.toArray(),segments:Object.fromEntries(
        Object.entries(plan.feet).map(([side,foot])=>[side,foot.segments.length])),
      feet:Object.fromEntries(Object.entries(feet).map(([side,state])=>{
        const sole=state.foot.localToWorld(state.soleLocal.clone());
        return [side,{weight:state.weight,sole:sole.toArray(),anchor:state.anchor?.toArray()??null,
          error:state.anchor?sole.distanceTo(state.anchor):null}];
      }))};
    }
  };
}
