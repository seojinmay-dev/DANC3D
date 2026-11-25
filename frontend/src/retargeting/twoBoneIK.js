import * as THREE from 'three';

// Rotate `bone` (minimal rotation) so that `child` points at `target`.
export function aimBone(bone,child,target) {
  const origin=bone.getWorldPosition(new THREE.Vector3());
  const current=child.getWorldPosition(new THREE.Vector3()).sub(origin);
  const desired=target.clone().sub(origin);
  if(current.lengthSq()<1e-12||desired.lengthSq()<1e-12)return;
  const rotation=new THREE.Quaternion().setFromUnitVectors(current.normalize(),desired.normalize());
  const world=bone.getWorldQuaternion(new THREE.Quaternion()).premultiply(rotation);
  const parent=bone.parent.getWorldQuaternion(new THREE.Quaternion());
  bone.quaternion.copy(parent.invert().multiply(world)).normalize();
  bone.updateWorldMatrix(true,true);
}

// Analytic hip/knee/ankle solve. Bone translations and scales are never stretched.
export function solveTwoBoneIK({upper,lower,foot,target,pole,footRotation}) {
  upper.updateWorldMatrix(true,true);
  const hip=upper.getWorldPosition(new THREE.Vector3());
  const knee=lower.getWorldPosition(new THREE.Vector3());
  const ankle=foot.getWorldPosition(new THREE.Vector3());
  const a=hip.distanceTo(knee),b=knee.distanceTo(ankle);
  if(a<1e-6||b<1e-6)return;
  const direction=target.clone().sub(hip);
  const rawDistance=direction.length();
  if(rawDistance<1e-8)direction.copy(ankle).sub(hip);
  if(direction.lengthSq()<1e-12)direction.set(0,-1,0);
  direction.normalize();
  const distance=THREE.MathUtils.clamp(rawDistance,Math.abs(a-b)+1e-6,a+b-1e-6);
  const bend=knee.clone().sub(hip).addScaledVector(direction,-knee.clone().sub(hip).dot(direction));
  if(bend.length()<Math.min(a,b)*.04) {
    bend.copy(pole).addScaledVector(direction,-pole.dot(direction));
  }
  if(bend.lengthSq()<1e-10) {
    bend.set(1,0,0).addScaledVector(direction,-direction.x);
    if(bend.lengthSq()<1e-10)bend.set(0,0,1).addScaledVector(direction,-direction.z);
  }
  bend.normalize();
  const along=(a*a-b*b+distance*distance)/(2*distance);
  const height=Math.sqrt(Math.max(0,a*a-along*along));
  const desiredKnee=hip.clone().addScaledVector(direction,along).addScaledVector(bend,height);
  const reachableTarget=hip.clone().addScaledVector(direction,distance);
  aimBone(upper,lower,desiredKnee);
  aimBone(lower,foot,reachableTarget);
  if(footRotation) {
    const parent=foot.parent.getWorldQuaternion(new THREE.Quaternion());
    foot.quaternion.copy(parent.invert().multiply(footRotation)).normalize();
    foot.updateWorldMatrix(true,true);
  }
}
