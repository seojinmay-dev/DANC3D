import * as THREE from 'three';

const capitalize = side => side[0].toUpperCase() + side.slice(1);

// The retargeter is mirrored: source left drives the character's right leg.
export function getMirroredLeg(character, sourceSide) {
  const side = sourceSide === 'left' ? 'right' : 'left';
  const prefix = 'mixamorig' + capitalize(side);
  const leg = {
    upper: character.bones[side + 'UpLeg'],
    lower: character.bones[side + 'Leg'],
    foot: character.model.getObjectByName(prefix + 'Foot'),
    toe: character.model.getObjectByName(prefix + 'ToeBase')
  };
  if (!Object.values(leg).every(Boolean)) throw Error('발 접지에 필요한 캐릭터 뼈대를 찾지 못했습니다.');
  const position = object => object.getWorldPosition(new THREE.Vector3());
  const hip = position(leg.upper), knee = position(leg.lower);
  const ankle = position(leg.foot), toe = position(leg.toe);
  return {
    ...leg, hip, ankle, toePosition: toe,
    length: hip.distanceTo(knee) + knee.distanceTo(ankle),
    restRotation: leg.foot.getWorldQuaternion(new THREE.Quaternion()),
    scale: leg.foot.getWorldScale(new THREE.Vector3())
  };
}

// Precise skinned bounds include the shoes. Rig-only tests fall back to a toe-based sole plane.
export function findNaturalFloor(character, legs, legLength) {
  const bounds = new THREE.Box3().setFromObject(character.model, true);
  return Number.isFinite(bounds.min.y) ? bounds.min.y :
    Math.min(...legs.map(leg => Math.min(leg.ankle.y, leg.toePosition.y))) - legLength * .025;
}

export function translateRoot(character, root, delta) {
  const position = root.getWorldPosition(new THREE.Vector3()).add(delta);
  root.position.copy(root.parent.worldToLocal(position));
  character.model.updateMatrixWorld(true);
}

// Horizontal (world-up) part of the root's rotation since rest.
export function rootYaw(root, restRotation) {
  const deltaRotation = root.getWorldQuaternion(new THREE.Quaternion()).multiply(restRotation.clone().invert());
  const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(deltaRotation);
  return new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.atan2(forward.x, forward.z));
}

// Heel/toe sole corners (foot-local), built on the rest floor plane.
export function createSoleCorners(leg, floorY, halfWidth) {
  const forward = leg.toePosition.clone().sub(leg.ankle); forward.y = 0; forward.normalize();
  const across = new THREE.Vector3(-forward.z, 0, forward.x).multiplyScalar(halfWidth);
  const heel = leg.ankle.clone(); heel.y = floorY;
  const toe = leg.toePosition.clone(); toe.y = floorY;
  return [heel, toe].flatMap(point => [-1, 1].map(sign =>
    leg.foot.worldToLocal(point.clone().addScaledVector(across, sign))));
}

export function lowestCornerY(leg) {
  return Math.min(...leg.soleCorners.map(point => leg.foot.localToWorld(point.clone()).y));
}

// A foot may slide or lift, but must not pass through the floor.
export function liftAboveFloor(leg, floorY, pole, solveTwoBoneIK) {
  const lowest = lowestCornerY(leg);
  if (lowest >= floorY - .0005) return;
  const target = leg.foot.getWorldPosition(new THREE.Vector3()); target.y += floorY - lowest;
  solveTwoBoneIK({upper: leg.upper, lower: leg.lower, foot: leg.foot, target, pole,
    footRotation: leg.foot.getWorldQuaternion(new THREE.Quaternion())});
}

// Frame-rate independent exponential approach with time constant tau (seconds).
export const approach = (value, target, dt, tau) => target + (value - target) * Math.exp(-Math.max(0, dt) / tau);

// Shift the pelvis until every locked foot target is within its leg's reach,
// with limits on how far it may drop, rise and move sideways.
export function fitPelvisToReach(goals, {iterations, maxDrop, maxRise, maxHorizontal}) {
  const shift = new THREE.Vector3();
  for (let i = 0; i < iterations; i++) {
    for (const {hip, target, length} of goals) {
      const direction = target.clone().sub(hip).sub(shift);
      const distance = direction.length(), reach = length - 1e-5;
      if (distance > reach) shift.addScaledVector(direction, 1 - reach / distance);
      shift.y = THREE.MathUtils.clamp(shift.y, -maxDrop, maxRise);
      const horizontal = Math.hypot(shift.x, shift.z);
      if (horizontal > maxHorizontal) { shift.x *= maxHorizontal / horizontal; shift.z *= maxHorizontal / horizontal; }
    }
  }
  return shift;
}
