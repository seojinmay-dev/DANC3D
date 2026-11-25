import * as THREE from 'three';
import {trackingAlpha} from '../core/trackingSmoothing.js';
import {calibrationUp, characterPelvisQuaternion, frameQuaternion, pelvisFrame
} from '../core/frameDelta.js';
import {aimBone} from '../../twoBoneIK.js';

// Near full extension the knee angle is extremely sensitive to the hip–ankle distance, so
// landmark noise would flicker the knee. Ease the reach (in leg lengths) into straight over
// the last EXTENSION_EASE: zero slope at straight, continuous slope where the ease begins.
export const LEG_IK = {extensionEase: .07};
function easeReach(reach) {
  const ease = LEG_IK.extensionEase;
  if (!(ease > 0) || reach <= 1 - ease) return reach;
  const x = Math.max(0, (1 - reach) / ease);
  return 1 - ease * (2 * x * x - x * x * x);
}
const WORLD_UP = new THREE.Vector3(0, 1, 0);

// Legs by two-bone IK in the pelvis frame:
//  - ankle target: the user's hip → ankle vector, normalized by the user's leg length,
//    scaled to the character's leg (so squat depth and stance width follow the user);
//  - knee direction: the user's knee offset from the hip–ankle line, which also sets the
//    thigh twist (turned-out knees, pliés).
// Mirrored: the user's left leg drives the character's right leg.
export function createLegRetargeter(character, {smoothing = true, rootDelta} = {}) {
  const position = object => object.getWorldPosition(new THREE.Vector3());
  character.model.updateMatrixWorld(true);
  const legs = [
    createLeg('left', 'Right', character.bones.rightUpLeg, character.bones.rightLeg),
    createLeg('right', 'Left', character.bones.leftUpLeg, character.bones.leftLeg)
  ];
  const characterPelvis = characterPelvisQuaternion(character);
  let worldUp, calibrated = false;

  function createLeg(userSide, characterSide, upper, lower) {
    const foot = character.model.getObjectByName(`mixamorig${characterSide}Foot`);
    const toe = foot?.children.find(child => child.name.includes('ToeBase'));
    if (!foot || !toe) throw new Error('Foot/ToeBase bone을 찾지 못했습니다.');
    const hip = position(upper), knee = position(lower), ankle = position(foot);
    // The knee faces where the toes point in the bind pose; kept in thigh-local space.
    const forward = position(toe).sub(ankle); forward.y = 0; forward.normalize();
    return {
      userSide, upper, lower, foot,
      thigh: hip.distanceTo(knee), shin: knee.distanceTo(ankle),
      rest: [upper, lower, foot].map(bone => bone.quaternion.clone()),
      kneeFrontLocal: forward.applyQuaternion(upper.getWorldQuaternion(new THREE.Quaternion()).invert()),
      footUpLocal: WORLD_UP.clone().applyQuaternion(foot.getWorldQuaternion(new THREE.Quaternion()).invert()),
      ankleTarget: new THREE.Vector3(0, -1, 0), poleTarget: new THREE.Vector3(),
      ankle: new THREE.Vector3(0, -1, 0), pole: new THREE.Vector3(),
      userLength: 1, targetFlat: 1, flat: 1
    };
  }

  const joint = (pose, leg, name) => pose.joints[`${leg.userSide}${name}`];

  // User leg vectors in the user's pelvis frame, mirrored and in leg lengths.
  function measure(pose, leg, pelvisInverse) {
    const hip = joint(pose, leg, 'Hip');
    const toLocal = point => {
      const v = point.clone().sub(hip).applyQuaternion(pelvisInverse).divideScalar(leg.userLength);
      v.x *= -1;
      return v;
    };
    const ankle = toLocal(joint(pose, leg, 'Ankle')), knee = toLocal(joint(pose, leg, 'Knee'));
    const axis = ankle.clone().normalize();
    return {ankle, pole: knee.addScaledVector(axis, -knee.dot(axis))};
  }

  function calibrate(pose) {
    if (!pose?.bodyFrame) return false;
    worldUp = calibrationUp(pose);
    const pelvis = pelvisFrame(pose, worldUp);
    if (!pelvis) return false;
    const pelvisInverse = frameQuaternion(pelvis.matrix).invert();
    for (const leg of legs) {
      leg.userLength = joint(pose, leg, 'Hip').distanceTo(joint(pose, leg, 'Knee')) +
        joint(pose, leg, 'Knee').distanceTo(joint(pose, leg, 'Ankle'));
      if (!(leg.userLength > 1e-6)) return false;
      const {ankle, pole} = measure(pose, leg, pelvisInverse);
      leg.ankleTarget.copy(ankle); leg.ankle.copy(ankle);
      leg.poleTarget.copy(pole); leg.pole.copy(pole);
      leg.targetFlat = 1; leg.flat = 1;
      [leg.upper, leg.lower, leg.foot].forEach((bone, i) => bone.quaternion.copy(leg.rest[i]));
    }
    calibrated = true;
    return true;
  }

  function setPose(pose) {
    if (!calibrated) return;
    const pelvis = pelvisFrame(pose, worldUp);
    if (!pelvis) return;
    const pelvisInverse = frameQuaternion(pelvis.matrix).invert();
    // A foot on the floor stays flat (the ankle flexes); only a foot raised above the other
    // one follows the shin (kicks, knee lifts).
    const heights = legs.map(leg => joint(pose, leg, 'Ankle').dot(worldUp));
    const lowest = Math.min(...heights);
    for (const [i, leg] of legs.entries()) {
      // An untracked leg keeps its previous target.
      if (pose.metadata?.trackingParts?.[`${leg.userSide}Leg`] === false) continue;
      const {ankle, pole} = measure(pose, leg, pelvisInverse);
      if (![ankle.x, ankle.y, ankle.z, pole.x, pole.y, pole.z].every(Number.isFinite)) continue;
      leg.ankleTarget.copy(ankle); leg.poleTarget.copy(pole);
      leg.targetFlat = 1 - THREE.MathUtils.smoothstep((heights[i] - lowest) / leg.userLength, .04, .15);
    }
  }

  function solve(leg, pelvis) {
    const {upper, lower, foot, thigh, shin} = leg;
    [upper, lower, foot].forEach((bone, i) => bone.quaternion.copy(leg.rest[i]));
    upper.updateWorldMatrix(true, true);
    const hip = position(upper);
    const reach = leg.ankle.clone().applyQuaternion(pelvis);
    reach.setLength(easeReach(reach.length()) * (thigh + shin));
    const direction = reach.lengthSq() > 1e-12 ? reach.clone().normalize() : new THREE.Vector3(0, -1, 0);
    const distance = THREE.MathUtils.clamp(reach.length(), Math.abs(thigh - shin) + 1e-6, thigh + shin - 1e-6);

    // Bend direction: the user's knee offset, or the knee front for a nearly straight leg.
    const perpendicular = v => v.clone().addScaledVector(direction, -v.dot(direction));
    const front = perpendicular(leg.kneeFrontLocal.clone().applyQuaternion(upper.getWorldQuaternion(new THREE.Quaternion())));
    const pole = perpendicular(leg.pole.clone().applyQuaternion(pelvis));
    const poleWeight = THREE.MathUtils.smoothstep(pole.length(), .03, .08);
    const bendDirection = (front.lengthSq() > 1e-10 ? front.normalize() : new THREE.Vector3(0, 0, 1))
      .multiplyScalar(1 - poleWeight);
    if (poleWeight > 0) bendDirection.addScaledVector(pole.normalize(), poleWeight);
    if (bendDirection.lengthSq() < 1e-10) bendDirection.copy(front.lengthSq() > 1e-10 ? front : new THREE.Vector3(0, 0, 1));
    bendDirection.normalize();

    const along = (thigh * thigh - shin * shin + distance * distance) / (2 * distance);
    const height = Math.sqrt(Math.max(0, thigh * thigh - along * along));
    const knee = hip.clone().addScaledVector(direction, along).addScaledVector(bendDirection, height);
    aimBone(upper, lower, knee);

    // Twist the thigh about its axis so the knee faces the bend direction.
    const axis = knee.clone().sub(hip).normalize();
    const current = leg.kneeFrontLocal.clone().applyQuaternion(upper.getWorldQuaternion(new THREE.Quaternion()));
    current.addScaledVector(axis, -current.dot(axis));
    const wanted = bendDirection.clone().addScaledVector(axis, -bendDirection.dot(axis));
    if (current.lengthSq() > 1e-10 && wanted.lengthSq() > 1e-10) {
      current.normalize(); wanted.normalize();
      const twist = Math.atan2(axis.dot(new THREE.Vector3().crossVectors(current, wanted)), current.dot(wanted));
      const world = new THREE.Quaternion().setFromAxisAngle(axis, twist).multiply(upper.getWorldQuaternion(new THREE.Quaternion()));
      upper.quaternion.copy(upper.parent.getWorldQuaternion(new THREE.Quaternion()).invert().multiply(world)).normalize();
      upper.updateWorldMatrix(true, true);
    }
    aimBone(lower, foot, hip.clone().addScaledVector(direction, distance));
  }

  // Blend the foot towards level with the floor (keeping its heading).
  function levelFoot({foot, footUpLocal}, weight) {
    if (weight <= 0) return;
    const footWorld = foot.getWorldQuaternion(new THREE.Quaternion());
    const currentUp = footUpLocal.clone().applyQuaternion(footWorld).normalize();
    const desiredWorld = new THREE.Quaternion().setFromUnitVectors(currentUp, WORLD_UP).multiply(footWorld);
    const desiredLocal = foot.parent.getWorldQuaternion(new THREE.Quaternion()).invert().multiply(desiredWorld);
    foot.quaternion.slerp(desiredLocal, weight);
  }

  function update(deltaSeconds) {
    if (!calibrated) return;
    const alpha = trackingAlpha(deltaSeconds, smoothing);
    character.model.updateMatrixWorld(true);
    const pelvis = rootDelta().multiply(characterPelvis);
    for (const leg of legs) {
      leg.ankle.lerp(leg.ankleTarget, alpha);
      leg.pole.lerp(leg.poleTarget, alpha);
      leg.flat = THREE.MathUtils.lerp(leg.flat, leg.targetFlat, alpha);
      solve(leg, pelvis);
      levelFoot(leg, leg.flat);
    }
    character.model.updateMatrixWorld(true);
  }

  return {calibrate, setPose, update};
}
