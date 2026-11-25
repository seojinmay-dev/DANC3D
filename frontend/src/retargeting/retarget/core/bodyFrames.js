import * as THREE from 'three';

const EPSILON = 0.000001;

// Gram-Schmidt body basis: X = right, Y = up, Z = front. Mutates and returns the inputs.
export function createOrthonormalFrame(right, up) {
  if (right.lengthSq() < EPSILON || up.lengthSq() < EPSILON) return null;
  right.normalize();
  up.addScaledVector(right, -up.dot(right));
  if (up.lengthSq() < EPSILON) return null;
  up.normalize();
  const front = new THREE.Vector3().crossVectors(right, up);
  if (front.lengthSq() < EPSILON) return null;
  front.normalize();
  up.crossVectors(front, right).normalize();
  return {right, up, front, matrix: new THREE.Matrix4().makeBasis(right, up, front)};
}

// Arms and the spine use the pose's torso frame (shoulder + hip axes).
export function createUpperBodyFrame(pose) {
  return pose?.bodyFrame?.matrix ?? null;
}
