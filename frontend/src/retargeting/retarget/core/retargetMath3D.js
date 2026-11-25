import * as THREE from 'three';

const EPSILON = 0.000001;

export function midpoint3D(a, b) {
  return new THREE.Vector3((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2);
}

export function getDirection3D(from, to) {
  if (!from || !to) return null;
  const direction = new THREE.Vector3(to.x - from.x, to.y - from.y, to.z - from.z);
  return direction.lengthSq() < EPSILON ? null : direction.normalize();
}
