import * as THREE from 'three';
import {STATUS_COLORS, torsoColor} from '../../services/evaluationSkeleton.js';

// Evaluation bone names are the user's (MediaPipe) sides. The live retargeter mirrors the
// user onto the character, so the user's left limb is the character's Right bones.
export const EVALUATION_BONES_3D = [
  ['left_upper_arm', 'RightArm', 'RightForeArm'], ['left_forearm', 'RightForeArm', 'RightHand'],
  ['right_upper_arm', 'LeftArm', 'LeftForeArm'], ['right_forearm', 'LeftForeArm', 'LeftHand'],
  ['left_thigh', 'RightUpLeg', 'RightLeg'], ['left_shin', 'RightLeg', 'RightFoot'],
  ['right_thigh', 'LeftUpLeg', 'LeftLeg'], ['right_shin', 'LeftLeg', 'LeftFoot']
];
// Shoulders and hips quad, same as the 2D overlay's TORSO_EDGES.
const TORSO_EDGES_3D = [['LeftArm', 'RightArm'], ['RightArm', 'RightUpLeg'], ['RightUpLeg', 'LeftUpLeg'], ['LeftUpLeg', 'LeftArm']];

const UP = new THREE.Vector3(0, 1, 0);
const colorCache = new Map();
const color = css => {
  if (!colorCache.has(css)) colorCache.set(css, new THREE.Color(css));
  return colorCache.get(css);
};

// Draws the evaluation result on the character's bones, on top of the mesh.
export function createEvaluationSkeleton3D(scene, model) {
  const bone = name => {
    const object = model.getObjectByName(`mixamorig${name}`);
    if (!object) throw new Error(`Bone을 찾지 못했습니다: ${name}`);
    return object;
  };
  const group = new THREE.Group();
  group.renderOrder = 999;
  scene.add(group);

  const cylinder = new THREE.CylinderGeometry(1, 1, 1, 10);
  const sphere = new THREE.SphereGeometry(1, 12, 8);
  const overlayMaterial = css => new THREE.MeshBasicMaterial({color: css, depthTest: false, depthWrite: false, transparent: true, opacity: .9});

  const segment = (key, from, to) => {
    const mesh = new THREE.Mesh(cylinder, overlayMaterial(STATUS_COLORS.INVALID));
    mesh.renderOrder = 999;
    group.add(mesh);
    return {key, from: bone(from), to: bone(to), mesh};
  };
  const limbs = EVALUATION_BONES_3D.map(([key, from, to]) => segment(key, from, to));
  const torso = TORSO_EDGES_3D.map(([from, to]) => segment('torso', from, to));

  const jointMaterial = overlayMaterial('#ffffff');
  const jointBones = [...new Set(EVALUATION_BONES_3D.flatMap(([, a, b]) => [a, b]))].map(bone);
  const joints = jointBones.map(object => {
    const mesh = new THREE.Mesh(sphere, jointMaterial);
    mesh.renderOrder = 1001;
    group.add(mesh);
    return {object, mesh};
  });

  // Thickness follows the character's own scale (shoulder-to-hip length in bind pose).
  model.updateMatrixWorld(true);
  const shoulder = bone('LeftArm').getWorldPosition(new THREE.Vector3());
  const hip = bone('LeftUpLeg').getWorldPosition(new THREE.Vector3());
  const unit = shoulder.distanceTo(hip) || .5;
  const normalRadius = unit * .042, priorityRadius = unit * .065, jointRadius = unit * .06;

  const a = new THREE.Vector3(), b = new THREE.Vector3(), direction = new THREE.Vector3();
  const place = ({from, to, mesh}, radius) => {
    from.getWorldPosition(a); to.getWorldPosition(b);
    direction.subVectors(b, a);
    const length = direction.length();
    mesh.visible = length > 1e-6;
    if (!mesh.visible) return;
    mesh.position.addVectors(a, b).multiplyScalar(.5);
    mesh.quaternion.setFromUnitVectors(UP, direction.divideScalar(length));
    mesh.scale.set(radius, length, radius);
  };

  return {
    // Call after the pose is applied for this frame.
    update(result) {
      const priority = result?.priority_error_bone;
      const torsoPriority = result?.priority_feedback_type === 'torso_rotation';
      const torsoCss = torsoColor(result);
      for (const edge of torso) {
        edge.mesh.material.color.copy(color(torsoCss));
        place(edge, torsoPriority ? priorityRadius : normalRadius);
      }
      for (const limb of limbs) {
        const isPriority = limb.key === priority;
        limb.mesh.material.color.copy(color(STATUS_COLORS[result?.bone_status?.[limb.key]] || STATUS_COLORS.INVALID));
        // Priority bone is drawn last so it stays on top, as in the 2D overlay.
        limb.mesh.renderOrder = isPriority ? 1000 : 999;
        place(limb, isPriority ? priorityRadius : normalRadius);
      }
      for (const {object, mesh} of joints) {
        object.getWorldPosition(mesh.position);
        mesh.scale.setScalar(jointRadius);
      }
    },
    dispose() {
      scene.remove(group);
      cylinder.dispose(); sphere.dispose(); jointMaterial.dispose();
      for (const {mesh} of [...limbs, ...torso]) mesh.material.dispose();
    }
  };
}
