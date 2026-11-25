import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

const REQUIRED_BONES = {
  leftArm: 'LeftArm', rightArm: 'RightArm', leftForeArm: 'LeftForeArm', rightForeArm: 'RightForeArm',
  leftUpLeg: 'LeftUpLeg', rightUpLeg: 'RightUpLeg', leftLeg: 'LeftLeg', rightLeg: 'RightLeg'
};

// scene.js의 카메라(y=1.0, z=3.3) 구도에 딱 맞는 표준 캐릭터 높이 (1.glb 기준 약 1.66m)
const TARGET_CHARACTER_HEIGHT = 1.66;

/**
 * 불러온 GLTF 모델을 표준 키(1.66m)와 바닥 중심(0, 0, 0)으로 정규화합니다.
 * 내부 뼈대(Armature)의 로컬 회전/구조는 건드리지 않아 기존 리타겟팅을 해치지 않습니다.
 */
function normalizeModelTransform(rawScene, targetHeight = TARGET_CHARACTER_HEIGHT) {
  const wrapper = new THREE.Group();
  wrapper.name = 'NormalizedCharacter';
  wrapper.add(rawScene);

  // 1. 월드 매트릭스 갱신 후 실제 바운딩 박스(크기) 측정
  wrapper.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(rawScene);
  const size = new THREE.Vector3();
  box.getSize(size);

  // 2. 높이를 targetHeight(1.66m)로 통일 (3.glb처럼 100배 큰 모델도 자동 축소)
  if (size.y > 0) {
    const scaleFactor = targetHeight / size.y;
    wrapper.scale.setScalar(scaleFactor);
  }

  // 3. 스케일 적용 후 발바닥 최하단(min.y)을 y=0, 좌우/앞뒤 중심을 x=0, z=0에 정렬
  wrapper.updateMatrixWorld(true);
  const scaledBox = new THREE.Box3().setFromObject(rawScene);
  const center = new THREE.Vector3();
  scaledBox.getCenter(center);

  wrapper.position.set(
    -center.x,
    -scaledBox.min.y,
    -center.z
  );

  wrapper.updateMatrixWorld(true);
  return wrapper;
}

export async function loadCharacter(scene, modelPath, { showSkeleton = false } = {}) {
  const gltf = await new GLTFLoader().loadAsync(modelPath);
  const rawModel = gltf.scene;

  // Bones can move the mesh far from its bind-pose bounds; never cull it.
  rawModel.traverse(child => {
    if (child.isMesh) child.frustumCulled = false;
  });

  // 기존 리타겟팅 로직이 기대하는 본 이름 검증 (mixamorig...)
  const bones = {};
  for (const [key, name] of Object.entries(REQUIRED_BONES)) {
    bones[key] = rawModel.getObjectByName(`mixamorig${name}`);
    if (!bones[key]) throw new Error(`Bone을 찾지 못했습니다: ${key}`);
  }

  // 크기 및 바닥 위치 정규화 적용
  const model = normalizeModelTransform(rawModel, TARGET_CHARACTER_HEIGHT);
  scene.add(model);

  const skeletonHelper = new THREE.SkeletonHelper(model);
  skeletonHelper.visible = showSkeleton; // 기본값 false: 얇은 뼈대 선 숨김
  scene.add(skeletonHelper);

  return {model, bones, skeletonHelper};
}