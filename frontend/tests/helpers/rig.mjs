import {readFileSync} from 'node:fs';
import * as THREE from 'three';

// Read the actual GLB's rig without textures/WebGL; use the same sanitized node names as GLTFLoader.
export function loadRig({geometry=false}={}) {
  const buffer = readFileSync(new URL('../../public/models/dancer.glb',import.meta.url));
  const doc = JSON.parse(buffer.subarray(20,20+buffer.readUInt32LE(12)).toString());
  const binary=buffer.subarray(28+buffer.readUInt32LE(12));
  function accessor(index) {
    const entry=doc.accessors[index],view=doc.bufferViews[entry.bufferView];
    const dimensions={SCALAR:1,VEC2:2,VEC3:3,VEC4:4,MAT4:16}[entry.type];
    const bytes={5121:1,5123:2,5126:4}[entry.componentType];
    const read={5121:'readUInt8',5123:'readUInt16LE',5126:'readFloatLE'}[entry.componentType];
    const values=entry.componentType===5126?new Float32Array(entry.count*dimensions):new Uint16Array(entry.count*dimensions);
    for(let i=0;i<entry.count;i++)for(let j=0;j<dimensions;j++) {
      const offset=(view.byteOffset||0)+(entry.byteOffset||0)+i*(view.byteStride||dimensions*bytes)+j*bytes;
      values[i*dimensions+j]=binary[read](offset);
    }
    return new THREE.BufferAttribute(values,dimensions,entry.normalized||false);
  }
  const joints = new Set(doc.skins.flatMap(s=>s.joints));
  const nodes = doc.nodes.map((entry,i)=>{
    const node = geometry&&entry.skin!==undefined ? new THREE.SkinnedMesh() : joints.has(i) ? new THREE.Bone() : new THREE.Object3D();
    node.name = THREE.PropertyBinding.sanitizeNodeName(entry.name || '');
    if(entry.translation)node.position.fromArray(entry.translation);
    if(entry.rotation)node.quaternion.fromArray(entry.rotation);
    if(entry.scale)node.scale.fromArray(entry.scale);
    if(entry.matrix)new THREE.Matrix4().fromArray(entry.matrix).decompose(node.position,node.quaternion,node.scale);
    if(entry.skin!==undefined)node.isSkinnedMesh=true;
    return node;
  });
  doc.nodes.forEach((entry,i)=>entry.children?.forEach(child=>nodes[i].add(nodes[child])));
  const model = new THREE.Group(); doc.scenes[doc.scene||0].nodes.forEach(i=>model.add(nodes[i]));
  const scene = new THREE.Scene(); scene.add(model); model.updateMatrixWorld(true);
  if(geometry)doc.nodes.forEach((entry,i)=>{
    if(entry.skin===undefined)return;
    const primitive=doc.meshes[entry.mesh].primitives[0],attributes=primitive.attributes;
    nodes[i].geometry.setAttribute('position',accessor(attributes.POSITION));
    nodes[i].geometry.setAttribute('skinIndex',accessor(attributes.JOINTS_0));
    nodes[i].geometry.setAttribute('skinWeight',accessor(attributes.WEIGHTS_0));
    const skin=doc.skins[entry.skin],inverses=accessor(skin.inverseBindMatrices).array;
    nodes[i].bind(new THREE.Skeleton(skin.joints.map(joint=>nodes[joint]),
      skin.joints.map((_,j)=>new THREE.Matrix4().fromArray(inverses,j*16))),new THREE.Matrix4());
  });
  const bones = Object.fromEntries(['leftArm','rightArm','leftForeArm','rightForeArm','spine','spine1','spine2',
    'leftUpLeg','rightUpLeg','leftLeg','rightLeg'].map(key=>[key,model.getObjectByName('mixamorig'+key[0].toUpperCase()+key.slice(1))]));
  return {model,bones};
}

