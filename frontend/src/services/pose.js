import {FilesetResolver,PoseLandmarker} from '@mediapipe/tasks-vision';
export async function createPoseDetector() {
  const base=`${import.meta.env.BASE_URL}mediapipe`;
  return PoseLandmarker.createFromOptions(await FilesetResolver.forVisionTasks(`${base}/wasm`),{
    baseOptions:{modelAssetPath:`${base}/pose_landmarker_full.task`,delegate:'CPU'},
    runningMode:'VIDEO',numPoses:1,outputSegmentationMasks:false,
  });
}
export function poseFrame(result,video,timestamp) {
  const image=result.landmarks?.[0],world=result.worldLandmarks?.[0];
  const image_landmarks=image?.length===33?image.map(p=>({...p})):null;
  const world_landmarks=world?.length===33&&world.every(p=>[p.x,p.y,p.z].every(Number.isFinite))?world.map(p=>[p.x,p.y,p.z]):null;
  const world_confidence=world_landmarks?world.map(p=>Math.min(p.visibility??1,p.presence??1)):null;
  return {timestamp_ms:timestamp,image_width:video.videoWidth,image_height:video.videoHeight,image_landmarks,world_landmarks,world_confidence,pose_detected:!!image_landmarks};
}
export function drawPose(canvas,frame) {
  canvas.width=frame.image_width;canvas.height=frame.image_height;
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!frame.image_landmarks)return;
  ctx.strokeStyle='#a5f2c4';ctx.fillStyle='#ffffff';ctx.lineWidth=3;
  for(const {start,end} of PoseLandmarker.POSE_CONNECTIONS) {
    const a=frame.image_landmarks[start],b=frame.image_landmarks[end];
    ctx.beginPath();ctx.moveTo(a.x*canvas.width,a.y*canvas.height);ctx.lineTo(b.x*canvas.width,b.y*canvas.height);ctx.stroke();
  }
  for(const p of frame.image_landmarks){ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,4,0,Math.PI*2);ctx.fill();}
}
