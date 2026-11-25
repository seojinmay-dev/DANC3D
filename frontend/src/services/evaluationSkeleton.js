// Desktop Demo draw_skeleton policy. Coordinates remain original; CSS mirrors the overlay.
export const EVALUATION_BONES=[
  ['left_upper_arm',11,13],['left_forearm',13,15],
  ['right_upper_arm',12,14],['right_forearm',14,16],
  ['left_thigh',23,25],['left_shin',25,27],
  ['right_thigh',24,26],['right_shin',26,28],
];
export const TORSO_EDGES=[[11,12],[12,24],[24,23],[23,11]];
// OpenCV BGR converted to Canvas RGB.
export const STATUS_COLORS={OK:'rgb(120,215,80)',ERROR:'rgb(245,35,35)',INVALID:'rgb(145,145,145)'};
export function torsoColor(result) {
  const error=result?.torso_yaw_error_deg;
  // Same display-only threshold as Desktop Demo/core: no angle calculation here.
  return STATUS_COLORS[!Number.isFinite(error)?'INVALID':Math.abs(error)>=15?'ERROR':'OK'];
}
export function drawEvaluationPose(canvas,frame,result) {
  canvas.width=frame.image_width;canvas.height=frame.image_height;
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  const landmarks=frame.image_landmarks;
  if(!landmarks)return;
  const scale=Math.min(600/canvas.width,380/canvas.height);
  const normal=Math.max(1,Math.round(4/scale)),priorityWidth=Math.max(normal+1,Math.round(6/scale));
  const point=index=>{
    const p=landmarks[index];
    return p&&Number.isFinite(p.x)&&Number.isFinite(p.y)?[Math.round(p.x*canvas.width),Math.round(p.y*canvas.height)]:null;
  };
  // Canvas clips strokes to its viewport, matching Desktop's clipped connections.
  function line(parent,child,color,width) {
    const a=point(parent),b=point(child);if(!a||!b)return;
    ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();
  }
  for(const [parent,child] of TORSO_EDGES)line(parent,child,torsoColor(result),result?.priority_feedback_type==='torso_rotation'?priorityWidth:normal);
  const priority=result?.priority_error_bone;
  const ordered=[...EVALUATION_BONES.filter(b=>b[0]!==priority),...EVALUATION_BONES.filter(b=>b[0]===priority)];
  for(const [name,parent,child] of ordered)line(parent,child,STATUS_COLORS[result?.bone_status?.[name]]||STATUS_COLORS.INVALID,name===priority?priorityWidth:normal);
  const joints=[...new Set(EVALUATION_BONES.flatMap(([,a,b])=>[a,b]))].sort((a,b)=>a-b);
  for(const index of joints){
    const p=point(index);if(!p||p[0]<0||p[0]>=canvas.width||p[1]<0||p[1]>=canvas.height)continue;
    ctx.beginPath();ctx.arc(...p,Math.max(1,Math.round(3/scale)),0,Math.PI*2);
    ctx.fillStyle='#ffffff';ctx.fill();ctx.strokeStyle='#000000';ctx.lineWidth=Math.max(1,Math.round(1/scale));ctx.stroke();
  }
}
