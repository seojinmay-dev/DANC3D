import test from 'node:test';
import assert from 'node:assert/strict';
import {EVALUATION_BONES,TORSO_EDGES,STATUS_COLORS,drawEvaluationPose,torsoColor} from '../src/services/evaluationSkeleton.js';
function fixture(){
  const strokes=[],joints=[];
  const ctx={clearRect(){},beginPath(){},moveTo(x,y){this.start=[x,y];},lineTo(x,y){this.end=[x,y];},stroke(){if(this.end)strokes.push({a:this.start,b:this.end,color:this.strokeStyle,width:this.lineWidth});this.end=null;},arc(x,y){joints.push([x,y]);},fill(){}};
  const canvas={getContext:()=>ctx};
  const frame={image_width:600,image_height:380,image_landmarks:Array.from({length:33},(_,i)=>({x:(i+1)/40,y:.5}))};
  return {canvas,frame,strokes,joints};
}
test('Desktop connection indices, status colors, priority order and body joints',()=>{
  assert.deepEqual(EVALUATION_BONES.map(b=>b.slice(1)),[[11,13],[13,15],[12,14],[14,16],[23,25],[25,27],[24,26],[26,28]]);
  assert.deepEqual(TORSO_EDGES,[[11,12],[12,24],[24,23],[23,11]]);
  const {canvas,frame,strokes,joints}=fixture();const original=JSON.stringify(frame);
  drawEvaluationPose(canvas,frame,{bone_status:Object.fromEntries(EVALUATION_BONES.map(([name])=>[name,'OK'])),priority_error_bone:'left_upper_arm',torso_yaw_error_deg:0});
  assert.equal(strokes.length,12);assert.equal(joints.length,12);
  assert.equal(strokes.at(-1).width,6);assert.deepEqual(strokes.at(-1).a,[180,190]);
  assert.ok(strokes.every(s=>s.color===STATUS_COLORS.OK));assert.equal(JSON.stringify(frame),original);
});
test('Error comes from server status, missing status gray, null pose clears overlay',()=>{
  const {canvas,frame,strokes}=fixture();
  drawEvaluationPose(canvas,frame,{bone_status:{left_upper_arm:'ERROR'},torso_yaw_error_deg:null});
  assert.equal(strokes[4].color,STATUS_COLORS.ERROR);assert.equal(strokes[5].color,STATUS_COLORS.INVALID);assert.equal(strokes[0].color,STATUS_COLORS.INVALID);
  strokes.length=0;drawEvaluationPose(canvas,{...frame,image_landmarks:null},null);assert.equal(strokes.length,0);
  drawEvaluationPose(canvas,frame,null);assert.ok(strokes.every(s=>s.color===STATUS_COLORS.INVALID));
});
test('Desktop torso threshold and priority thickness',()=>{
  assert.equal(torsoColor({torso_yaw_error_deg:14.999}),STATUS_COLORS.OK);
  assert.equal(torsoColor({torso_yaw_error_deg:-15}),STATUS_COLORS.ERROR);
  assert.equal(torsoColor(null),STATUS_COLORS.INVALID);
  const {canvas,frame,strokes}=fixture();drawEvaluationPose(canvas,frame,{torso_yaw_error_deg:15,priority_feedback_type:'torso_rotation'});
  assert.ok(strokes.slice(0,4).every(s=>s.color===STATUS_COLORS.ERROR&&s.width===6));
});
