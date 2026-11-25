import test from 'node:test';
import assert from 'node:assert/strict';
import {createPracticeController} from '../src/services/practiceController.js';
import {createContentClock} from '../src/services/contentClock.js';
const wait=ms=>new Promise(r=>setTimeout(r,ms));
class Socket extends EventTarget {
  readyState=1;sent=[];pending=[];
  send(text){const p=JSON.parse(text);this.sent.push(p);this.pending.push(p);}
  answer(){const p=this.pending.shift();this.dispatchEvent(new MessageEvent('message',{data:JSON.stringify({session_id:'s',state:'running',playback_time_sec:p.playback_time_sec,evaluation_performed:true,pose_detected:true,valid_user_bone_count:8,aligned_reference_index:0,displayed_score:82,displayed_feedback:'NONE'})}));}
  close(){this.readyState=3;this.dispatchEvent(new Event('close'));}
}
function setup({duration=10,control}={}) {
  let now=0;const socket=new Socket(),abort=new AbortController(),states=[],ended=[];
  const clock=createContentClock(.75,duration,()=>now);
  const camera={statusRef:{current:'running'},latestFrameRef:{current:{timestamp_ms:1,world_landmarks:null,playback_time_sec:0}}};
  const runtime=createPracticeController({socket,session:{session_id:'s',duration_sec:duration},clock,camera,signal:abort.signal,control:control|| (async(_,action)=>({status:action==='end'?'ended':action==='pause'?'paused':'running',record:{mean_raw_final_score:null,valid_sample_count:0}})),publish:s=>states.push(s),onEnded:r=>ended.push(r),lastTimestamp:1});
  const frame=()=>{camera.latestFrameRef.current={timestamp_ms:camera.latestFrameRef.current.timestamp_ms+1,world_landmarks:null,playback_time_sec:clock.read()};};
  return {runtime,socket,clock,camera,states,ended,advance:ms=>now+=ms,frame,cleanup:()=>{abort.abort();runtime.dispose();}};
}
test('Pause drains pending response before REST; resume retains time and skips paused frames; reset first zero',async()=>{
  const calls=[];const f=setup({control:async(_,a)=>{calls.push(a);return {};}});
  try{
    f.runtime.start();f.advance(1000);f.frame();await wait(30);assert.equal(f.socket.sent.length,1);
    const pause=f.runtime.act('pause');const duplicate=f.runtime.act('pause');assert.equal(pause,duplicate);await wait(10);assert.deepEqual(calls,[]);
    f.socket.answer();await pause;assert.deepEqual(calls,['pause']);const frozen=f.clock.read();f.advance(1000);f.frame();await wait(30);assert.equal(f.clock.read(),frozen);assert.equal(f.socket.sent.length,1);
    await f.runtime.act('resume');f.advance(100);f.frame();await wait(30);assert.ok(f.socket.sent.at(-1).playback_time_sec>frozen);f.socket.answer();
    await f.runtime.act('reset');assert.equal(f.clock.read(),0);f.frame();await wait(30);assert.equal(f.socket.sent.at(-1).playback_time_sec,0);f.socket.answer();await wait(10);assert.ok(f.states.some(s=>s.phase==='running'));
    assert.deepEqual(calls,['pause','resume','reset']);
  }finally{f.cleanup();}
});
test('End uncertain response can retry same session, duplicates coalesce, null preserved',async()=>{
  let calls=0;const f=setup({control:async(id,a)=>{assert.equal(id,'s');assert.equal(a,'end');calls++;await wait(10);if(calls===1)throw Error('timeout');return {record:{mean_raw_final_score:null,valid_sample_count:0}};}});
  try{
    const first=f.runtime.act('end');assert.equal(first,f.runtime.act('end'));await first;assert.equal(f.ended.length,0);assert.ok(f.states.some(s=>s.failedAction==='end'));
    await f.runtime.act('end');assert.equal(f.ended.length,1);assert.equal(f.ended[0].mean_raw_final_score,null);await f.runtime.act('end');assert.equal(calls,2);
  }finally{f.cleanup();}
});
test('Content duration automatically ends once without further pose',async()=>{
  const f=setup({duration:1});try{f.runtime.start();f.advance(2000);await wait(60);assert.equal(f.ended.length,1);assert.equal(f.socket.sent.length,0);}finally{f.cleanup();}
});
test('Failed Pause freezes evaluation until same-action reconciliation; competing reset ignored',async()=>{
  let calls=0;const f=setup({control:async(_,a)=>{assert.equal(a,'pause');if(++calls===1)throw Error('timeout');return {status:'paused'};}});
  try{f.runtime.start();await f.runtime.act('pause');f.advance(1000);f.frame();await wait(30);assert.equal(f.socket.sent.length,0);await f.runtime.act('reset');assert.equal(calls,1);await f.runtime.act('pause');assert.equal(calls,2);assert.ok(f.states.some(s=>s.phase==='paused'));}finally{f.cleanup();}
});
