import test from 'node:test';import assert from 'node:assert/strict';
import {evaluatePose,streamEvaluation,validateWorld} from '../src/services/continuousEvaluation.js';
import {createContentClock} from '../src/services/contentClock.js';
class Socket extends EventTarget {
  readyState=1;sent=[];pending=0;maxPending=0;
  send(text){const body=JSON.parse(text);this.sent.push(body);this.pending++;this.maxPending=Math.max(this.maxPending,this.pending);this.handler?.(body);}
  respond(body){this.pending--;this.dispatchEvent(new MessageEvent('message',{data:JSON.stringify({session_id:'s',state:'running',evaluation_performed:true,pose_detected:body.world_landmarks!==null,displayed_score:body.world_landmarks===null?80:82,displayed_feedback:'팔을 펴주세요.',...body})}));}
}
const world=Array.from({length:33},()=>[.1,.2,.3]);
test('raw world and null accepted, malformed values rejected',()=>{validateWorld(world);validateWorld(null);for(const data of [[],[[1,2,3]],Array(33).fill([NaN,0,0]),Array(33).fill([true,0,0])])assert.throws(()=>validateWorld(data));});
test('all supported speeds apply once to captured time, not send time',()=>{for(const speed of [.5,.75,1,1.25,1.5]){let time=100;const clock=createContentClock(speed,100,()=>time);clock.start();const capture=clock.read(1100);time=2100;assert.equal(capture,speed);assert.equal(clock.read(),2*speed);}});
test('null frame retains server displayed values and only protocol fields sent',async()=>{const s=new Socket();s.handler=body=>queueMicrotask(()=>s.respond(body));const r=await evaluatePose(s,'s',{world_landmarks:null,playback_time_sec:.5,timestamp_ms:999},{signal:new AbortController().signal});assert.equal(r.pose_detected,false);assert.equal(r.displayed_score,80);assert.equal(r.displayed_feedback,'팔을 펴주세요.');assert.deepEqual(Object.keys(s.sent[0]).sort(),['playback_time_sec','type','world_landmarks']);});
test('one request at a time, drops backlog, preserves capture time, stops at duration',async()=>{
 const s=new Socket(),controller=new AbortController();const ref={current:{timestamp_ms:2,playback_time_sec:.2,world_landmarks:world}};let current=.3;
 s.handler=body=>setTimeout(()=>{if(s.sent.length===1)ref.current={timestamp_ms:5,playback_time_sec:.5,world_landmarks:null};else current=1;s.respond(body);},30);
 const results=[];await streamEvaluation(s,'s',ref,{signal:controller.signal,clock:{read:()=>current},duration:1,lastTimestamp:1,onResult:r=>results.push(r)});
 assert.equal(s.maxPending,1);assert.deepEqual(s.sent.map(x=>x.playback_time_sec),[.2,.5]);assert.equal(results.length,2);assert.equal(s.sent[1].world_landmarks,null);
});
test('unchanged frame not sent twice and abort cleans waiting loop',async()=>{const s=new Socket(),c=new AbortController();const ref={current:{timestamp_ms:2,playback_time_sec:.2,world_landmarks:world}};s.handler=body=>{s.respond(body);setTimeout(()=>c.abort(),45);};await assert.rejects(streamEvaluation(s,'s',ref,{signal:c.signal,clock:{read:()=>.3},duration:10,lastTimestamp:1,onResult:()=>{}}),{name:'AbortError'});assert.equal(s.sent.length,1);});
test('delayed response and socket close fail safely',async()=>{const frame={world_landmarks:world,playback_time_sec:1};await assert.rejects(evaluatePose(new Socket(),'s',frame,{signal:new AbortController().signal,timeoutMs:25}),/지연/);const s=new Socket();const p=evaluatePose(s,'s',frame,{signal:new AbortController().signal});s.dispatchEvent(new Event('close'));await assert.rejects(p,/종료/);});
