import test from 'node:test';
import assert from 'node:assert/strict';
import {createReferencePlayback} from '../src/services/referencePlayback.js';
import {createContentClock} from '../src/services/contentClock.js';
const flush=async()=>{for(let i=0;i<12;i++)await Promise.resolve();};
function setup({slow=false,speed=1}={}){
 let now=0,tick,pending=0,maxPending=0,id=0;
 const clock=createContentClock(speed,100,()=>now),requests=[],shown=[],revoked=[],waiting=[];
 const stop=createReferencePlayback({readTime:clock.read,urlFor:t=>t,onFrame:url=>shown.push([now,Number(url.split(':')[1])]),onError:()=>{},
 schedule:fn=>{tick=fn;return 1;},cancel:()=>{},createUrl:b=>`blob:${b}`,revokeUrl:u=>revoked.push(u),decode:async()=>{},
 fetchFrame:(time,{signal})=>{requests.push(time);pending++;maxPending=Math.max(maxPending,pending);return new Promise((resolve,reject)=>{
 const finish=()=>{pending--;resolve({ok:true,blob:async()=>time});};
 signal.addEventListener('abort',()=>{pending--;reject(Object.assign(Error('abort'),{name:'AbortError'}));},{once:true});
 if(slow)waiting.push(finish);else finish();
 });}});
 return {clock,requests,shown,revoked,waiting,stop,get maxPending(){return maxPending;},async step(ms){now+=ms;tick();await flush();},flush};
}
test('30fps clock following, bounded decoded cache, all speeds applied once',async()=>{
 for(const speed of [.5,.75,1,1.25,1.5]){
 const f=setup({speed});await flush();f.clock.start();for(let i=0;i<600;i++)await f.step(1000/60);
 const displayed=f.shown.filter(([at])=>at>1000).length/9;
 console.log(`Reference simulated ${speed}x: ${displayed.toFixed(1)} displayed fps, ${f.requests.length} requests, max in-flight ${f.maxPending}`);
 assert.ok(displayed>Math.min(30*speed,60)-2);assert.equal(f.maxPending,1);
 assert.ok(Math.abs(f.shown.at(-1)[1]-10*speed)<.1);
 assert.equal(new Set(f.requests).size,f.requests.length);
 assert.ok(f.requests.length-f.revoked.length<=12);f.stop();assert.equal(f.revoked.length,f.requests.length);
 }
});
test('pause makes no new requests; resume follows clock; reset uses cached zero',async()=>{
 const f=setup();await flush();f.clock.start();for(let i=0;i<60;i++)await f.step(1000/60);
 f.clock.pause();await f.step(0);const count=f.requests.length,frame=f.shown.at(-1);for(let i=0;i<60;i++)await f.step(1000/60);
 assert.equal(f.requests.length,count);assert.deepEqual(f.shown.at(-1),frame);
 f.clock.start();await f.step(100);assert.ok(f.requests.at(-1)>.9);f.clock.reset();await f.step(0);await f.step(0);assert.equal(f.shown.at(-1)[1],0);f.stop();
});
test('slow responses skip backlog, never overwrite newer frame; reset aborts stale generation',async()=>{
 const f=setup({slow:true});f.waiting.shift()();await flush();f.clock.start();await f.step(1200);
 assert.equal(f.requests.at(-1),1.2);await f.step(300);f.waiting.shift()();await flush();await f.step(0);
 assert.equal(f.requests.at(-1),1.5);assert.equal(f.maxPending,1);
 f.clock.reset();await f.step(0);await flush();await f.step(0);
 assert.equal(f.shown.at(-1)[1],0);const count=f.shown.length;f.waiting.shift()();await flush();assert.equal(f.shown.length,count);f.stop();
});
test('prefetch is decoded before use and cleanup releases pending decode URL',async()=>{
 const f=setup();await flush();f.clock.start();await f.step(1);assert.ok(f.requests.includes(1/30));const count=f.requests.length;await f.step(20);
 assert.equal(f.shown.at(-1)[1],1/30);assert.ok(f.requests.length<=count+1);f.stop();
});
test('failure stops retries and keeps last image; disposal revokes a decoding URL',async()=>{
 let tick,requests=0,error='',resolveDecode;const revoked=[];
 const stop=createReferencePlayback({readTime:()=>0,urlFor:t=>t,onFrame:()=>assert.fail('disposed frame displayed'),onError:e=>error=e,
 schedule:fn=>{tick=fn;return 1;},cancel:()=>{},fetchFrame:async()=>{requests++;return {ok:true,blob:async()=>1};},
 createUrl:()=> 'pending',revokeUrl:u=>revoked.push(u),decode:()=>new Promise(r=>resolveDecode=r)});
 await flush();stop();assert.deepEqual(revoked,['pending']);resolveDecode();await flush();assert.deepEqual(revoked,['pending']);assert.equal(error,'');
 const stopFailure=createReferencePlayback({readTime:()=>0,urlFor:t=>t,onFrame:()=>assert.fail('failed image displayed'),onError:e=>error=e,
 schedule:fn=>{tick=fn;return 1;},cancel:()=>{},fetchFrame:async()=>{requests++;throw Error('offline');}});
 await flush();const count=requests;for(let i=0;i<100;i++)tick();assert.equal(requests,count);assert.equal(error,'offline');stopFailure();
});
