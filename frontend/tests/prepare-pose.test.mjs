import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readdir,readFile,rm,stat,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {ensureModel,MODEL_URL,MODEL_SHA256,MODEL_SIZE,TIMEOUT_MS} from '../scripts/prepare-pose.mjs';

// Small fake model streamed in several chunks; no network and no real model file are used.
const DATA=new Uint8Array(256000).map((_,i)=>i%256);
const SHA=createHash('sha256').update(DATA).digest('hex');
const EXPECTED={size:DATA.length,sha256:SHA};
const noNetwork=()=>{throw new Error('network must not be used');};

function body(bytes,{chunk=65536,onFirst,failAfter,error}={}) {
  let offset=0,reads=0;
  return new ReadableStream({async pull(controller) {
    if(failAfter!==undefined&&reads>=failAfter) {controller.error(error);return;}
    if(offset>=bytes.length) {controller.close();return;}
    controller.enqueue(bytes.slice(offset,offset+=chunk));reads++;
    if(reads===1&&onFirst) await onFirst();
  }});
}
const fetchOf=(bytes,options,init)=>{
  const calls=[];
  const fetchImpl=async(url,request)=>{calls.push({url,request});return new Response(body(bytes,options),init);};
  return {fetchImpl,calls};
};

async function setup(t) {
  const dir=await mkdtemp(join(tmpdir(),'danc3d-prepare-pose-'));
  t.after(()=>rm(dir,{recursive:true,force:true}));
  const logs={log:[],warn:[]};
  t.mock.method(console,'log',(...a)=>logs.log.push(a.join(' ')));
  t.mock.method(console,'warn',(...a)=>logs.warn.push(a.join(' ')));
  return {dir,target:join(dir,'models','mediapipe','pose_landmarker_full.task'),logs};
}
const tempFiles=async target=>(await readdir(join(target,'..')).catch(()=>[])).filter(name=>name.endsWith('.part'));
const exists=path=>stat(path).then(()=>true,()=>false);
async function writeExisting(target,bytes) {await mkdir(join(target,'..'),{recursive:true});await writeFile(target,bytes);}
async function assertFailure(target,fetchImpl,text,options={}) {
  await assert.rejects(ensureModel(target,{fetchImpl,...EXPECTED,...options}),error=>{
    assert.match(error.message,text);assert.ok(error.message.includes(MODEL_URL));return true;
  });
  assert.equal(await exists(target),false);
  assert.deepEqual(await tempFiles(target),[]);
}

test('pinned model constants match the official generation URL, size and SHA-256',()=>{
  assert.equal(MODEL_URL,'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task?generation=1682642787774579');
  assert.equal(MODEL_SHA256,'4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad');
  assert.equal(MODEL_SIZE,9398198);assert.equal(TIMEOUT_MS,60000);
});

test('existing model is used without any network request',async t=>{
  const {target,logs}=await setup(t);await writeExisting(target,DATA);
  assert.deepEqual(await ensureModel(target,{fetchImpl:noNetwork,...EXPECTED}),{path:target,downloaded:false});
  assert.deepEqual(logs.warn,[]);assert.deepEqual(await readFile(target),Buffer.from(DATA));
});

test('existing model with a different size or hash only warns and is kept',async t=>{
  const {target,logs}=await setup(t);
  await writeExisting(target,Buffer.from('user model'));
  assert.equal((await ensureModel(target,{fetchImpl:noNetwork,...EXPECTED})).downloaded,false);
  assert.match(logs.warn[0],/Warning: .* expected 256000/);
  assert.equal(await readFile(target,'utf8'),'user model');
  await writeFile(target,new Uint8Array(DATA.length));
  await ensureModel(target,{fetchImpl:noNetwork,...EXPECTED});
  assert.match(logs.warn[1],/SHA-256 differs/);
  assert.deepEqual(await readFile(target),Buffer.alloc(DATA.length));
});

test('missing model is downloaded from the pinned URL and verified',async t=>{
  const {target,logs}=await setup(t);const {fetchImpl,calls}=fetchOf(DATA);
  assert.deepEqual(await ensureModel(target,{fetchImpl,...EXPECTED}),{path:target,downloaded:true});
  assert.equal(calls.length,1);assert.equal(calls[0].url,MODEL_URL);assert.ok(calls[0].request.signal instanceof AbortSignal);
  assert.deepEqual(await readFile(target),Buffer.from(DATA));
  assert.deepEqual(await tempFiles(target),[]);assert.ok(logs.log.some(line=>line.includes('Saved MediaPipe model')));
});

test('hash mismatch, oversized and truncated downloads save nothing',async t=>{
  const {target}=await setup(t);
  await assertFailure(target,fetchOf(new Uint8Array(DATA.length)).fetchImpl,/SHA-256 mismatch/);
  await assertFailure(target,fetchOf(new Uint8Array([...DATA,1])).fetchImpl,/exceeds/);
  await assertFailure(target,fetchOf(DATA.slice(1)).fetchImpl,/expected 256000/);
});

test('network failure and HTTP errors save nothing',async t=>{
  const {target}=await setup(t);
  await assertFailure(target,async()=>{throw new TypeError('fetch failed');},/fetch failed/);
  await assertFailure(target,fetchOf(new Uint8Array(),{},{status:404}).fetchImpl,/HTTP 404/);
});

test('timeout before the response or while reading cleans the partial file',async t=>{
  const {target}=await setup(t);
  const waitAbort=signal=>new Promise((_,reject)=>signal.addEventListener('abort',()=>reject(signal.reason)));
  await assertFailure(target,(url,{signal})=>waitAbort(signal),/timeout/i,{timeoutMs:20});
  const stalled=async(url,{signal})=>new Response(new ReadableStream({
    start(controller) {controller.enqueue(DATA.slice(0,1000));},
    pull(controller) {return waitAbort(signal).catch(error=>controller.error(error));},
  }));
  await assertFailure(target,stalled,/timeout/i,{timeoutMs:20});
});

test('stream error mid-download cleans the partial file',async t=>{
  const {target}=await setup(t);
  await assertFailure(target,fetchOf(DATA,{failAfter:1,error:new Error('connection reset')}).fetchImpl,/connection reset/);
});

test('final file created by another process during download is never overwritten',async t=>{
  const {target,logs}=await setup(t);
  const {fetchImpl}=fetchOf(DATA,{onFirst:()=>writeFile(target,'created by another process')});
  assert.deepEqual(await ensureModel(target,{fetchImpl,...EXPECTED}),{path:target,downloaded:false});
  assert.equal(await readFile(target,'utf8'),'created by another process');
  assert.ok(logs.warn.some(line=>line.includes('created by another process')));
  assert.ok(logs.warn.some(line=>line.startsWith('Warning:')));
  assert.deepEqual(await tempFiles(target),[]);
});

test('concurrent downloads leave one verified model and no temp files',async t=>{
  const {target}=await setup(t);
  let arrived=0,release;const bothStarted=new Promise(resolve=>{release=resolve;});
  const onFirst=()=>{if(++arrived===2) release();return bothStarted;};
  const results=await Promise.all([0,1].map(()=>ensureModel(target,{fetchImpl:fetchOf(DATA,{onFirst}).fetchImpl,...EXPECTED})));
  assert.deepEqual(results.map(r=>r.downloaded).sort(),[false,true]);
  assert.deepEqual(await readFile(target),Buffer.from(DATA));
  assert.deepEqual(await tempFiles(target),[]);
});

test('importing the script never runs the copy step, even when argv[1] is not a file',async()=>{
  const {execFileSync}=await import('node:child_process');
  const script=new URL('../scripts/prepare-pose.mjs',import.meta.url).href;
  const output=execFileSync(process.execPath,['--input-type=module','-e',`await import(${JSON.stringify(script)});console.log('imported');`,'/nonexistent/argv-path'],{encoding:'utf8'});
  assert.equal(output.trim(),'imported');
});
