// Read-only backend code; all test files, DB and browser profiles stay under frontend/.
import {spawn} from 'node:child_process';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.DANC3D_PLAYWRIGHT_MODULE||'playwright-core');
const root=resolve(new URL('../',import.meta.url).pathname);
const cache=join(root,'node_modules/.cache/danc3d-validation');await mkdir(cache,{recursive:true});
const python=process.env.DANC3D_PYTHON||'python';
const run=join(cache,`run-${Date.now()}`);await mkdir(run);
const env={...process.env,TMPDIR:run,TMP:run,TEMP:run,PYTHONDONTWRITEBYTECODE:'1',MPLCONFIGDIR:join(run,'mpl')};
let backend,vite,context;const logs=[],children=[];
function server(command,args,options){const child=spawn(command,args,{...options,env:{...env,...options?.env},stdio:['ignore','pipe','pipe']});children.push(child);child.stdout.on('data',x=>logs.push(x.toString()));child.stderr.on('data',x=>logs.push(x.toString()));return child;}
async function available(url){try{await fetch(url,{signal:AbortSignal.timeout(1000)});return false;}catch{return true;}}
async function wait(url){for(let i=0;i<200;i++){try{if((await fetch(url)).ok)return;}catch{}await new Promise(r=>setTimeout(r,100));}throw Error('Server unavailable '+url);}
const report={};
try{
 for(const url of ['http://127.0.0.1:18010/health','http://localhost:5175'])if(!await available(url))throw Error(`Verification port occupied; no existing server will be stopped: ${url}`);
 backend=server(python,['-B','-m','uvicorn','app.main:app','--host','127.0.0.1','--port','18010'],{cwd:resolve(root,'..'),env:{DANC3D_DB_PATH:join(run,'verification.db'),DANC3D_CORS_ORIGINS:'http://localhost:3000,http://localhost:5173,http://localhost:5175'}});
 vite=server('npm',['run','dev','--','--port','5175','--strictPort'],{cwd:root,env:{DANC3D_API_TARGET:'http://127.0.0.1:18010'}});
 await wait('http://127.0.0.1:18010/health');await wait('http://localhost:5175');
 if(backend.exitCode!==null||vite.exitCode!==null)throw Error('Owned verification server failed');
 const args=['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream'];
 if(process.env.DANC3D_FAKE_CAMERA_VIDEO)args.push(`--use-file-for-fake-video-capture=${process.env.DANC3D_FAKE_CAMERA_VIDEO}`);
 context=await chromium.launchPersistentContext(join(run,'chrome'),{executablePath:process.env.DANC3D_CHROME||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args,env,baseURL:'http://localhost:5175'});
 const page=await context.newPage(),errors=[],poseMessages=[],evaluationResults=[];let starts=0,readinessCalls=0;
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>{if(r.url().endsWith('/api/sessions/start'))starts++;if(r.url().endsWith('/api/readiness'))readinessCalls++;});
 page.on('websocket',ws=>{ws.on('framereceived',f=>{try{const data=JSON.parse(f.payload);if(data.session_id)evaluationResults.push(data);}catch{}});ws.on('framesent',f=>{try{poseMessages.push(JSON.parse(f.payload));}catch{}});});
 await page.addInitScript(()=>{window.__streams=[];const get=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);navigator.mediaDevices.getUserMedia=async o=>{const s=await get(o);window.__streams.push(s);return s;};window.__poseCreates=0;window.__poseCloses=0;window.__noPose=false;window.__evalSockets=[];const Native=window.WebSocket;window.WebSocket=class extends Native{constructor(...args){super(...args);if(String(args[0]).includes('/ws/sessions/')){window.__evalSockets.push(this);this.addEventListener('message',e=>window.__lastEval=JSON.parse(e.data));}}};});
 // Instrument the real SDK factory, preserving actual inference and all returned coordinates.
 await page.route('**/src/services/pose.js',async route=>{
   const response=await route.fetch();let source=await response.text();
   source=source.replace('export async function createPoseDetector()', 'async function originalCreatePoseDetector()');
   source=source.replace('export function poseFrame(result,video,timestamp)', 'function originalPoseFrame(result,video,timestamp)');
   source+='\nexport function poseFrame(...args){const frame=originalPoseFrame(...args);if(window.__noPose){frame.image_landmarks=null;frame.world_landmarks=null;frame.pose_detected=false;}return frame;}';
   source+='\nexport async function createPoseDetector(){window.__poseCreates++;const d=await originalCreatePoseDetector();const close=d.close.bind(d);d.close=()=>{window.__poseCloses++;close();};return d;}';
   await route.fulfill({response,body:source});
 });
 // Controlled clock boundary for automatic-End integration; real webcam timing is separate.
 await page.route('**/src/services/contentClock.js',async route=>{
   const response=await route.fetch();let source=await response.text();
   source=source.replace('export function createContentClock(', 'function originalClock(');
   source+='\nexport function createContentClock(speed,duration,...args){const c=originalClock(speed,duration,...args);const read=c.read;return {...c,read:(...a)=>window.__reachEnd?duration:read(...a)};}';
   await route.fulfill({response,body:source});
 });
 await page.goto('http://localhost:5175');const guest=await page.evaluate(()=>localStorage.getItem('danc3d.guest_id'));assert.ok(guest);
 await page.reload();assert.equal(await page.evaluate(()=>localStorage.getItem('danc3d.guest_id')),guest);
 await page.locator('#nickname').fill('연동 검증');await page.getByRole('button',{name:'연습 시작 →',exact:true}).click();await page.waitForFunction(()=>document.querySelectorAll('.song-card').length===10);report.song_count=10;
 const songResponses=await page.request.get('/api/songs');const songs=await songResponses.json();
 report.references=[];
 for(const song of songs){const before=Date.now();const r=await page.request.get(`/api/songs/${song.song_id}/reference-frame?time_sec=0`);assert.equal(r.status(),200);assert.match(r.headers()['content-type'],/image\/jpeg/);report.references.push({song_id:song.song_id,ms:Date.now()-before,row:r.headers()['x-reference-row'],frame:r.headers()['x-frame-index']});}
 await page.getByRole('textbox',{name:'곡 또는 아티스트 검색'}).fill('소원을');await page.locator('.song-card').click();await page.getByRole('button',{name:'선택 완료 →'}).click();await page.getByRole('button',{name:'0.75x',exact:true}).click();await page.getByRole('button',{name:'다음 →'}).click();
 await page.waitForFunction(()=>document.querySelector('.pose-debug')?.textContent.includes('Image 33/33'),{},{timeout:120000});assert.equal(readinessCalls,0);assert.equal(starts,0);assert.equal(await page.evaluate(()=>window.__streams.length),1);assert.equal(await page.evaluate(()=>window.__poseCreates),1);
 await page.keyboard.press('Space');await page.locator('.counting').waitFor({timeout:15000});assert.equal(starts,0);
 let release;const gate=new Promise(r=>release=r);
 await page.route('**/api/sessions/start',async route=>{await gate;await route.continue();});
 await page.locator('.practice-heading').waitFor({timeout:8000});await page.waitForTimeout(700);assert.equal(starts,1);assert.ok((await page.locator('.practice-status').textContent()).includes('0.000초'));assert.equal(await page.evaluate(()=>window.__streams.length),1);assert.equal(await page.evaluate(()=>window.__poseCreates),1);assert.equal(await page.evaluate(()=>window.__poseCloses),0);release();
 await page.waitForFunction(()=>document.querySelector('.practice-status')?.textContent.includes('실시간 평가 중'),{},{timeout:20000});assert.equal(starts,1);
 await page.waitForTimeout(1400);assert.ok(poseMessages.length>=5);assert.equal(poseMessages[0].playback_time_sec,0);
 for(const msg of poseMessages){assert.deepEqual(Object.keys(msg).sort(),['playback_time_sec','type','world_landmarks']);if(msg.world_landmarks!==null){assert.equal(msg.world_landmarks.length,33);assert.ok(msg.world_landmarks.every(row=>row.length===3&&row.every(Number.isFinite)));}}
 for(let i=1;i<poseMessages.length;i++)assert.ok(poseMessages[i].playback_time_sec>poseMessages[i-1].playback_time_sec);
 await page.waitForFunction(()=>{const value=Number(document.querySelector('.score strong').textContent);return Number.isInteger(value)&&value>=0&&value<=100;});
 await page.waitForFunction(()=>{const expected=window.__lastEval.displayed_feedback;return (expected==='NONE'?['—','Good']:[expected]).includes(document.querySelector('.feedback-panel strong').textContent);});
 assert.ok(await page.locator('.reference-panel img').evaluate(img=>img.complete&&img.naturalWidth>0));
 const sessionId=evaluationResults[0].session_id;
 await page.getByRole('button',{name:'일시 정지',exact:true}).click();await page.getByRole('button',{name:'이어하기',exact:true}).waitFor();
 const frozen=await page.locator('.practice-status').textContent(),sent=poseMessages.length;await page.waitForTimeout(400);assert.equal(await page.locator('.practice-status').textContent(),frozen);assert.equal(poseMessages.length,sent);
 assert.equal(await page.evaluate(()=>window.__streams[0].getTracks()[0].readyState),'live');
 await page.getByRole('button',{name:'이어하기',exact:true}).click();await page.waitForFunction(()=>document.querySelector('.practice-status')?.textContent.includes('실시간 평가 중'));await page.waitForTimeout(300);assert.ok(poseMessages.length>sent);
 const beforeReset=poseMessages.length;await page.getByRole('button',{name:'처음부터 다시',exact:true}).click();await page.waitForFunction(()=>document.querySelector('.practice-status')?.textContent.includes('실시간 평가 중'));await page.waitForTimeout(250);
 assert.ok(poseMessages.slice(beforeReset).some(p=>p.playback_time_sec===0));assert.ok(!evaluationResults.some(r=>r.code==='playback_reset_required'));
 await page.evaluate(()=>window.__noPose=true);await page.waitForFunction(()=>window.__lastEval?.pose_detected===false);
 assert.ok(poseMessages.some(m=>m.world_landmarks===null));
 let loseEnd=true;
 await page.route(`**/api/sessions/${sessionId}/end`,async route=>{if(loseEnd){loseEnd=false;await route.fetch();await route.abort();}else await route.continue();});
 await page.getByRole('button',{name:'연습 종료 · 기록 저장',exact:true}).click();await page.getByRole('button',{name:'같은 요청 다시 시도',exact:true}).waitFor();
 assert.ok(!(await page.locator('body').textContent()).includes('기록 저장 완료'));
 await page.getByRole('button',{name:'같은 요청 다시 시도',exact:true}).click();await page.getByText('기록 저장 완료',{exact:true}).waitFor();
 const ended=await (await page.request.post(`/api/sessions/${sessionId}/end`)).json();assert.equal(ended.status,'ended');assert.ok(ended.record.valid_sample_count>0);assert.ok(Number.isFinite(ended.record.mean_raw_final_score));
 const duplicate=await (await page.request.post(`/api/sessions/${sessionId}/end`)).json();assert.deepEqual(duplicate.record,ended.record);
 const records=await (await page.request.get(`/api/records/${guest}`)).json();assert.equal(records.filter(r=>r.session_id===sessionId).length,1);
 assert.equal(await page.locator('.result-score strong').textContent(),(ended.record.mean_raw_final_score*100).toFixed(1));report.end_record=ended.record;
 // No-Pose End and duplicate End use a separate session in this temporary DB only.
 const empty=await (await page.request.post('/api/sessions/start',{data:{song_id:'033_D00_001_F',guest_id:guest,nickname:'null 검증',playback_speed:1,mode:'video'}})).json();
 const emptyEnd=await (await page.request.post(`/api/sessions/${empty.session_id}/end`)).json();assert.equal(emptyEnd.record.mean_raw_final_score,null);assert.equal(emptyEnd.record.valid_sample_count,0);report.null_record=emptyEnd.record;
 await page.waitForTimeout(300);assert.ok(await page.evaluate(()=>window.__streams.every(s=>s.getTracks().every(t=>t.readyState==='ended'))));assert.equal(await page.evaluate(()=>window.__poseCloses),1);assert.deepEqual(errors,[]);
 report.controls={pause:true,resume:true,reset_zero:true,end:true,duplicate_end:true};
 await page.evaluate(()=>{window.__noPose=false;window.__reachEnd=false;});
 await page.getByRole('button',{name:'다시 연습',exact:true}).click();await page.getByRole('button',{name:'다음 →'}).click();
 await page.waitForFunction(()=>document.querySelector('.pose-debug')?.textContent.includes('Image 33/33'),{},{timeout:120000});
 await page.keyboard.press('Space');await page.waitForFunction(()=>document.querySelector('.practice-status')?.textContent.includes('실시간 평가 중'),{},{timeout:20000});
 await page.evaluate(()=>window.__reachEnd=true);await page.getByText('기록 저장 완료',{exact:true}).waitFor();
 const afterAuto=await (await page.request.get(`/api/records/${guest}`)).json();assert.equal(afterAuto.length,3);assert.equal(new Set(afterAuto.map(r=>r.session_id)).size,3);report.automatic_end=true;
 await page.waitForTimeout(200);assert.ok(await page.evaluate(()=>window.__streams.every(s=>s.getTracks().every(t=>t.readyState==='ended'))));assert.equal(await page.evaluate(()=>window.__poseCloses),2);

 // Verify rejection of an Origin outside the temporary server allowlist.
 await page.goto('http://127.0.0.1:5175');
 const rejected=await page.evaluate(()=>new Promise(resolve=>{const ws=new WebSocket('ws://127.0.0.1:18010/ws/sessions/origin-probe');ws.onerror=()=>resolve(true);ws.onopen=()=>{ws.close();resolve(false);};}));
 assert.equal(rejected,true);report.origin_rejection=true;
 report.pass=true;report.guest_id=guest;report.pose_count=poseMessages.length;report.first_pose=poseMessages[0];report.first_result=evaluationResults[0];report.session_start_count=starts;report.readiness_calls=readinessCalls;
 console.log('PASS Practice3-1: actual Pause/Resume/Reset first-zero, End response-loss recovery, one SQLite record per session, null score, camera cleanup, songs/readiness/evaluation preserved');
}finally{
 await context?.close();
 for(const child of children)child.kill('SIGTERM');
 await writeFile(join(run,'servers.log'),logs.join(''));await writeFile(join(run,'report.json'),JSON.stringify(report,null,2));
 console.log('Verification artifacts:',run);
}
