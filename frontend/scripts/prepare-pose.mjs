import {cp,mkdir,readFile,open,link,stat,unlink} from 'node:fs/promises';
import {realpathSync} from 'node:fs';
import {createHash,randomUUID} from 'node:crypto';
import {basename,dirname,join} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=new URL('../',import.meta.url);
// Pose Landmarker Full (float16), pinned by generation so "latest" changes never alter the bytes.
// Keep in sync with app/services/pose_model.py (checked by tests/test_pose_model.py).
export const MODEL_URL='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task?generation=1682642787774579';
export const MODEL_SHA256='4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad';
export const MODEL_SIZE=9398198;
export const TIMEOUT_MS=60000;
const fileSha256=async path=>createHash('sha256').update(await readFile(path)).digest('hex');
const isFile=path=>stat(path).then(s=>s.isFile(),()=>false);

// An existing model is used as-is; a size/hash difference only warns and is never deleted or replaced.
async function useExisting(target,expected) {
  const {size}=await stat(target);
  if(size!==expected.size) console.warn(`Warning: ${target} is ${size} bytes, expected ${expected.size}; using the existing file as-is.`);
  else if(await fileSha256(target)!==expected.sha256) console.warn(`Warning: ${target} SHA-256 differs from ${expected.sha256}; using the existing file as-is.`);
  return {path:target,downloaded:false};
}

// Downloads only when the model is missing. The verified unique temp file is hard-linked into place:
// link() fails with EEXIST instead of overwriting a final file another process created meanwhile.
// size/sha256 exist only so tests can use small fake files; callers keep the pinned defaults.
export async function ensureModel(target,{fetchImpl=globalThis.fetch,timeoutMs=TIMEOUT_MS,size=MODEL_SIZE,sha256=MODEL_SHA256}={}) {
  const expected={size,sha256};
  if(await isFile(target)) return useExisting(target,expected);
  let temp,file;
  try {
    await mkdir(dirname(target),{recursive:true});
    console.log(`MediaPipe model not found: ${target}\nDownloading ${MODEL_URL}`);
    temp=join(dirname(target),`${basename(target)}.${process.pid}.${randomUUID()}.part`);
    file=await open(temp,'wx');
    const response=await fetchImpl(MODEL_URL,{signal:AbortSignal.timeout(timeoutMs)});
    if(!response.ok) throw new Error(`HTTP ${response.status}`);
    const hash=createHash('sha256');let received=0;
    for await(const chunk of response.body) {
      received+=chunk.length;
      if(received>size) throw new Error(`download exceeds ${size} bytes`);
      hash.update(chunk);await file.write(chunk);
    }
    if(received!==size) throw new Error(`downloaded ${received} bytes, expected ${size}`);
    const digest=hash.digest('hex');
    if(digest!==sha256) throw new Error(`SHA-256 mismatch: got ${digest}, expected ${sha256}`);
    await file.sync();await file.close();file=undefined;
    try {await link(temp,target);}
    catch(error) {
      if(error.code!=='EEXIST') throw error;
      console.warn(`${target} was created by another process; keeping it.`);
      return await useExisting(target,expected);
    }
    console.log(`Saved MediaPipe model: ${target}`);
    return {path:target,downloaded:true};
  } catch(error) {
    throw new Error(`Could not prepare MediaPipe model ${target}: ${error.message}\nDownload it manually from ${MODEL_URL}\nand save it to that path (${size} bytes, SHA-256 ${sha256}).`,{cause:error});
  } finally {
    await file?.close().catch(()=>{});
    if(temp) await unlink(temp).catch(()=>{});
  }
}

async function main() {
  await ensureModel(fileURLToPath(new URL('../models/mediapipe/pose_landmarker_full.task',root)));
  await mkdir(new URL('public/mediapipe/',root),{recursive:true});
  await cp(new URL('node_modules/@mediapipe/tasks-vision/wasm/',root),new URL('public/mediapipe/wasm/',root),{recursive:true});
  await cp(new URL('../models/mediapipe/pose_landmarker_full.task',root),new URL('public/mediapipe/pose_landmarker_full.task',root));
  console.log('Local full model SHA256:',createHash('sha256').update(await readFile(new URL('public/mediapipe/pose_landmarker_full.task',root))).digest('hex'));
}

// Run only as a script (npm run prepare-pose), not when imported by tests or other modules.
const isScript=()=>{try {return import.meta.url===pathToFileURL(realpathSync(process.argv[1])).href;} catch {return false;}};
if(process.argv[1]&&isScript()) {
  await main().catch(error=>{console.error(error.message);process.exitCode=1;});
}
