// Display only: the caller supplies the existing content clock, already speed-scaled.
export function createReferencePlayback({readTime,urlFor,onFrame,onError,
  fetchFrame=fetch,createUrl=blob=>URL.createObjectURL(blob),revokeUrl=url=>URL.revokeObjectURL(url),
  decode=async url=>{const image=new Image();image.src=url;await image.decode();},
  schedule=requestAnimationFrame,cancel=cancelAnimationFrame}) {
  const cache=new Map(),pendingUrls=new Set();
  const release=url=>{pendingUrls.delete(url);revokeUrl(url);};
  let disposed=false,failed=false,task=null,raf,lastTime=null,generation=0,displayed=-1;
  const frameAt=time=>Math.max(0,Math.ceil(time*30-.5)); // nearest 30fps slot, ties go left
  function trim(target) {
    for(const [frame,url] of cache) {
      if(cache.size<=12)break;
      if(frame!==displayed&&frame!==target){cache.delete(frame);revokeUrl(url);}
    }
  }
  function show(target) {
    // Never display future prefetches or move backwards within a clock generation.
    const frame=[...cache.keys()].filter(f=>f<=target&&f>=displayed).sort((a,b)=>b-a)[0];
    if(frame!==undefined&&frame!==displayed){displayed=frame;onFrame(cache.get(frame));}
  }
  function load(frame) {
    const controller=new AbortController(),epoch=generation;
    const current={controller,epoch};task=current;
    const timeout=setTimeout(()=>controller.abort(),5000);
    let url;
    (async()=>{
      try {
        const response=await fetchFrame(urlFor(frame/30),{signal:controller.signal});
        if(!response.ok)throw Error(`기준 JPG를 불러올 수 없습니다. (HTTP ${response.status})`);
        const blob=await response.blob();
        if(disposed||epoch!==generation||controller.signal.aborted)return;
        url=createUrl(blob);pendingUrls.add(url);
        await new Promise((resolve,reject)=>{
          const aborted=()=>reject(Object.assign(Error('aborted'),{name:'AbortError'}));
          controller.signal.addEventListener('abort',aborted,{once:true});
          Promise.resolve().then(()=>decode(url)).then(resolve,reject).finally(()=>controller.signal.removeEventListener('abort',aborted));
        });
        if(disposed||epoch!==generation||controller.signal.aborted)return;
        cache.set(frame,url);pendingUrls.delete(url);url=null;
        const target=frameAt(readTime());trim(target);show(target);onError('');
      }catch(e){
        if(!disposed&&epoch===generation){failed=true;onError(e.name==='AbortError'?'기준 JPG 요청 시간이 초과되었습니다.':e.message);}
      }finally{
        if(url&&pendingUrls.has(url))release(url);clearTimeout(timeout);if(task===current)task=null;
      }
    })();
  }
  function tick() {
    if(disposed)return;
    const time=readTime(),target=frameAt(time),moving=lastTime!==null&&time>lastTime;
    if(lastTime!==null&&time<lastTime){
      generation++;task?.controller.abort();failed=false;displayed=-1;
    }
    lastTime=time;show(target);trim(target);
    if(!task&&!failed){
      // No queue: missing current wins; prefetch only while the clock advances.
      const next=[target,...(moving?[target+1,target+2]:[])].find(f=>!cache.has(f));
      if(next!==undefined)load(next);
    }
    raf=schedule(tick);
  }
  tick();
  return ()=>{
    disposed=true;generation++;cancel(raf);task?.controller.abort();
    for(const url of pendingUrls)release(url);
    for(const url of cache.values())revokeUrl(url);cache.clear();
  };
}
