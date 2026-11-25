export function validateWorld(world) {
  if(world===null)return;
  if(!Array.isArray(world)||world.length!==33||world.some(row=>!Array.isArray(row)||row.length!==3||row.some(v=>typeof v!=='number'||!Number.isFinite(v))))throw Error('World Landmark는 유한 숫자의 33×3 배열 또는 null이어야 합니다.');
}
function delay(ms,signal) {
  return new Promise((resolve,reject)=>{
    const abort=()=>{clearTimeout(timer);reject(new DOMException('Cancelled','AbortError'));};
    const timer=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve();},ms);
    if(signal.aborted)abort();else signal.addEventListener('abort',abort,{once:true});
  });
}
export function evaluatePose(socket,sessionId,frame,{signal,timeoutMs=3000}={}) {
  validateWorld(frame.world_landmarks);
  const playback=frame.playback_time_sec;
  if(!Number.isFinite(playback)||playback<0)throw Error('콘텐츠 시간이 올바르지 않습니다.');
  return new Promise((resolve,reject)=>{
    const cleanup=()=>{clearTimeout(timer);socket.removeEventListener('message',message);socket.removeEventListener('close',closed);socket.removeEventListener('error',closed);signal.removeEventListener('abort',abort);};
    const fail=e=>{cleanup();reject(e);};const closed=()=>fail(Error('실시간 평가 연결이 종료되었습니다. 다시 준비해주세요.'));
    const abort=()=>fail(new DOMException('Cancelled','AbortError'));
    const message=event=>{
      try{
        const data=JSON.parse(event.data);if(data.type==='error')throw Error(`평가 오류: ${data.code} (${data.status})`);
        if(data.session_id!==sessionId||data.playback_time_sec!==playback||data.state!=='running'||typeof data.evaluation_performed!=='boolean'||typeof data.pose_detected!=='boolean'||(data.displayed_score!==null&&!Number.isFinite(data.displayed_score))||typeof data.displayed_feedback!=='string')throw Error('실시간 평가 응답 계약이 일치하지 않습니다.');
        cleanup();resolve(data);
      }catch(e){fail(e);}
    };
    const timer=setTimeout(()=>fail(Error('평가 응답이 3초 이상 지연되었습니다. 전송을 중단합니다.')),timeoutMs);
    socket.addEventListener('message',message);socket.addEventListener('close',closed,{once:true});socket.addEventListener('error',closed,{once:true});signal.addEventListener('abort',abort,{once:true});
    if(signal.aborted){abort();return;}
    try{socket.send(JSON.stringify({type:'pose',playback_time_sec:playback,world_landmarks:frame.world_landmarks}));}catch(e){fail(e);}
  });
}
export async function streamEvaluation(socket,sessionId,latestFrameRef,{signal,clock,duration,lastTimestamp,onResult,timeoutMs=3000,isCameraRunning=()=>true,shouldContinue=()=>true}) {
  let previousTimestamp=lastTimestamp,previousPlayback=0;
  while(!signal.aborted&&shouldContinue()){
    if(!isCameraRunning())throw Error('카메라 추론이 중단되었습니다. 다시 준비해주세요.');
    if(socket.readyState!==1)throw Error('WebSocket 연결이 종료되었습니다.');
    if(clock.read()>=duration)return;
    const frame=latestFrameRef.current;
    if(!frame||frame.timestamp_ms<=previousTimestamp||!Number.isFinite(frame.playback_time_sec)||frame.playback_time_sec<=previousPlayback||frame.playback_time_sec>=duration){await delay(20,signal);continue;}
    // The one pending request retains its capture time; no queue of old frames is built.
    previousTimestamp=frame.timestamp_ms;previousPlayback=frame.playback_time_sec;
    const result=await evaluatePose(socket,sessionId,frame,{signal,timeoutMs});
    if(signal.aborted)return;
    onResult(result);
  }
}
