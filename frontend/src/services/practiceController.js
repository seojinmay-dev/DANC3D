import {evaluatePose,streamEvaluation} from './continuousEvaluation.js';
// All REST controls drain the sole pending evaluation before changing server state.
export function createPracticeController({socket,session,clock,camera,signal,control,publish,onEnded,lastTimestamp}) {
  let mode='running',generation=0,loop=Promise.resolve(),busy=null,ended=false,failedAction=null;
  const report=update=>{if(!signal.aborted)publish(update);};
  function run(zero=false){
    const token=++generation;mode='running';
    loop=(async()=>{
      if(zero){
        report({phase:'first',result:null});let previous=camera.latestFrameRef.current?.timestamp_ms??-1;
        while(!signal.aborted&&mode==='running'&&token===generation){
          const frame=camera.latestFrameRef.current;
          if(!frame||frame.timestamp_ms<=previous){await new Promise(r=>setTimeout(r,20));continue;}
          previous=frame.timestamp_ms;
          const result=await evaluatePose(socket,session.session_id,{...frame,playback_time_sec:0},{signal});
          if(mode!=='running'||token!==generation)return;
          report({result});lastTimestamp=frame.timestamp_ms;
          if(result.evaluation_performed&&result.pose_detected&&result.valid_user_bone_count>0&&result.aligned_reference_index>=0)break;
        }
        if(mode!=='running'||signal.aborted)return;
      }
      clock.start();report({phase:'running',error:'',failedAction:null});
      await streamEvaluation(socket,session.session_id,camera.latestFrameRef,{signal,clock,duration:session.duration_sec,lastTimestamp,isCameraRunning:()=>camera.statusRef.current==='running',shouldContinue:()=>mode==='running'&&token===generation,onResult:result=>{lastTimestamp=camera.latestFrameRef.current?.timestamp_ms??lastTimestamp;if(mode==='running'&&token===generation)report({result});}});
      if(mode==='running'&&!signal.aborted&&clock.read()>=session.duration_sec){mode='completed';clock.pause();setTimeout(()=>{if(!signal.aborted)act('end');},0);}
    })().catch(e=>{if(signal.aborted)return;mode='error';clock.pause();socket.close();report({phase:'error',error:e.message});});
  }
  function act(action){
    if(ended||signal.aborted)return Promise.resolve();
    if(busy)return busy;
    // A failed/uncertain control must be reconciled by repeating that same action.
    if(failedAction&&action!==failedAction&&action!=='end')return Promise.resolve();
    mode='controlling';++generation;clock.pause();report({busy:action,error:''});
    busy=(async()=>{
      await loop;if(signal.aborted)return;
      try{
        if(['resume','reset'].includes(action)&&socket.readyState!==1)throw Error('WebSocket이 닫혀 이어할 수 없습니다. 종료하여 기록을 저장해주세요.');
        const response=await control(session.session_id,action,{signal});
        if(signal.aborted)return;
        failedAction=null;
        if(action==='end'){ended=true;mode='ended';socket.close();report({phase:'ended',busy:null,failedAction:null});onEnded(response.record);}
        else if(action==='pause'){mode='paused';report({phase:'paused',busy:null,failedAction:null});}
        else if(action==='reset'){
          clock.reset();report({phase:'first',result:null,busy:null,failedAction:null});run(true);
        }else{lastTimestamp=camera.latestFrameRef.current?.timestamp_ms??lastTimestamp;report({busy:null,failedAction:null});run();}
      }catch(e){if(!signal.aborted){failedAction=action;mode='uncertain';report({phase:'uncertain',error:`${e.message} 서버 상태가 불확실합니다. 같은 요청을 다시 시도하거나 종료해주세요.`,busy:null,failedAction:action});}}
    })().finally(()=>{busy=null;});
    return busy;
  }
  return {start:(zero=false)=>run(zero),act,dispose(){mode='disposed';++generation;clock.pause();socket.close();}};
}
