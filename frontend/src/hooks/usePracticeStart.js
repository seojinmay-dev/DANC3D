import {useEffect,useRef,useState} from 'react';
import {createSession,controlSession} from '../services/api.js';
import {createPracticeController} from '../services/practiceController.js';
import {connectEvaluation} from '../services/evaluationSocket.js';
export default function usePracticeStart({song,speed,guestId,nickname,camera,clock,onEnded,enabled=true}) {
  const [attempt,setAttempt]=useState(0),[state,setState]=useState({phase:'creating',session:null,result:null,error:'',canRetry:false,busy:null,failedAction:null});
  const sessionRef=useRef(null),creationRef=useRef(null),runtimeRef=useRef(null),endRef=useRef(onEnded);
  endRef.current=onEnded;
  useEffect(()=>{
    if(!enabled){setState(s=>({...s,phase:'preparingAvatar'}));return;}
    let disposed=false,timer,socket,runtime;
    const controller=new AbortController();
    const publish=update=>{if(!disposed)setState(s=>({...s,...update}));};
    const initialize=async()=>{
      try{
        publish({phase:sessionRef.current?'connecting':'creating',error:'',canRetry:false});
        if(!nickname.trim()){const e=Error('홈에서 닉네임을 입력한 뒤 다시 준비해주세요.');e.safeToRetry=true;throw e;}
        if(!sessionRef.current){
          if(!creationRef.current)creationRef.current=createSession({song_id:song.song_id,guest_id:guestId,nickname:nickname.trim(),playback_speed:speed,mode:'video'},{signal:controller.signal});
          sessionRef.current=await creationRef.current;
        }
        if(disposed)return;
        publish({session:sessionRef.current,phase:'connecting'});
        socket=await connectEvaluation(sessionRef.current.session_id,{signal:controller.signal});
        if(disposed){socket.close();return;}
        runtime=createPracticeController({socket,session:sessionRef.current,clock,camera,signal:controller.signal,control:controlSession,publish,onEnded:record=>endRef.current(record)});
        runtimeRef.current=runtime;publish({controlReady:true});runtime.start(true);
      }catch(e){
        if(disposed)return;clock.pause();socket?.close();
        if(sessionRef.current&&!runtime){
          const closedSocket=socket||{readyState:3,close(){}};
          runtime=createPracticeController({socket:closedSocket,session:sessionRef.current,clock,camera,signal:controller.signal,control:controlSession,publish,onEnded:record=>endRef.current(record)});
          runtimeRef.current=runtime;publish({controlReady:true});
        }
        if(e.safeToRetry)creationRef.current=null;
        publish({phase:'error',error:e.message,canRetry:!socket&&(!!sessionRef.current||!!e.safeToRetry)});
      }
    };
    timer=setTimeout(initialize,0);
    return()=>{disposed=true;clearTimeout(timer);controller.abort();runtime?.dispose();runtimeRef.current=null;socket?.close();clock.pause();};
  },[attempt,song.song_id,speed,guestId,nickname,camera.latestFrameRef,clock,enabled]);
  return {...state,retry:()=>setAttempt(n=>n+1),control:action=>runtimeRef.current?.act(action)};
}
