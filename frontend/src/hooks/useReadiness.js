import {useCallback,useEffect,useRef,useState} from 'react';
import {requestReadiness} from '../services/api.js';
import {createCalibrationCapture} from '../services/avatarCalibration.js';
// Single in-flight request, latest frame only, maximum 5 requests/sec, 3s timeout.
export default function useReadiness(latestFrameRef,cameraStatus,navigate,{requireCalibration=false,onCalibration}={}) {
  const [phase,setPhase]=useState('idle'),[result,setResult]=useState(null),[error,setError]=useState(''),[count,setCount]=useState(5),[duration,setDuration]=useState(5);
  const attemptRef=useRef(null),phaseRef=useRef('idle'),cancelRef=useRef(()=>{});
  const captureRef=useRef(null);
  if(!captureRef.current)captureRef.current=createCalibrationCapture();
  const begin=useCallback(()=>{
    cancelRef.current();attemptRef.current=crypto.randomUUID();phaseRef.current='waiting';
    captureRef.current.reset();onCalibration?.(null);
    setResult(null);setError('');setPhase('waiting');
  },[onCalibration]);
  const reset=useCallback(()=>{cancelRef.current();captureRef.current.reset();onCalibration?.(null);attemptRef.current=null;phaseRef.current='idle';setPhase('idle');setResult(null);setError('');},[onCalibration]);
  useEffect(()=>{
    const key=e=>{
      if(e.code==='Space'&&!['INPUT','TEXTAREA'].includes(e.target.tagName)){
        e.preventDefault();if(!e.repeat&&phaseRef.current==='idle')begin();
      }
    };
    window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key);
  },[begin]);
  useEffect(()=>{
    if(phase!=='waiting'||cameraStatus!=='running')return;
    const attempt=attemptRef.current;
    let stopped=false,timer,controller,lastTimestamp=-1;
    const cancel=()=>{stopped=true;clearTimeout(timer);controller?.abort();};
    cancelRef.current=cancel;
    const poll=async()=>{
      if(stopped)return;
      const frame=latestFrameRef.current;
      if(!frame||frame.timestamp_ms===lastTimestamp){timer=setTimeout(poll,100);return;}
      lastTimestamp=frame.timestamp_ms;controller=new AbortController();
      const timeout=setTimeout(()=>controller.abort(),3000);
      try{
        const response=await requestReadiness(attempt,frame,{signal:controller.signal});
        if(stopped||attemptRef.current!==attempt||phaseRef.current!=='waiting')return;
        if(response.attempt_id!==attempt){timer=setTimeout(poll,200);return;}
        setResult(response);
        const calibration=captureRef.current.accept(frame,response);
        if(response.ready&&(!requireCalibration||calibration)){onCalibration?.(calibration);phaseRef.current='countdown';setDuration(response.countdown_duration_sec);setCount(Math.ceil(response.countdown_duration_sec));setPhase('countdown');return;}
      }catch(e){
        if(stopped||attemptRef.current!==attempt)return;
        setError(e.name==='AbortError'?'준비 판정 응답 시간이 초과되었습니다. 다시 시도해주세요.':e instanceof TypeError?'준비 판정 서버에 연결할 수 없습니다.':e.message);
        phaseRef.current='error';setPhase('error');return;
      }finally{clearTimeout(timeout);}
      if(!stopped)timer=setTimeout(poll,200);
    };
    poll();return cancel;
  },[phase,cameraStatus,latestFrameRef,requireCalibration,onCalibration]);
  useEffect(()=>{
    if(phase!=='countdown')return;
    const started=performance.now();
    const timer=setInterval(()=>{const elapsed=(performance.now()-started)/1000;setCount(Math.max(1,Math.ceil(duration-elapsed)));if(elapsed>=duration){clearInterval(timer);phaseRef.current='finished';navigate('practice');}},50);
    return()=>clearInterval(timer);
  },[phase,duration,navigate]);
  useEffect(()=>()=>{cancelRef.current();attemptRef.current=null;},[]);
  return {phase,result,error,count,duration,begin,reset};
}
