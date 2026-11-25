import {useEffect,useState} from 'react';
import {fetchRecords,fetchSongRecords} from '../services/api.js';
export default function useRecords(guestId,songId,identityToken=null) {
  const [attempt,setAttempt]=useState(0),[state,setState]=useState({loading:true,data:null,error:''});
  useEffect(()=>{
    let disposed=false;const controller=new AbortController();
    setState({loading:true,data:null,error:''});
    // Defer actual fetch so StrictMode's discarded effect does not issue a request.
    const timer=setTimeout(async()=>{
      try{
        const data=await (songId?fetchSongRecords(guestId,songId,{signal:controller.signal}):fetchRecords(guestId,{signal:controller.signal}));
        if(!disposed)setState({loading:false,data,error:''});
      }catch(e){if(!disposed)setState({loading:false,data:null,error:e.name==='TimeoutError'?'기록 조회 시간이 초과되었습니다. 다시 시도해주세요.':e.message});}
    },0);
    return()=>{disposed=true;clearTimeout(timer);controller.abort();};
  },[guestId,songId,attempt,identityToken]);
  return {...state,retry:()=>setAttempt(n=>n+1)};
}
