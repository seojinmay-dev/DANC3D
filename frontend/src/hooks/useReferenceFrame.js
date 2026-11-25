import {useEffect,useState} from 'react';
import {referenceFrameUrl} from '../services/api.js';
import {createReferencePlayback} from '../services/referencePlayback.js';
export default function useReferenceFrame(songId,readTime) {
  const [src,setSrc]=useState(''),[error,setError]=useState(''),[retry,setRetry]=useState(0);
  useEffect(()=>{
    setSrc('');setError('');
    if(!songId)return;
    return createReferencePlayback({readTime,urlFor:time=>referenceFrameUrl(songId,time),onFrame:setSrc,onError:setError});
  },[songId,readTime,retry]);
  return {src,error,retry:()=>setRetry(n=>n+1)};
}
