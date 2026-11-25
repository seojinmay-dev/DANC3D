import {useCallback,useRef} from 'react';
import usePoseCamera from '../hooks/usePoseCamera.js';
import Ready from '../pages/Ready.jsx';
import Practice from '../pages/Practice.jsx';
export default function PracticeFlow({page,...props}) {
  const camera=usePoseCamera();
  const calibrationRef=useRef(null);
  const saveCalibration=useCallback(frame=>{calibrationRef.current=frame;},[]);
  return page==='ready'?<Ready {...props} camera={camera} onCalibration={saveCalibration}/>:<Practice {...props} camera={camera} calibrationFrame={calibrationRef.current}/>;
}
