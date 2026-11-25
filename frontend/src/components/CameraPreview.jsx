import {memo} from 'react';
import {Button} from './UI.jsx';
export default memo(function CameraPreview({camera}) {
  return <div className="camera-panel"><div className="camera-view"><video ref={camera.videoRef} autoPlay muted playsInline/><canvas ref={camera.canvasRef}/><span className="media-label">나의 웹캠 · 좌우 반전 표시</span>{camera.status==='initializing'&&<span className="camera-hint" role="status">카메라 연결 중...</span>}</div><div className="camera-controls"><span role="status">{camera.status==='running'?'카메라 사용 중':camera.status==='initializing'?'카메라 준비 중…':'카메라 꺼짐'}</span>{camera.error&&<Button secondary onClick={camera.retry}>카메라 다시 시도</Button>}</div>{camera.error&&<p className="camera-error" role="alert">{camera.error}</p>}</div>;
});
