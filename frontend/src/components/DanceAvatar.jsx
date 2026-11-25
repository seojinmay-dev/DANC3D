import {useEffect, useRef, useState} from 'react';
import {createScene} from '../retargeting/three/scene.js';
import {loadCharacter} from '../retargeting/three/character.js';
import {createFullBodyRetargeter} from '../retargeting/retarget/fullBodyRetargeter.js';
import {bakeReferenceAnimation} from '../retargeting/referenceAnimation.js';
import {createLiveAvatarDriver} from '../retargeting/liveAvatarDriver.js';
import {createEvaluationSkeleton3D} from '../retargeting/three/evaluationSkeleton3D.js';
import {createPoseRecorder, poseRecordingEnabled} from '../services/poseRecorder.js';

export default function DanceAvatar({poseFrameRef, modelUrl, isReference = false,
  songId, readTime, currentTime = 0, calibrationFrame, onReady, evaluationRef}) {
  const mountRef = useRef(null);
  const timeRef = useRef(null), readyRef = useRef(onReady);
  timeRef.current = readTime || (() => currentTime);
  readyRef.current = onReady;
  const [status, setStatus] = useState('3D 캐릭터 준비 중…');
  const [error, setError] = useState('');

  useEffect(() => {
    const container = mountRef.current;
    const {scene, camera, renderer, controls, dispose} = createScene(container);
    const controller = new AbortController();
    let raf, playback, live, overlay, previousRenderTime;
    let ready = false;
    // Developer recording (?recordPose): Shift+R downloads the landmark stream.
    const recorder = !isReference && poseRecordingEnabled() ? createPoseRecorder(calibrationFrame) : null;
    const onKey = event => {
      if (recorder && event.shiftKey && event.key.toLowerCase() === 'r') console.info('pose recording frames:', recorder.download());
    };
    if (recorder) window.addEventListener('keydown', onKey);
    setStatus('3D 캐릭터 준비 중…'); setError(''); readyRef.current?.(false);
    const initialize = async () => {
      try {
        const character = await loadCharacter(scene, modelUrl);
        if (controller.signal.aborted) { dispose(); return; }
        if (isReference) {
          const retargeter = createFullBodyRetargeter(character, {pelvisPositionMode: 'world', smoothing: false});
          const response = await fetch(`/references/${encodeURIComponent(songId)}.json`, {signal:controller.signal});
          if (!response.ok) throw Error(`기준 안무 3D 데이터를 불러올 수 없습니다. (HTTP ${response.status})`);
          const data = await response.json();
          if (controller.signal.aborted) return;
          setStatus('기준 안무 움직임 준비 중…');
          playback = await bakeReferenceAnimation({data, character, retargeter, signal:controller.signal});
        } else {
          live = createLiveAvatarDriver(character, calibrationFrame);
          if (!live) throw Error('T자 보정 정보가 없습니다. 준비 화면에서 다시 시작해주세요.');
          // Evaluation colors replace the default skeleton helper on the user's character.
          if (evaluationRef) {
            overlay = createEvaluationSkeleton3D(scene, character.model);
            character.skeletonHelper.visible = false;
          }
        }
        if (controller.signal.aborted) return;
        ready = true; setStatus(''); readyRef.current?.(true);
      } catch (e) {
        if (!controller.signal.aborted) { setStatus(''); setError(e.message); }
      }
    };
    const animate = now => {
      if (controller.signal.aborted) return;
      const deltaSeconds = previousRenderTime === undefined ? 1/60 : (now - previousRenderTime)/1000;
      previousRenderTime = now;
      if (ready) {
        if (isReference) playback.apply(timeRef.current());
        // Camera frames are stamped with performance.now(), so the driver uses that clock.
        else {
          recorder?.push(poseFrameRef?.current);
          live.update(poseFrameRef?.current, performance.now(), deltaSeconds);
          overlay?.update(evaluationRef.current);
        }
        controls.update(); renderer.render(scene, camera);
      }
      raf = requestAnimationFrame(animate);
    };
    initialize(); raf = requestAnimationFrame(animate);
    return () => {
      controller.abort(); cancelAnimationFrame(raf); overlay?.dispose(); dispose();
      if (recorder) window.removeEventListener('keydown', onKey);
      renderer.domElement.remove();
    };
  }, [modelUrl, isReference, songId, calibrationFrame, poseFrameRef, evaluationRef]);

  return <div className="dance-avatar" style={{width:'100%', height:'100%', position:'relative', overflow:'hidden'}}>
    <div className="dance-avatar-mount" ref={mountRef} style={{width:'100%', height:'100%'}}/>
    {(status || error) && <p role={error ? 'alert' : 'status'} style={{position:'absolute', inset:'45% 12px auto', textAlign:'center', color:'#1a1f2c'}}>{error || status}</p>}
  </div>;
}
