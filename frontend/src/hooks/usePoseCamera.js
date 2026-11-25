import { useCallback, useEffect, useRef, useState } from 'react';
import { createPoseDetector, drawPose, poseFrame } from '../services/pose.js';

const cameraError = e => ({
  NotAllowedError: '카메라 권한이 거부되었습니다.',
  NotFoundError: '카메라를 찾을 수 없습니다.',
  NotReadableError: '카메라가 다른 앱에서 사용 중입니다.'
}[e.name] || `카메라 Pose 오류: ${e.message}`);

let permissionQueue = Promise.resolve();

export default function usePoseCamera() {
  const displayRef = useRef(null),
    canvasRef = useRef(null),
    streamRef = useRef(null),
    latestFrameRef = useRef(null),
    contentTimeReaderRef = useRef(null);
  const statusRef = useRef('initializing'),
    overlayRef = useRef(null);

  const [attempt, setAttempt] = useState(0),
    [status, setStatus] = useState('initializing'),
    [error, setError] = useState(''),
    [debug, setDebug] = useState(null);

  statusRef.current = status;

  const videoRef = useCallback(node => {
    if (displayRef.current) displayRef.current.srcObject = null;
    displayRef.current = node;
    if (node && streamRef.current) {
      node.srcObject = streamRef.current;
      node.play().catch(() => {});
    }
  }, []);

  useEffect(() => {
    let disposed = false,
      stream,
      detector,
      raf,
      timer,
      lastTime = -1,
      lastTimestamp = -1;

    const source = document.createElement('video');
    source.muted = true;
    source.playsInline = true;

    const release = () => {
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach(t => t.stop());
      detector?.close();
      detector = null;
      source.srcObject = null;
      if (streamRef.current === stream) {
        streamRef.current = null;
        if (displayRef.current?.srcObject === stream) displayRef.current.srcObject = null;
      }
    };

    const clear = () => {
      latestFrameRef.current = null;
      const c = canvasRef.current;
      c?.getContext('2d').clearRect(0, 0, c.width, c.height);
    };

    const fail = e => {
      release();
      if (!disposed) {
        clear();
        setDebug(null);
        setError(cameraError(e));
        setStatus('error');
      }
    };

    setStatus('initializing');
    setError('');
    setDebug(null);
    clear();

    const tick = now => {
      if (disposed) return;
      try {
        if (source.readyState >= 2 && source.currentTime !== lastTime) {
          lastTime = source.currentTime;
          const timestamp = Math.max(performance.now(), lastTimestamp + 0.001);
          lastTimestamp = timestamp;
          const contentTime = contentTimeReaderRef.current?.(timestamp);
          const frame = poseFrame(detector.detectForVideo(source, timestamp), source, timestamp);
          if (Number.isFinite(contentTime)) frame.playback_time_sec = contentTime;
          latestFrameRef.current = frame;

          if (canvasRef.current) (overlayRef.current || drawPose)(canvasRef.current, frame);
          const next = {
            imageCount: frame.image_landmarks?.length || 0,
            worldCount: frame.world_landmarks?.length || 0,
            detected: frame.pose_detected
          };
          setDebug(previous =>
            previous &&
            previous.imageCount === next.imageCount &&
            previous.worldCount === next.worldCount &&
            previous.detected === next.detected
              ? previous
              : next
          );
        }
        raf = requestAnimationFrame(tick);
      } catch (e) {
        fail(e);
      }
    };

    const initialize = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw Error('HTTPS 또는 localhost 환경이 필요합니다.');
        const request = permissionQueue.then(async () => {
          if (disposed) return null;
          const acquired = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false
          });
          if (disposed) {
            acquired.getTracks().forEach(t => t.stop());
            return null;
          }
          return acquired;
        });
        permissionQueue = request.then(() => undefined, () => undefined);
        stream = await request;
        if (disposed) {
          release();
          return;
        }
        streamRef.current = stream;
        source.srcObject = stream;
        await source.play();
        if (disposed) {
          release();
          return;
        }
        if (displayRef.current) {
          displayRef.current.srcObject = stream;
          await displayRef.current.play();
        }
        if (disposed) {
          release();
          return;
        }
        detector = await createPoseDetector();
        if (disposed) {
          release();
          return;
        }
        setStatus('running');
        raf = requestAnimationFrame(tick);
      } catch (e) {
        fail(e);
      }
    };

    timer = setTimeout(initialize, 0);
    return () => {
      disposed = true;
      clearTimeout(timer);
      release();
      clear();
    };
  }, [attempt]);

  return {
    overlayRef,
    videoRef,
    canvasRef,
    latestFrameRef,
    contentTimeReaderRef,
    statusRef,
    status,
    error,
    debug,
    retry: () => setAttempt(n => n + 1)
  };
}
