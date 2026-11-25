import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { drawEvaluationPose } from '../services/evaluationSkeleton.js';
import LiveEvaluation from '../components/LiveEvaluation.jsx';
import { Button } from '../components/UI.jsx';
import CameraPreview from '../components/CameraPreview.jsx';
import DanceAvatar from '../components/DanceAvatar.jsx';
import { formatTime } from '../services/formatTime.js';
import { createContentClock } from '../services/contentClock.js';
import usePracticeStart from '../hooks/usePracticeStart.js';
import useReferenceFrame from '../hooks/useReferenceFrame.js';
import './Practice.css';

export default function Practice({ song, speed, mode, refChar = '1', userChar = '1', guestId, nickname, camera, navigate, onEnded, calibrationFrame }) {
  const clock = useMemo(() => createContentClock(speed, song.duration_sec), [speed, song.duration_sec]);
  const [referenceReady, setReferenceReady] = useState(false);
  const [userReady, setUserReady] = useState(false);
  const onReferenceReady = useCallback(value => setReferenceReady(value), []);
  const onUserReady = useCallback(value => setUserReady(value), []);
  const start = usePracticeStart({
    song,
    speed,
    guestId,
    nickname,
    camera,
    clock,
    onEnded,
    enabled: mode !== 'avatar' || (referenceReady && userReady)
  });
  const resultRef = useRef(null);
  resultRef.current = start.result;

  // 1. 오디오 객체 생성 및 초기 세팅
  const audioRef = useRef(null);
  const songId = song.song_id || song.id;
  const audioUrl = song.audio_url || `/audio/${songId}.mp3`;
  const audioStartSec = Number(song.audio_start_sec ?? 0);

  useEffect(() => {
    const audio = new Audio(audioUrl);
    audio.preload = 'auto';
    audio.playbackRate = speed;
    audio.currentTime = audioStartSec;
    audioRef.current = audio;

    return () => {
      audio.pause();
      audio.src = '';
      audioRef.current = null;
    };
  }, [audioUrl, audioStartSec, speed]);

  // 2. 연습 상태(start.phase)에 따른 오디오 재생 / 일시정지 / 초기화 제어
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    if (start.phase === 'running') {
      const expectedTime = audioStartSec + clock.read();
      if (Math.abs(audio.currentTime - expectedTime) > 0.15) {
        audio.currentTime = expectedTime;
      }
      audio.playbackRate = speed;
      audio.play().catch(err => console.warn('오디오 재생 대기/차단:', err));
    } else if (start.phase === 'first') {
      audio.pause();
      audio.currentTime = audioStartSec;
    } else {
      audio.pause();
    }
  }, [start.phase, audioStartSec, clock, speed]);

  useEffect(() => {
    camera.overlayRef.current = (canvas, frame) => drawEvaluationPose(canvas, frame, resultRef.current);
    return () => { camera.overlayRef.current = null; };
  }, [camera.overlayRef]);

  const duration = start.session?.duration_sec || song.duration_sec;
  const timeRef = useRef(0);
  const [time, setTime] = useState(0);

  // 아바타 모드일 때는 불필요한 백엔드 이미지/포즈 요청을 차단해 웹캠 리타겟팅 지연 방지
  const noopReader = useMemo(() => () => 0, []);
  const isVideoMode = mode === 'video';
  const reference = useReferenceFrame(
    isVideoMode ? song.song_id : null,
    isVideoMode ? clock.read : noopReader
  );

  useEffect(() => {
    camera.contentTimeReaderRef.current = clock.read;
    return () => { camera.contentTimeReaderRef.current = null; };
  }, [camera.contentTimeReaderRef, clock]);

  // 3. 진행 시간 업데이트 및 오디오 싱크 드리프트 보정
  useEffect(() => {
    const timer = setInterval(() => {
      const current = clock.read();
      timeRef.current = current;
      setTime(current);

      const audio = audioRef.current;
      if (audio && !audio.paused && start.phase === 'running') {
        const expectedAudioTime = audioStartSec + current;
        if (Math.abs(audio.currentTime - expectedAudioTime) > 0.2) {
          audio.currentTime = expectedAudioTime;
        }
      }
    }, 50);
    return () => clearInterval(timer);
  }, [clock, audioStartSec, start.phase]);

  const labels = {
    preparingAvatar: '3D 캐릭터와 기준 안무 준비 중…',
    creating: '연습 준비 중…',
    connecting: '연습 준비 중…',
    first: '동작을 기다리고 있습니다…',
    running: '연습 중',
    paused: '일시 정지',
    uncertain: '세션 상태 확인 필요',
    ended: '기록 저장 완료',
    completed: '연습 종료 처리 중',
    error: '평가 연결 중단'
  };

  return (
    <section className="practice-page">
      <div className="practice-heading">
        <h1>{song.artist} · {song.title}</h1>
        <p className="practice-status" role="status">
          {start.busy
            ? `${{pause:'일시 정지',resume:'이어하기',reset:'초기화',end:'종료 및 기록 저장'}[start.busy]} 처리 중…`
            : labels[start.phase]} · {speed}x
        </p>
        <div className="progress-wrap">
          <span>{formatTime(time)} / {formatTime(duration)}</span>
          <progress aria-label="기준 안무 진행률" max={duration} value={time} />
        </div>
      </div>

      <div className="practice-stage">
        <div className="practice-grid">
          {/* 1. 왼쪽 패널: 기준 안무 3D 캐릭터 또는 비디오 */}
          <div className="reference-panel">
            {mode === 'avatar' ? (
              <>
                <span className="media-label" style={{ zIndex: 10 }}>기준 안무 · 3D 캐릭터</span>
                <DanceAvatar
                  songId={song.song_id}
                  readTime={clock.read}
                  onReady={onReferenceReady}
                  modelUrl={`/models/${refChar}.glb`}
                  isReference={true}
                />
              </>
            ) : (
              <>
                <span className="media-label">기준 안무 · 거울모드</span>
                {reference.src ? <img src={reference.src} alt={`${song.title} 기준 안무`} /> : <p>기준 안무를 불러오는 중…</p>}
              </>
            )}
          </div>

          {/* 2. 오른쪽 패널: 사용자 웹캠 리타겟팅 3D 캐릭터 또는 웹캠 원본 */}
          {mode === 'avatar' ? (
            <div className="reference-panel" style={{ position: 'relative' }}>
              <span className="media-label" style={{ zIndex: 10 }}>사용자 웹캠 리타겟팅</span>

              <DanceAvatar
                poseFrameRef={camera.latestFrameRef}
                calibrationFrame={calibrationFrame}
                onReady={onUserReady}
                evaluationRef={resultRef}
                modelUrl={`/models/${userChar}.glb`}
              />

              <div style={{ position: 'absolute', opacity: 0, pointerEvents: 'none', width: '1px', height: '1px', overflow: 'hidden' }}>
                <CameraPreview camera={camera} />
              </div>
            </div>
          ) : (
            <CameraPreview camera={camera} />
          )}
        </div>

        <div className="practice-notices">
          {mode === 'video' && reference.error && (
            <div className="camera-error" role="alert">
              {reference.error}
              <Button secondary onClick={reference.retry}>기준 안무 다시 시도</Button>
            </div>
          )}

          {start.error && (
            <div className="camera-error" role="alert">
              {start.error}
              {start.canRetry && <Button secondary onClick={start.retry}>연결 다시 시도</Button>}
            </div>
          )}
        </div>
      </div>

      <div className="practice-feedback"><LiveEvaluation result={start.result} collapseDetails /></div>

      <div className="practice-actions">
        {mode === 'avatar' && !start.controlReady && <Button secondary onClick={() => navigate('ready')}>준비 화면으로</Button>}
        <Button
          secondary
          disabled={!start.controlReady || !!start.busy || !['running', 'paused'].includes(start.phase) || !!start.failedAction}
          onClick={() => start.control(start.phase === 'paused' ? 'resume' : 'pause')}
        >
          {start.phase === 'paused' ? '이어하기' : '일시 정지'}
        </Button>

        <Button
          secondary
          disabled={!start.controlReady || !!start.busy || !!start.failedAction}
          onClick={() => start.control('reset')}
        >
          처음부터 다시
        </Button>

        {start.failedAction && (
          <Button
            secondary
            disabled={!!start.busy}
            onClick={() => start.control(start.failedAction)}
          >
            같은 요청 다시 시도
          </Button>
        )}

        <Button
          disabled={!start.controlReady || !!start.busy}
          onClick={() => start.control('end')}
        >
          연습 종료 · 기록 저장
        </Button>
      </div>
    </section>
  );
}