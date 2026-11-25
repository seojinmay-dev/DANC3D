import PageLayout from '../components/PageLayout.jsx';
import {Button} from '../components/UI.jsx';
import useReadiness from '../hooks/useReadiness.js';
import CameraPreview from '../components/CameraPreview.jsx';
import './Ready.css';

export default function Ready({navigate, camera, mode, onCalibration}) {
  const readiness = useReadiness(camera.latestFrameRef, camera.status, navigate, {
    requireCalibration: mode === 'avatar', onCalibration
  });
  const {phase, count, result, duration} = readiness;
  const steps = Math.max(1, Math.ceil(duration));
  const progress = phase === 'countdown' ? Math.min(1, Math.max(0, (steps - count + 1) / steps)) : 0;
  const checks = [
    ['Pose 검출', result?.pose_detected], ['전신 감지', result?.full_body_visible],
    ['팔 수평', result?.arms_horizontal], ['팔 펴기', result?.arms_extended],
    ['좌우 팔 펼침', result?.arms_spread]
  ];
  const message = phase === 'idle' ? 'SPACE를 눌러 시작하세요.'
    : phase === 'waiting' ? (mode === 'avatar' ? 'T자 자세를 잠시 유지해주세요.' : 'T자 자세를 취해주세요.')
    : phase === 'error' ? '준비 판정을 다시 시도해주세요.' : '잠시 후 연습이 시작됩니다.';
  const actions=<>
    {readiness.error && <p className="camera-error" role="alert">{readiness.error}</p>}
    {phase === 'idle' ? <Button onClick={readiness.begin}>준비 시작 <span>→</span></Button>
      : phase === 'error' ? <Button onClick={readiness.begin}>준비 판정 다시 시도</Button>
      : <Button secondary onClick={readiness.reset}>준비 다시 시작</Button>}
  </>;
  return <PageLayout className="ready-page" title="연습 준비" onBack={() => navigate(mode === 'avatar' ? 'characters' : 'settings')}
    scrollLabel="웹캠 및 준비 판정" actions={actions}>
    <div className="ready-grid">
      <CameraPreview camera={camera}/>
      <div className="ready-info">
        <h2>연습을 시작할 준비가 되었나요?</h2>
        <p>{message}</p>
        <div className={`countdown ${phase === 'countdown' ? 'counting' : ''}`} aria-live="polite">
          <svg className="countdown-ring" viewBox="0 0 115 115" aria-hidden="true" focusable="false">
            <circle className="countdown-track" cx="57.5" cy="57.5" r="54.5"/>
            <circle className="countdown-fill" cx="57.5" cy="57.5" r="54.5"
              pathLength="100" strokeDasharray="100" strokeDashoffset={100 * (1 - progress)}
              transform="rotate(-90 57.5 57.5)"/>
          </svg>
          <span>{phase === 'countdown' ? count : phase === 'waiting' ? 'T' : '5'}</span>
        </div>
        <ul className="readiness-list" aria-label="준비 상태">{checks.map(([text, passed]) => <li key={text}>
          <span className={`readiness-indicator ${passed ? 'is-passed' : ''}`} aria-hidden="true">{passed ? '✓' : ''}</span>
          <span className="readiness-name">{text}</span>
          <small className={passed ? 'is-passed' : ''}>{passed === undefined ? '판정 대기' : passed ? '확인됨' : '미완료'}</small>
        </li>)}</ul>
      </div>
    </div>
  </PageLayout>;
}
