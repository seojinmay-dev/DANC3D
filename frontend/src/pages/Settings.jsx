import { Button } from '../components/UI.jsx';
import PageLayout from '../components/PageLayout.jsx';

export default function Settings({ song, speed, setSpeed, mode, setMode, navigate }) {
  return (
    <PageLayout title="플레이 설정" onBack={() => navigate('songs')} scrollLabel="연습 설정"
      actions={<Button onClick={() => navigate(mode === 'avatar' ? 'characters' : 'ready')}>다음 <span>→</span></Button>}>

      <div className="song-summary">
        <div className="thumbnail">
          {song && (
            <img 
              src={`/images/${song.title.replace(/\s/g, '')}.jpg`} 
              alt={song.title} 
              className="cover-img" 
            />
          )}
        </div>
        <div>
          <h2>{song?.artist} · {song?.title}</h2>
          <p>{song?.artist} | 1분 58초</p> 
        </div>
      </div>

      <h2 className="section-title">연습 모드</h2>
      <div className="modes">
        {/* 1. 기존 기준 안무 영상 + 웹캠 모드 */}
        <button 
          className={`mode ${mode === 'video' ? 'selected' : ''}`} 
          aria-pressed={mode === 'video'}
          onClick={() => setMode('video')}
        >
          <span className="mode-icon">▣</span>
          <div>
            <strong>기준 안무 영상 + 웹캠</strong>
            <p>기준 영상과 나의 동작을<br />비교하며 연습합니다.</p>
          </div>
          {mode === 'video' && <span className="check">✓</span>}
        </button>

        {/* 2. 3D 캐릭터 모드 (활성화됨) */}
        <button 
          className={`mode ${mode === 'avatar' ? 'selected' : ''}`} 
          aria-pressed={mode === 'avatar'}
          onClick={() => setMode('avatar')}
        >
          <span className="mode-icon">♙</span>
          <div>
            <strong>캐릭터 모드</strong>
            <p>3D 캐릭터와 함께<br />연습하는 모드입니다.</p>
          </div>
          {mode === 'avatar' && <span className="check">✓</span>}
        </button>
      </div>

      <div className="speed-row">
        <h2 className="section-title">재생 속도</h2>
        <div className="speed-options" role="group" aria-label="재생 속도">
          {[.5, .75, 1, 1.25, 1.5].map(n => (
            <button 
              key={n} 
              className={speed === n ? 'selected' : ''} 
              aria-pressed={speed === n} 
              onClick={() => setSpeed(n)}
            >
              {n === 1 ? '1.0' : n}x
            </button>
          ))}
        </div>
      </div>

    </PageLayout>
  );
}
