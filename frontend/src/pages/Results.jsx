import PageLayout from '../components/PageLayout.jsx';
import {Button} from '../components/UI.jsx';
import RecordsTable from '../components/RecordsTable.jsx';
import useRecords from '../hooks/useRecords.js';
import {consistentEndRecord,formatRecordScore,recordScore} from '../services/records.js';
import './Results.css';

export default function Results({current,song,guestId,navigate,identity}) {
  const history=useRecords(guestId,current?.song_id,identity?.token);
  if(!current)return <PageLayout className="results-page" title="연습 결과"
    actions={<Button onClick={()=>navigate('history')}>연습 히스토리 보기</Button>}>
    <p className="empty">현재 연습 결과가 없습니다. 저장된 기록은 히스토리에서 확인해주세요.</p>
  </PageLayout>;
  const score=recordScore(current),matchingSong=song?.song_id===current.song_id?song:null;
  const title=history.data?.title||matchingSong?.title||current.song_id;
  const best=history.data?.best_score_100;
  const actions=<>
    <Button secondary onClick={()=>navigate(matchingSong?'settings':'songs')}>다시 연습</Button>
    <Button onClick={()=>navigate('home')}>홈으로</Button>
  </>;
  return <PageLayout className="results-page" title="연습 결과" scrollLabel="연습 결과 및 곡별 기록" actions={actions}>
      <h2 className="results-song">{matchingSong?.artist?`${matchingSong.artist} · `:''}{title}</h2>
      <div className="result-summary">
        <article className="result-stat result-stat-current">
          <div className="result-stat-heading"><span>이번 연습 · 평균 점수</span><span className="result-saved">✓ 저장 완료</span></div>
          <div className="result-score"><strong>{score===null?'—':formatRecordScore(score)}</strong><span> / 100</span></div>
          <div className="result-stat-meta">
            <small>{new Date(current.ended_at).toLocaleString('ko-KR')} · {current.playback_speed}x</small>
            <small>유효 평가 표본 {current.valid_sample_count}개</small>
          </div>
        </article>
        <article className="result-stat">
          <div className="result-stat-heading"><span>해당 곡 · 최고 기록</span><span className="result-best-label">BEST</span></div>
          <div className="best-score"><strong>{history.data?best===null?'—':formatRecordScore(best):'—'}</strong><span> / 100</span></div>
          <div className="result-stat-meta"><small>{history.loading?'최고 기록 조회 중...':history.error?'최고 기록 조회 실패':best===null?'점수가 있는 기록이 없습니다.':'저장된 곡별 기록 기준'}</small></div>
        </article>
      </div>
      {score===null&&<p>유효 평가 표본이 부족하여 최종 점수가 없습니다. 기록은 저장되었습니다.</p>}
      {history.error&&<div className="camera-error" role="alert">{history.error} 이번 기록의 저장 완료 상태는 유지됩니다.<Button secondary onClick={history.retry}>기록 다시 시도</Button></div>}
      {history.data&&!consistentEndRecord(current,history.data.records)&&<div role="alert" className="camera-error">저장된 기록을 확인할 수 없습니다. 이번 연습 기록은 유지됩니다.<Button secondary onClick={history.retry}>기록 다시 시도</Button></div>}
      <h2 className="section-title">곡별 연습 히스토리</h2>
      {history.loading&&<p role="status" className="empty">곡별 기록을 불러오는 중...</p>}
      {history.data&&<RecordsTable records={history.data.records} best={best}/>}
  </PageLayout>;
}
