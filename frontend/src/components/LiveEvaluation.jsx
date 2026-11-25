import {memo, useEffect, useRef, useState} from 'react';
import {createScoreDisplay} from '../services/scoreDisplay.js';
const parts={left_arm:'왼팔',right_arm:'오른팔',left_leg:'왼다리',right_leg:'오른다리'};
const bones={left_upper_arm:'왼쪽 상완',left_forearm:'왼쪽 아래팔',right_upper_arm:'오른쪽 상완',right_forearm:'오른쪽 아래팔',left_thigh:'왼쪽 허벅지',left_shin:'왼쪽 종아리',right_thigh:'오른쪽 허벅지',right_shin:'오른쪽 종아리'};
const GOOD_SCORE=80;
const statuses={OK:'정상',ERROR:'교정 필요',INVALID:'평가 불가'};
const similarity=value=>Number.isFinite(value)?`${(value*100).toFixed(1)}%`:'—';
// Time-based smoothing + 0.5s hold for the shown score; a null result (new/reset practice) clears it.
function useDisplayedScore(result) {
  const display=useRef(null),[score,setScore]=useState(null);
  if(!display.current)display.current=createScoreDisplay();
  useEffect(()=>{
    if(!result){display.current.reset();setScore(null);return;}
    if(result.evaluation_performed)setScore(display.current.update(result.raw_final_score,performance.now()/1000));
  },[result]);
  return score;
}
export default memo(function LiveEvaluation({result,collapseDetails=false}) {
  const score=useDisplayedScore(result);
  // NONE only means no correction is confirmed yet, so "Good" also needs a high shown score.
  const good=result?.evaluation_performed&&result.displayed_feedback==='NONE'&&Number.isFinite(score)&&score>=GOOD_SCORE;
  const feedback=!result?.evaluation_performed?'평가를 기다리고 있습니다.':good?'Good':result.displayed_feedback==='NONE'?'—':result.displayed_feedback;
  const similarities=<div className="similarity-panel"><span>전체 동작 유사도 <strong>{similarity(result?.overall_similarity)}</strong></span>{Object.entries(parts).map(([key,label])=><span key={key}>{label} <strong>{similarity(result?.part_similarities?.[key])}</strong></span>)}</div>;
  return <>
    <div className="feedback-panel">
      <div><span className="muted">실시간 피드백{result&&!result.sample_recorded?' · 이전 표시값 유지':''}</span>
        <strong className={good?'feedback-good':undefined}>{feedback}</strong>
        <span className="muted">{result?.pose_detected?'동작 인식됨':result?'전신이 보이도록 서주세요.':'동작 인식 대기'} · {result?.evaluation_performed?'평가 수행':'평가 대기'}</span>
      </div>
      <div className="score"><span>Score</span><div><strong>{Number.isFinite(score)?score:'—'}</strong><span> / 100</span></div></div>
    </div>
    {!collapseDetails&&similarities}
    <details className="evaluation-details">
      <summary>{collapseDetails?'부위별 점수 · 상세 평가':'동작별 평가'}</summary>
      <div className="evaluation-detail-content">
        {collapseDetails&&similarities}
        <p>타이밍: {result?.timing_feedback==='LATE'?'동작 지연':result?.timing_feedback==='NONE'?'지연 피드백 없음':'대기'}</p>
        <div>{Object.entries(bones).map(([key,label])=><span key={key}>{label}: {statuses[result?.bone_status?.[key]]||result?.bone_status?.[key]||'대기'}</span>)}</div>
      </div>
    </details>
  </>;
});
