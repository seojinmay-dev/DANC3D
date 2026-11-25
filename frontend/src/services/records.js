// Record scores are RAW means (0..1), not realtime displayed_score.
export const recordScore=record=>record.mean_raw_final_score===null?null:record.mean_raw_final_score*100;
// Preserve the existing Results presentation: one decimal; no scoring correction.
export const formatRecordScore=value=>value===null?'평가 표본 부족':value.toFixed(1);
const subMillisecond=row=>Number((row.ended_at.match(/\.(\d+)/)?.[1]?.slice(3)||'').padEnd(6,'0'));
export function validateRecords(rows,guestId,songId,userId=null) {
  if(!Array.isArray(rows))throw Error('기록 응답 형식이 올바르지 않습니다.');
  const ids=new Set();
  for(const row of rows){
    const ownerMatches=userId===null?row?.guest_id===guestId&&row?.user_id==null:Number.isSafeInteger(userId)&&userId>0&&row?.user_id===userId;
    if(!row||typeof row.session_id!=='string'||!row.session_id||ids.has(row.session_id)||!ownerMatches||typeof row.song_id!=='string'||!row.song_id||(songId&&row.song_id!==songId)||typeof row.title!=='string'||typeof row.artist!=='string'||!Number.isFinite(row.playback_speed)||row.mode!=='video'||!Number.isInteger(row.valid_sample_count)||row.valid_sample_count<0||!Number.isInteger(row.recovery_trigger_count)||row.recovery_trigger_count<0||typeof row.ended_at!=='string'||!Number.isFinite(Date.parse(row.ended_at))||!(row.mean_raw_final_score===null||Number.isFinite(row.mean_raw_final_score)&&row.mean_raw_final_score>=0&&row.mean_raw_final_score<=1)||row.score_100!==recordScore(row))throw Error('기록 응답의 사용자·점수·필드를 확인할 수 없습니다.');
    ids.add(row.session_id);
  }
  // ISO dates may use different offsets: compare actual instants.
  return [...rows].sort((a,b)=>Date.parse(b.ended_at)-Date.parse(a.ended_at)||subMillisecond(b)-subMillisecond(a)||b.session_id.localeCompare(a.session_id));
}
export function validateSongRecords(data,guestId,songId,userId=null) {
  if(!data||data.song_id!==songId||typeof data.title!=='string')throw Error('곡별 기록 응답 형식이 올바르지 않습니다.');
  const records=validateRecords(data.records,guestId,songId,userId);
  const scores=records.map(r=>r.score_100).filter(s=>s!==null),best=scores.length?Math.max(...scores):null;
  if(data.best_score_100!==best)throw Error('최고 기록 응답이 실제 기록과 일치하지 않습니다.');
  return {...data,records};
}
export function consistentEndRecord(current,rows) {
  const saved=rows.find(r=>r.session_id===current.session_id);
  return !!saved&&['session_id','guest_id','user_id','song_id','playback_speed','mode','mean_raw_final_score','valid_sample_count','recovery_trigger_count','ended_at'].every(key=>saved[key]===current[key]);
}
