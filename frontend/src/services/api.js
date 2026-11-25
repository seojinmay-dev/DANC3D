import {getRequestIdentity,getIdentity,clearIdentity} from './identity.js';
import {validateRecords,validateSongRecords} from './records.js';
export const API_BASE=(import.meta.env?.VITE_API_BASE_URL||'/api').replace(/\/$/,'');

export async function loginUser(username,password,{signal}={}) {
  const body=new URLSearchParams({username,password,grant_type:'password'});
  const response=await fetch(`${API_BASE}/login`,{
    method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded',accept:'*/*'},body,signal
  });
  if(!response.ok)throw Error(`로그인 실패: 아이디 또는 비밀번호가 올바르지 않습니다. (상태코드: ${response.status})`);
  return response.json();
}

export async function registerUser(username,password,{signal}={}) {
  const query=new URLSearchParams({username,password});
  const response=await fetch(`${API_BASE}/register?${query}`,{method:'POST',headers:{accept:'*/*'},signal});
  if(!response.ok) {
    const data=await response.json().catch(()=>null);
    throw Error(`회원가입 실패: ${typeof data?.detail==='string'?data.detail:'서버 연결을 확인해주세요.'}`);
  }
  return response.json();
}

export async function fetchSongs({signal}={}) {
  const response=await fetch(`${API_BASE}/songs`,{signal});
  if(!response.ok)throw new Error(`곡 목록을 불러오지 못했습니다. (HTTP ${response.status})`);
  const songs=await response.json();
  if(!Array.isArray(songs)||songs.some(s=>!s||typeof s.song_id!=='string'||!s.song_id||typeof s.title!=='string'||typeof s.artist!=='string'||!Number.isFinite(s.duration_sec)||s.duration_sec<0||typeof s.reference_ready!=='boolean')||new Set(songs.map(s=>s.song_id)).size!==songs.length)throw new Error('곡 목록 응답 형식이 올바르지 않습니다.');
  return songs;
}

export async function requestReadiness(attemptId,frame,{signal}={}) {
  const image_landmarks=frame.image_landmarks===null?null:frame.image_landmarks.map(({x,y,visibility})=>{
    if(![x,y,visibility].every(Number.isFinite))throw new Error('Image Landmark 좌표 또는 visibility가 유효하지 않습니다.');
    return {x,y,visibility};
  });
  const response=await fetch(`${API_BASE}/readiness`,{method:'POST',headers:{'Content-Type':'application/json'},signal,body:JSON.stringify({attempt_id:attemptId,image_width:frame.image_width,image_height:frame.image_height,image_landmarks})});
  if(!response.ok)throw new Error(response.status===422?'준비 판정 입력이 올바르지 않습니다. (HTTP 422)':`준비 판정 서버에 연결하지 못했습니다. (HTTP ${response.status})`);
  const data=await response.json();
  if(typeof data.ready!=='boolean'||data.status!==(data.ready?'READY_FOR_COUNTDOWN':'TPOSE_WAITING')||typeof data.attempt_id!=='string'||!Number.isFinite(data.countdown_duration_sec)||data.countdown_duration_sec<=0||['pose_detected','full_body_visible','arms_horizontal','arms_extended','arms_spread'].some(k=>typeof data[k]!=='boolean'))throw new Error('준비 판정 응답 형식이 올바르지 않습니다.');
  return data;
}

export function referenceFrameUrl(songId,time) {
  return `${API_BASE}/songs/${encodeURIComponent(songId)}/reference-frame?time_sec=${time.toFixed(3)}`;
}

export async function createSession(body,{signal}={}) {
  const identity=getRequestIdentity();
  const response=await fetch(`${API_BASE}/sessions/start`,{method:'POST',headers:{'Content-Type':'application/json', ...identityHeaders(identity)},body:JSON.stringify(body),signal});
  checkLoginResponse(response,identity);
  if(response.status!==201){const detail=await response.json().catch(()=>null);const error=new Error(`세션 생성 실패 (HTTP ${response.status}). 닉네임과 Reference 준비 상태를 확인해주세요.`);error.safeToRetry=(response.status>=400&&response.status<500)||['session_capacity_reached','reference_data_unavailable'].includes(detail?.detail?.code);throw error;}
  const data=await response.json();
  if(typeof data.session_id!=='string'||data.status!=='running'||!Number.isFinite(data.duration_sec)||data.duration_sec<=0||data.song_id!==body.song_id||data.playback_speed!==body.playback_speed)throw Error('세션 생성 응답을 확인할 수 없습니다. 중복 방지를 위해 자동 재생성을 중단합니다.');
  return data;
}

export async function controlSession(sessionId,action,{signal,timeoutMs=10000}={}) {
  if(!['pause','resume','reset','end'].includes(action))throw Error('지원하지 않는 세션 제어입니다.');
  const identity=getRequestIdentity();
  const response=await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/${action}`,{method:'POST',headers: {...identityHeaders(identity)},signal:signal?AbortSignal.any([signal,AbortSignal.timeout(timeoutMs)]):AbortSignal.timeout(timeoutMs)});
  checkLoginResponse(response,identity);
  if(!response.ok)throw Error(`세션 ${action} 실패 (HTTP ${response.status}). 같은 세션에서 다시 시도해주세요.`);
  const data=await response.json(),expected=action==='pause'?'paused':action==='end'?'ended':'running';
  if(data.session_id!==sessionId||data.status!==expected||(action==='reset'&&data.playback_time_sec!==0))throw Error('세션 제어 응답을 확인할 수 없습니다. 같은 요청을 다시 시도해주세요.');
  if(action==='end'&&(!data.record||data.record.session_id!==sessionId||!Number.isInteger(data.record.valid_sample_count)||!(data.record.mean_raw_final_score===null||Number.isFinite(data.record.mean_raw_final_score))))throw Error('기록 저장 응답을 확인할 수 없습니다. 종료를 다시 시도해주세요.');
  return data;
}

export async function fetchRecords(guestId,{signal}={}) {
  const identity=getRequestIdentity();
  const response=await fetch(`${API_BASE}/records/${encodeURIComponent(guestId)}`,{
    headers: { ...identityHeaders(identity) },
    signal:signal?AbortSignal.any([signal,AbortSignal.timeout(10000)]):AbortSignal.timeout(10000)
  });
  checkLoginResponse(response,identity);
  if(!response.ok)throw Error(`연습 기록 조회 실패 (HTTP ${response.status}).`);
  return validateRecords(await response.json(),guestId,undefined,identity?.id??null);
}

export async function fetchSongRecords(guestId,songId,{signal}={}) {
  const identity=getRequestIdentity();
  const response=await fetch(`${API_BASE}/records/${encodeURIComponent(guestId)}/songs/${encodeURIComponent(songId)}`,{
    headers: { ...identityHeaders(identity) },
    signal:signal?AbortSignal.any([signal,AbortSignal.timeout(10000)]):AbortSignal.timeout(10000)
  });
  checkLoginResponse(response,identity);
  if(!response.ok)throw Error(response.status===404?'곡이 존재하지 않아 곡별 기록을 조회할 수 없습니다. (HTTP 404)':`곡별 기록 조회 실패 (HTTP ${response.status}).`);
  return validateSongRecords(await response.json(),guestId,songId,identity?.id??null);
}

const identityHeaders=identity=>identity?{Authorization:`Bearer ${identity.token}`} : {};
function checkLoginResponse(response,identity) {
  if(response.status!==401)return;
  if(identity&&getIdentity()?.token===identity.token)clearIdentity();
  throw Error('로그인이 만료되었거나 유효하지 않습니다. 홈에서 다시 로그인해주세요.');
}
