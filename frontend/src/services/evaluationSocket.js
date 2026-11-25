export function evaluationSocketUrl(sessionId) {
  const base=import.meta.env.VITE_WS_BASE_URL||`${location.protocol==='https:'?'wss:':'ws:'}//${location.host}/ws`;
  return `${base.replace(/\/$/,'')}/sessions/${encodeURIComponent(sessionId)}`;
}
export async function connectEvaluation(sessionId,{signal}) {
  const socket=new WebSocket(evaluationSocketUrl(sessionId));
  return new Promise((resolve,reject)=>{
    const cleanup=()=>{clearTimeout(timer);socket.removeEventListener('open',open);socket.removeEventListener('error',fail);socket.removeEventListener('close',fail);signal.removeEventListener('abort',abort);};
    const fail=()=>{cleanup();socket.close();reject(Error('WebSocket 연결 실패. 서버 실행과 Origin 설정을 확인해주세요.'));};
    const open=()=>{cleanup();resolve(socket);};const abort=()=>{cleanup();socket.close();reject(new DOMException('Cancelled','AbortError'));};
    const timer=setTimeout(fail,5000);
    socket.addEventListener('open',open);socket.addEventListener('error',fail);socket.addEventListener('close',fail);signal.addEventListener('abort',abort,{once:true});if(signal.aborted)abort();
  });
}
