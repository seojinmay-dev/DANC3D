const CHANGED = 'danc3d:identity-changed';

// JWT decoding here selects display/validation scope only. The server verifies the signature.
export function parseIdentity(token, username, now = Date.now()) {
  try {
    if (!token || token.split('.').length !== 3 || !username?.trim()) return null;
    const payload = JSON.parse(atob(token.split('.')[1].replace(/-/g,'+').replace(/_/g,'/')));
    const id = Number(payload.sub);
    if (typeof payload.sub !== 'string' || !/^[1-9]\d*$/.test(payload.sub) || !Number.isSafeInteger(id) ||
        !Number.isFinite(payload.exp) || payload.exp * 1000 <= now) return null;
    return {id, username:username.trim(), token, expiresAt:payload.exp * 1000};
  } catch { return null; }
}

export function getIdentity() {
  try { return parseIdentity(localStorage.getItem('token'), localStorage.getItem('userId')); }
  catch { return null; }
}
export const getIdentitySnapshot = () => JSON.stringify(getIdentity());
const notify = () => { if (typeof window !== 'undefined') window.dispatchEvent(new Event(CHANGED)); };
export function subscribeIdentity(listener) {
  window.addEventListener(CHANGED, listener); window.addEventListener('storage', listener);
  return () => { window.removeEventListener(CHANGED, listener); window.removeEventListener('storage', listener); };
}
export function saveIdentity({token, username}) {
  if (!parseIdentity(token, username)) throw Error('로그인 응답을 확인할 수 없습니다. 다시 로그인해주세요.');
  localStorage.setItem('token',token); localStorage.setItem('userId',username.trim()); notify();
}
export function clearIdentity() {
  try { localStorage.removeItem('token'); localStorage.removeItem('userId'); } catch {}
  notify();
}
export function getRequestIdentity() {
  const identity = getIdentity();
  let token;
  try { token = localStorage.getItem('token'); } catch {}
  if (token && !identity) { clearIdentity(); throw Error('로그인이 만료되었거나 유효하지 않습니다. 다시 로그인해주세요.'); }
  return identity;
}
export const getPracticeNickname = () => getIdentity()?.username || '비회원';
