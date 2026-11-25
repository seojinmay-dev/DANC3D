import {useEffect, useMemo, useSyncExternalStore} from 'react';
import {getIdentity, getIdentitySnapshot, subscribeIdentity, clearIdentity} from '../services/identity.js';

export default function useIdentity() {
  const snapshot = useSyncExternalStore(subscribeIdentity, getIdentitySnapshot, () => 'null');
  const identity = useMemo(() => JSON.parse(snapshot), [snapshot]);
  useEffect(() => {
    if (!identity) return;
    const timer = setTimeout(() => {
      if (getIdentity()?.token === identity.token || !getIdentity()) clearIdentity();
    }, Math.max(0, Math.min(identity.expiresAt - Date.now(), 2147483647)));
    return () => clearTimeout(timer);
  }, [identity]);
  return identity;
}
