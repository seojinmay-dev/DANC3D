const KEY='danc3d.guest_id';
let memoryGuest;
export function getGuestId() {
  if(memoryGuest)return memoryGuest;
  try{const stored=localStorage.getItem(KEY);if(stored&&/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(stored))return memoryGuest=stored;}catch{}
  memoryGuest=crypto.randomUUID();try{localStorage.setItem(KEY,memoryGuest);}catch{}
  return memoryGuest;
}
