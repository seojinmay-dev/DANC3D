// Format elapsed content time as mm:ss.
export function formatTime(seconds) {
  const n=Math.max(0,Math.floor(seconds));
  return `${Math.floor(n/60).toString().padStart(2,'0')}:${(n%60).toString().padStart(2,'0')}`;
}
