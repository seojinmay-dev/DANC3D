// MediaPipe milliseconds and DTW aligned time never drive this clock.
export function createContentClock(speed,duration,now=()=>performance.now()) {
  let seconds=0,anchor=null;
  const read=(at=now())=>Math.min(duration,seconds+(anchor===null?0:Math.max(0,at-anchor)/1000*speed));
  return {read,reset(){seconds=0;anchor=null;},start(){if(anchor===null)anchor=now();},pause(){seconds=read();anchor=null;}};
}
