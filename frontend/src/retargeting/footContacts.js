import {MathUtils} from 'three';

const distance = (a,b) => Math.hypot(...a.map((v,i)=>v-b[i]));
const percentile = (values,fraction) => [...values].sort((a,b)=>a-b)[Math.floor((values.length-1)*fraction)];

// Offline contact analysis uses source ankle height and world-space speed, not the retargeted feet.
export function analyzeFootContacts(data) {
  const times=data.frames.map((frame,i)=>frame.timestamp??i/data.fps);
  if(!times.length||!times.every((time,i)=>Number.isFinite(time)&&(!i||time>times[i-1]))) {
    throw Error('발 접지 분석에 필요한 시간 정보가 올바르지 않습니다.');
  }
  const ankles={left:[],right:[]},lengths=[];
  for(const frame of data.frames) {
    for(const side of ['left','right']) {
      const hip=frame.joints[side+'Hip'],knee=frame.joints[side+'Knee'],ankle=frame.joints[side+'Ankle'];
      if(![hip,knee,ankle].every(p=>p?.length===3&&p.every(Number.isFinite)))throw Error('발 접지 분석에 필요한 관절 좌표가 없습니다.');
      ankles[side].push(ankle);
      lengths.push(distance(hip,knee)+distance(knee,ankle));
    }
  }
  const legLength=percentile(lengths,.5);
  if(legLength<1e-4)throw Error('기준 안무의 다리 길이를 확인할 수 없습니다.');
  // A shared low percentile prevents treating an always-raised foot as a separate floor.
  const floor=percentile([...ankles.left,...ankles.right].map(p=>p[1]),.05);
  const feet={};
  for(const side of ['left','right']) {
    const points=ankles[side],segments=[];
    let start=-1;
    function finish(end) {
      if(start<0)return;
      if(times[end]-times[start]>=.10) {
        const stable=points.slice(start,Math.min(end+1,start+3));
        const anchor=[0,1,2].map(axis=>stable.reduce((sum,p)=>sum+p[axis],0)/stable.length);
        segments.push({start:times[start],end:times[end],anchor,
          first:start===0,last:end===times.length-1});
      }
      start=-1;
    }
    for(let i=0;i<points.length;i++) {
      const before=Math.max(0,i-1),after=Math.min(points.length-1,i+1);
      const speed=after===before?0:distance(points[after],points[before])/(times[after]-times[before]);
      const height=points[i][1]-floor;
      if(start<0) {
        if(height<=legLength*.055&&speed<=legLength*.18)start=i;
      } else {
        const drift=Math.hypot(points[i][0]-points[start][0],points[i][2]-points[start][2]);
        if(height>legLength*.10||speed>legLength*.45||drift>legLength*.10)finish(i-1);
      }
    }
    finish(points.length-1);
    feet[side]={segments,firstAnkle:[...points[0]]};
  }
  return {times,legLength,floor,feet};
}

export function sampleFootContact(plan,side,time) {
  const segments=plan.feet[side].segments;
  let lo=0,hi=segments.length;
  while(lo<hi) {
    const mid=(lo+hi)>>1;
    if(segments[mid].end<time)lo=mid+1;else hi=mid;
  }
  const segment=segments[lo];
  if(!segment||time<segment.start||time>segment.end)return null;
  const fade=Math.min(.08,(segment.end-segment.start)/3);
  const enter=segment.first?1:MathUtils.smoothstep(time,segment.start,segment.start+fade);
  const leave=segment.last?1:1-MathUtils.smoothstep(time,segment.end-fade,segment.end);
  return {anchor:segment.anchor,weight:Math.min(enter,leave)};
}
