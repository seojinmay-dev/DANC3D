import test from 'node:test';
import assert from 'node:assert/strict';
import {createLivePoseFilter} from '../src/services/livePoseFilter.js';
import {trackingAlpha} from '../src/retargeting/retarget/core/trackingSmoothing.js';

function frame(time, shift=0) {
  const image_landmarks = Array.from({length:33},()=>({x:.5+shift/2,y:.5,z:0,visibility:1,presence:1}));
  const world_landmarks = Array.from({length:33},()=>[shift,0,0]);
  return {timestamp_ms:time,image_landmarks,world_landmarks,world_confidence:Array(33).fill(1)};
}

test('confidence drops hold only affected limb, with hysteresis, without mutating raw input',()=>{
  const filter=createLivePoseFilter(); filter.update(frame(0),0);
  const noisy=frame(33,.1); noisy.image_landmarks[15].visibility=.2;
  const original=structuredClone(noisy), result=filter.update(noisy,33);
  assert.equal(result.parts.leftArm,false); assert.equal(result.parts.rightArm,true);
  assert.deepEqual(result.world_landmarks[15],[0,0,0]); assert.ok(result.world_landmarks[16][0]>0);
  assert.deepEqual(noisy,original);
  const hovering=frame(66,.1); hovering.image_landmarks[15].visibility=.55;
  assert.equal(filter.update(hovering,66).parts.leftArm,false);
  assert.equal(filter.update(frame(99,.1),99).parts.leftArm,true);
  const retained=frame(132,.1); retained.image_landmarks[15].visibility=.5;
  assert.equal(filter.update(retained,132).parts.leftArm,true);
});

test('world confidence, nonfinite coordinates, body loss and stale samples cannot drive the avatar',()=>{
  const filter=createLivePoseFilter(); filter.update(frame(0),0);
  const bad=frame(33); bad.world_confidence[27]=.1;
  assert.equal(filter.update(bad,33).parts.leftLeg,false);
  const nan=frame(66); nan.world_landmarks[15][0]=NaN;
  assert.equal(filter.update(nan,66).parts.leftArm,false);
  const lost=frame(99); lost.image_landmarks[23].visibility=.1;
  assert.equal(filter.update(lost,99),null); assert.equal(filter.canAnimate(100),false);
  filter.update(frame(132),132); assert.equal(filter.canAnimate(132),true);
  assert.equal(filter.update(frame(132),500),null); assert.equal(filter.canAnimate(500),false);
  assert.equal(filter.update(null,500),null);
});

test('only new sensor timestamps enter the filter; recovery does not replay a backlog',()=>{
  const filter=createLivePoseFilter(); filter.update(frame(0),0);
  assert.equal(filter.update(frame(0,99),10),null);
  assert.equal(filter.update(frame(-1,99),10),null);
  assert.equal(filter.update(frame(10),500),null);
  const recovered=filter.update(frame(600,.3),600);
  assert.ok(recovered.world_landmarks[15].every(Number.isFinite));
  assert.equal(recovered.world_landmarks[15][0],.3);
});

test('stationary synthetic jitter falls while a fast motion reaches 90% within 100ms',t=>{
  const filter=createLivePoseFilter(); let rawSquared=0, filteredSquared=0, count=0;
  for(let i=0;i<120;i++) {
    const noise=.004*Math.sin(i*2.3), now=i*1000/30;
    const result=filter.update(frame(now,noise),now);
    if(i>20){rawSquared+=noise**2;filteredSquared+=result.world_landmarks[15][0]**2;count++;}
  }
  const ratio=Math.sqrt(filteredSquared/rawSquared);
  assert.ok(ratio<.5,`jitter ratio ${ratio}`);
  let reached;
  for(let i=0;i<20;i++) {
    const now=(120+i)*1000/30, sample=filter.update(frame(now,.4),now);
    if(sample.world_landmarks[15][0]>=.36 && reached===undefined)reached=(i+1)*1000/30;
  }
  assert.ok(reached<=100,`step response ${reached}ms`);
  t.diagnostic(`Synthetic input: jitter RMS ratio ${ratio.toFixed(3)}, 90% step response within ${reached}ms (30Hz)`);
});

test('combined input filter and render interpolation reach fast movement within 120ms',t=>{
  const measurements=[];
  for(const renderHz of [30,60,120]) {
    const filter=createLivePoseFilter(); let target=0, actual=0, nextSensor=0, reached;
    for(let tick=0;tick<renderHz*2;tick++) {
      const now=tick*1000/renderHz;
      if(now+1e-6>=nextSensor) {
        const sample=filter.update(frame(nextSensor,nextSensor>=1000-1e-6?.4:0),now);
        target=sample.world_landmarks[15][0];nextSensor+=1000/30;
      }
      actual+=(target-actual)*trackingAlpha(1/renderHz);
      if(now>=1000 && actual>=.36 && reached===undefined)reached=now-1000;
    }
    assert.ok(reached<=120,`${renderHz}Hz combined lag: ${reached}ms`);
    measurements.push(`${renderHz}Hz: ${reached.toFixed(1)}ms`);
  }
  t.diagnostic(`Synthetic 30Hz sensor → filter → interpolation: ${measurements.join(', ')}`);
});

test('bone/root response follows elapsed seconds at 30,60,120Hz and handles tab suspension',()=>{
  const results=[];
  for(const fps of [30,60,120]) {
    let position=0;
    for(let i=0;i<fps/10;i++)position+=(1-position)*trackingAlpha(1/fps);
    results.push(position);
  }
  assert.ok(Math.max(...results)-Math.min(...results)<1e-12);
  assert.equal(trackingAlpha(0),0);assert.equal(trackingAlpha(NaN),0);
  assert.equal(trackingAlpha(1/60,false),1);assert.equal(trackingAlpha(20),trackingAlpha(.1));
});
