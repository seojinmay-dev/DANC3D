import test from 'node:test';
import assert from 'node:assert/strict';
import {createContentClock} from '../src/services/contentClock.js';
test('content starts at zero, speed applied once, pauses and clamps at duration',()=>{
  let now=100;const clock=createContentClock(.75,3,()=>now);
  now=5100;assert.equal(clock.read(),0);clock.start();now=6100;assert.equal(clock.read(),.75);
  clock.pause();now=9100;assert.equal(clock.read(),.75);clock.start();now=12100;assert.equal(clock.read(),3);
});
test('guest ID survives module reload and blocked storage remains usable',async()=>{
  const values=new Map();globalThis.localStorage={getItem:k=>values.get(k),setItem:(k,v)=>values.set(k,v)};
  const a=await import('../src/services/guest.js?test=a');const id=a.getGuestId();assert.equal(a.getGuestId(),id);
  const b=await import('../src/services/guest.js?test=b');assert.equal(b.getGuestId(),id);
  globalThis.localStorage={getItem(){throw Error('denied');},setItem(){throw Error('denied');}};
  const c=await import('../src/services/guest.js?test=c');assert.match(c.getGuestId(),/^[0-9a-f-]{36}$/);assert.equal(c.getGuestId(),c.getGuestId());delete globalThis.localStorage;
});
