import test from 'node:test';
import assert from 'node:assert/strict';
import {getPracticeNickname,getIdentity,parseIdentity,saveIdentity,clearIdentity,subscribeIdentity,getIdentitySnapshot,getRequestIdentity} from '../src/services/identity.js';
import {fetchRecords,fetchSongRecords} from '../src/services/api.js';

const token=(id=7,exp=Math.floor(Date.now()/1000)+3600)=>`${btoa('{}')}.${btoa(JSON.stringify({sub:String(id),exp}))}.signature`;
function environment() {
  const descriptors=new Map(['localStorage','window','fetch'].map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
  const data=new Map();
  Object.defineProperty(globalThis,'localStorage',{configurable:true,value:{getItem:key=>data.get(key)??null,setItem:(key,value)=>data.set(key,value),removeItem:key=>data.delete(key)}});
  Object.defineProperty(globalThis,'window',{configurable:true,value:new EventTarget()});
  return {data,restore(){for(const [key,descriptor]of descriptors){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}};
}

test('reload restores valid account and name; malformed/expired tokens never count as logged in',()=>{
  const env=environment();
  try {
    saveIdentity({token:token(),username:'seojin'});
    assert.equal(getPracticeNickname(),'seojin');assert.equal(getIdentity().id,7);
    assert.equal(parseIdentity(token(7,1),'seojin'),null);
    assert.equal(parseIdentity(token('invalid'),'seojin'),null);
    assert.equal(parseIdentity('broken','seojin'),null);
    env.data.set('token','expired');assert.equal(getIdentity(),null);
    assert.throws(()=>getRequestIdentity(),/로그인/);assert.equal(env.data.has('userId'),false);
    assert.equal(getPracticeNickname(),'비회원');
    Object.defineProperty(globalThis,'localStorage',{configurable:true,get(){throw Error('storage denied');}});
    assert.equal(getPracticeNickname(),'비회원');
  } finally {env.restore();}
});

test('login/logout notify shared subscribers with matching display and request identity',()=>{
  const env=environment();const snapshots=[];const unsubscribe=subscribeIdentity(()=>snapshots.push(getIdentitySnapshot()));
  try {
    saveIdentity({token:token(),username:'seojin'});
    assert.equal(getRequestIdentity().id,7);assert.equal(JSON.parse(snapshots[0]).username,'seojin');
    clearIdentity();assert.equal(getIdentity(),null);assert.equal(snapshots[1],'null');assert.equal(env.data.size,0);
  } finally {unsubscribe();env.restore();}
});

test('record requests send the saved token and validate account ownership across browsers; 401 clears login',async()=>{
  const env=environment();
  try {
    const saved=token();saveIdentity({token:saved,username:'seojin'});
    const row={session_id:'s',guest_id:'old-browser',user_id:7,song_id:'song',title:'title',artist:'artist',
      playback_speed:1,mode:'video',mean_raw_final_score:.8,score_100:80,valid_sample_count:1,recovery_trigger_count:0,ended_at:'2026-11-17T00:00:00Z'};
    globalThis.fetch=async(url,options)=>{
      assert.equal(options.headers.Authorization,`Bearer ${saved}`);
      return {ok:true,status:200,json:async()=>url.includes('/songs/')?{song_id:'song',title:'title',records:[row],best_score_100:80}:[row]};
    };
    assert.equal((await fetchRecords('new-browser'))[0].user_id,7);
    assert.equal((await fetchSongRecords('new-browser','song')).records[0].guest_id,'old-browser');
    globalThis.fetch=async()=>({ok:false,status:401});
    await assert.rejects(fetchRecords('new-browser'),/다시 로그인/);
    assert.equal(getIdentity(),null);assert.equal(env.data.size,0);
  } finally {env.restore();}
});
