import test from 'node:test';
import assert from 'node:assert/strict';
import {recordScore,formatRecordScore,validateRecords,validateSongRecords,consistentEndRecord} from '../src/services/records.js';
const row=(id='s',extra={})=>({session_id:id,guest_id:'g',song_id:'song',title:'실제 제목',artist:'실제 가수',playback_speed:.75,mode:'video',mean_raw_final_score:.82,score_100:82,valid_sample_count:10,recovery_trigger_count:0,ended_at:'2026-10-05T10:00:00+09:00',...extra});
test('RAW mean score, existing decimal presentation and null do not substitute realtime or zero',()=>{
 assert.equal(recordScore(row()),82);assert.equal(formatRecordScore(82), '82.0');
 const empty=row('empty',{mean_raw_final_score:null,score_100:null,valid_sample_count:0,displayed_score:100});
 assert.equal(recordScore(empty),null);assert.equal(formatRecordScore(null),'평가 표본 부족');assert.equal(validateRecords([empty],'g').length,1);
});
test('Repeated songs stay separate, newest first by actual date including timezone offsets',()=>{
 const a=row('older',{ended_at:'2026-10-05T10:00:00+09:00'}),b=row('newer',{ended_at:'2026-10-05T02:00:00Z'});
 const input=[a,b];assert.deepEqual(validateRecords(input,'g').map(r=>r.session_id),['newer','older']);assert.equal(input[0],a);assert.deepEqual(validateRecords([],'g'),[]);
});
test('Song best excludes null, handles ties and all-null histories',()=>{
 const records=[row('a'),row('b'),row('empty',{mean_raw_final_score:null,score_100:null,valid_sample_count:0})];
 assert.equal(validateSongRecords({song_id:'song',title:'제목',records,best_score_100:82},'g','song').records.length,3);
 assert.equal(validateSongRecords({song_id:'song',title:'제목',records:records.slice(2),best_score_100:null},'g','song').best_score_100,null);
 assert.throws(()=>validateSongRecords({song_id:'song',title:'제목',records,best_score_100:100},'g','song'));
});
test('Invalid, cross-user, duplicate or mock shaped responses rejected',()=>{
 for(const records of [[row('s',{guest_id:'other'})],[row(),row()],[row('s',{score_100:.82})],[{id:'mock',score:78}],[row('s',{ended_at:'invalid'})]])assert.throws(()=>validateRecords(records,'g'));
 assert.throws(()=>validateRecords([row()],'g','missing-song'));
});
test('End record consistency includes raw precision and metadata without mutating stored result',()=>{
 const current=row();assert.equal(consistentEndRecord(current,[row()]),true);assert.equal(consistentEndRecord(current,[]),false);
 assert.equal(consistentEndRecord(current,[row('s',{mean_raw_final_score:.82001,score_100:82.001})]),false);assert.equal(current.mean_raw_final_score,.82);
});
test('Submillisecond dates keep true latest order instead of collapsing into a session-ID tie',()=>{
 const newer=row('a',{ended_at:'2026-10-05T01:00:00.000900Z'}),older=row('z',{ended_at:'2026-10-05T01:00:00.000100Z'});
 assert.deepEqual(validateRecords([older,newer],'g').map(r=>r.session_id),['a','z']);
});
test('member records follow account ID across browsers; other accounts and guest records stay separate',()=>{
 const member=row('member',{guest_id:'old-browser',user_id:7});
 assert.equal(validateRecords([member],'new-browser',undefined,7).length,1);
 assert.throws(()=>validateRecords([member],'new-browser',undefined,8));
 assert.throws(()=>validateRecords([member],'old-browser'));
 assert.throws(()=>validateRecords([row()],'g',undefined,7));
 assert.equal(validateSongRecords({song_id:'song',title:'title',records:[member],best_score_100:82},'new-browser','song',7).records.length,1);
 assert.throws(()=>validateRecords([row('bad',{user_id:7,mean_raw_final_score:2,score_100:200})],'g',undefined,7));
});
