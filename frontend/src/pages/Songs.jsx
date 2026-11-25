import PageLayout from '../components/PageLayout.jsx';
import {useEffect,useState} from 'react';
import {Button} from '../components/UI.jsx';
import {fetchSongs} from '../services/api.js';
import './Songs.css';

export default function Songs({song,setSong,navigate}) {
  const [query,setQuery]=useState(''),[songs,setSongs]=useState([]),[loading,setLoading]=useState(true),[error,setError]=useState(''),[attempt,setAttempt]=useState(0);
  useEffect(()=>{
    const controller=new AbortController();
    setLoading(true);setError('');setSongs([]);
    fetchSongs({signal:controller.signal}).then(data=>{
      if(!controller.signal.aborted)setSongs(data);
    }).catch(err=>{
      if(!controller.signal.aborted)setError(err instanceof TypeError?'곡 목록을 불러올 수 없습니다. 네트워크 연결을 확인한 뒤 다시 시도해주세요.':err.message);
    }).finally(()=>{if(!controller.signal.aborted)setLoading(false);});
    return()=>controller.abort();
  },[attempt]);
  const selected=songs.find(s=>s.song_id===song?.song_id);
  const items=songs.filter(s=>`${s.title} ${s.artist}`.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()));
  const toolbar=<>
    <label className="search"><span aria-hidden="true">⌕</span><input aria-label="곡 또는 아티스트 검색" placeholder="곡 제목 또는 아티스트를 검색하세요" value={query} onChange={e=>setQuery(e.target.value)}/></label>
    {!loading&&!error&&<div className="catalog-meta"><span>전체 {songs.length}곡 · 검색 결과 {items.length}곡</span><span>목록을 스크롤해 다른 곡도 확인하세요.</span></div>}
  </>;
  const actions=(
    <>
      <div className="songs-selection-copy" aria-live="polite">
        <span>{selected?'선택한 곡':'곡을 선택해주세요.'}</span>
        {selected&&<><strong>{selected.title}</strong><span>{selected.artist}</span></>}
      </div>
      <Button disabled={loading||!!error||!selected} onClick={()=>{setSong(selected);navigate('settings');}}>선택 완료 <span>→</span></Button>
    </>
  );
  return <PageLayout className="songs-page" title="기준 안무 선택" description="연습하고 싶은 곡을 골라주세요."
    toolbar={toolbar} actions={actions} scrollLabel="곡 목록" busy={loading}>
      {loading&&<p className="empty" role="status">곡 목록을 불러오는 중입니다…</p>}
      {error&&<div className="empty" role="alert"><p>{error}</p><Button secondary onClick={()=>setAttempt(n=>n+1)}>다시 시도</Button></div>}
      {!loading&&!error&&<>
        <div className="song-grid">{items.map(s=><button key={s.song_id} className={`song-card ${song?.song_id===s.song_id?'selected':''}`} aria-pressed={song?.song_id===s.song_id} onClick={()=>setSong(s)}>
          <div className="thumbnail"><img src={`/images/${s.title.replace(/\s/g,'')}.jpg`} alt={s.title} className="cover-img"/></div>
          <div className="song-card-copy"><strong>{s.title}</strong><span>{s.artist}</span></div>
          {song?.song_id===s.song_id&&<span className="selection-check" aria-label="선택됨">✓</span>}
        </button>)}</div>
        {!items.length&&<p className="empty">{songs.length?'검색 결과가 없습니다.':'등록된 곡이 없습니다.'}</p>}
      </>}

  </PageLayout>;
}
