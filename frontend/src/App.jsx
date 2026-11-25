import { useCallback, useEffect, useRef, useState } from 'react';
import Navigation from './components/Navigation.jsx';
import Home from './pages/Home.jsx';
import Songs from './pages/Songs.jsx';
import Settings from './pages/Settings.jsx';
import CharacterSelect from './pages/CharacterSelect.jsx';
import PracticeFlow from './components/PracticeFlow.jsx';
import { getGuestId } from './services/guest.js';
import { saveIdentity, clearIdentity } from './services/identity.js';
import useIdentity from './hooks/useIdentity.js';
import Results from './pages/Results.jsx';
import History from './pages/History.jsx';
import PageLayout from './components/PageLayout.jsx';

export default function App() {
  const [page, setPage] = useState('home');
  const [song, setSong] = useState(null);
  const [speed, setSpeed] = useState(1);
  const [current, setCurrent] = useState(null);
  const identity = useIdentity();
  const nickname = identity?.username || '비회원';
  const ownerRef = useRef(identity?.id ?? null);

  useEffect(() => {
    const owner = identity?.id ?? null;
    if (ownerRef.current !== owner) {
      setCurrent(null);
      setPage(previous => ['ready', 'practice', 'results'].includes(previous) ? 'home' : previous);
    }
    ownerRef.current = owner;
  }, [identity?.id]);

  const [mode, setMode] = useState('video');
  // 3D 캐릭터 선택 상태 (기본값 1번, 동일 캐릭터 체크됨)
  const [refChar, setRefChar] = useState('1');
  const [userChar, setUserChar] = useState('1');
  const [sameChar, setSameChar] = useState(true);

  const navigate = useCallback(next => {
    if (['settings', 'characters', 'ready', 'practice'].includes(next) && !song) next = 'songs';
    if (next === 'results') next = 'history';
    if (['songs', 'ready'].includes(next)) setCurrent(null);
    setPage(next);
    window.scrollTo(0, 0);
  }, [song]);

  const [guestId] = useState(getGuestId);
  const finishPractice = useCallback(record => {
    setCurrent(record);
    setPage('results');
    window.scrollTo(0, 0);
  }, []);

  const logout = useCallback(() => {
    clearIdentity();
    setCurrent(null);
    setPage('home');
    window.scrollTo(0, 0);
  }, []);

  const props = {
    onEnded: finishPractice,
    guestId,
    nickname,
    identity,
    onLogin: saveIdentity,
    onLogout: logout,
    song,
    setSong,
    speed,
    setSpeed,
    mode,
    setMode,
    refChar,
    setRefChar,
    userChar: sameChar ? refChar : userChar,
    setUserChar,
    sameChar,
    setSameChar,
    navigate
  };

  return <>
    <Navigation page={page} identity={identity} onLogout={logout} navigate={navigate} />
    <main className={`main ${page === 'home' ? 'home-main' : page === 'practice' ? 'practice-main' : 'page-main'}`}>
      {page === 'home' && <PageLayout className="home-page" scrollLabel="홈"><Home {...props} /></PageLayout>}
      {page === 'songs' && <Songs {...props} />}
      {page === 'settings' && <Settings {...props} />}
      {page === 'characters' && <CharacterSelect {...props} />}
      {['ready', 'practice'].includes(page) && <PracticeFlow page={page} {...props} />}
      {page === 'results' && <Results {...props} current={current} />}
      {page === 'history' && <History {...props} />}
    </main>
    {page !== 'practice' && <footer className="page-footer">DANC3D <span>나의 움직임, 나의 페이스.</span></footer>}
  </>;
}