import { useState, useEffect } from 'react';
import './Home.css';
import {loginUser,registerUser} from '../services/api.js';
import {clearIdentity} from '../services/identity.js';

export default function Home({ navigate, identity, onLogin, onLogout }) {
  const [logoText, setLogoText] = useState('DANCED');
  
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [isLoginMode, setIsLoginMode] = useState(true);

  // 로그인 상태 확인
  const loggedInUser = identity?.username;

  useEffect(() => {
    const timer = setTimeout(() => {
      setLogoText('DANC3D');
    }, 1500);
    return () => clearTimeout(timer);
  }, []);

  // 홈 화면용 로그아웃 핸들러
  const handleLogout = () => {
    onLogout();
    alert('로그아웃 되었습니다.');
  };

  const handleStart = async (event) => {
    event.preventDefault();
    if (!userId.trim() || !password.trim()) return;
    try {
      if (isLoginMode) {
        const data = await loginUser(userId, password);
        onLogin({ token: data.access_token, username: data.nickname || userId.trim() });
        navigate('songs');
      } else {
        await registerUser(userId, password);
        alert('회원가입이 완료되었습니다! 로그인해주세요.');
        setIsLoginMode(true);
        setPassword('');
      }
    } catch (error) {
      alert(error instanceof TypeError ? '서버 연결 중 문제가 발생했습니다.' : error.message);
    }
  };

  return (
    <section className="home">
      <div className="hero-section">
        <div className="eyebrow">YOUR NEXT MOVE</div>
        <h1 className="logo-title">
          <span className="purple-text">DANC</span>
          <span className={`animated-letter ${logoText === 'DANC3D' ? 'flip-to-3' : ''}`}>
            {logoText === 'DANC3D' ? '3' : 'E'}
          </span>
          <span className="purple-text">D</span>
        </h1>
        <p className="intro">
          실시간으로 따라하고, 바로 피드백 받는<br />새로운 댄스 연습 경험
        </p>
      </div>

      <div className="form-card">
        {loggedInUser ? (
          /* 로그인 상태일 때 보여줄 프로필 카드 */
          <div style={{ textAlign: 'center', padding: '20px 0' }}>
            <h2 style={{ fontSize: '20px', marginBottom: '24px', color: '#1a1f2c' }}>
              🎉 {loggedInUser}님, 환영합니다!
            </h2>
            <button className="start-button" onClick={() => navigate('songs')}>
              <span>연습하러 가기 →</span>
            </button>
            <button className="history-button" onClick={handleLogout} style={{ marginTop: '12px' }}>
              로그아웃
            </button>
          </div>
        ) : (
          /* 로그인 상태가 아닐 때 보여줄 기존 입력 폼 */
          <>
            <form onSubmit={handleStart}>
              <input
                id="userId"
                type="text"
                placeholder="아이디 입력"
                value={userId}
                onChange={e => setUserId(e.target.value)}
                required
              />
              <input
                id="password"
                type="password"
                placeholder="비밀번호 입력"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
              />
              <button
                type="submit"
                className="start-button"
                disabled={!userId.trim() || !password.trim()}
              >
                <span>{isLoginMode ? '연습 시작 →' : '회원 가입'}</span>
              </button>
            </form>

            <button 
              className="history-button" 
              onClick={() => {
                if (isLoginMode) {
                  clearIdentity();
                  navigate('songs'); 
                } else {
                  setIsLoginMode(true); 
                }
              }}
            >
               {isLoginMode ? '비회원으로 이용하기' : '로그인 하러가기'}
            </button>

            <div style={{ textAlign: 'center', marginTop: '16px' }}>
              <span 
                style={{ fontSize: '13px', color: '#6b707b', cursor: 'pointer', textDecoration: 'underline' }}
                onClick={() => {
                  if (isLoginMode) {
                    setIsLoginMode(false); 
                  } else {
                    navigate('songs'); 
                  }
                }}
              >
                {isLoginMode ? '회원 가입' : '비회원으로 이용하기'}
              </span>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
