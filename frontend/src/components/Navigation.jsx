import './Navigation.css';

export default function Navigation({ navigate, page, identity, onLogout }) {
  const loggedInUser = identity?.username;

  const handleLogout = () => {
    onLogout();
    alert('로그아웃 되었습니다.');
  };

  return (
    <header className="navigation">
      <div className="nav-inner">
        <button className="brand" onClick={() => navigate('home')}>
          <span className="brand-text">
            DANC<span className="brand-highlight">3</span>D
          </span>
        </button>

        <nav aria-label="주 메뉴">
          <button 
            className={page === 'home' ? 'active' : ''} 
            onClick={() => navigate('home')}
          >
            홈
          </button>
          <button 
            className={page === 'songs' ? 'active' : ''} 
            onClick={() => navigate('songs')}
          >
            연습하기
          </button>
          <button 
            className={page === 'history' ? 'active' : ''} 
            onClick={() => navigate('history')}
          >
            히스토리
          </button>
        </nav>

        <span className="profile" title={loggedInUser || '게스트'} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <circle cx="12" cy="8" r="3" />
            <path d="M5 21v-3a7 7 0 0114 0v3" />
          </svg>
          <span>{loggedInUser || '게스트'}</span>
          
          {/* 로그인 상태일 때만 로그아웃 버튼 표시 */}
          {loggedInUser && (
            <button 
              onClick={handleLogout}
              style={{
                marginLeft: '8px', padding: '4px 8px', fontSize: '12px', 
                background: '#f1f1f1', border: 'none', borderRadius: '4px', cursor: 'pointer'
              }}
            >
              로그아웃
            </button>
          )}
        </span>
      </div>
    </header>
  );
}
