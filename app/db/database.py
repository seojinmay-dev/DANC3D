"""연습 요약 및 회원 인증 정보를 저장하는 표준 sqlite3 저장소."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import math
import sqlite3
from app.db.models import SCHEMA


class Database:
    """호출별 연결과 트랜잭션을 사용하며 session_id로 중복 저장을 막는다."""
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def connect(self):
        """외래 키를 켠 연결을 커밋 또는 롤백하고 항상 닫는다."""
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self, songs):
        """시작 시 DB와 테이블을 생성하고 확인된 곡 정보를 동기화한다."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            
            # [추가] 기존 비회원 users 테이블과 겹치지 않도록 auth_users 테이블 자동 생성
            db.execute('''
                CREATE TABLE IF NOT EXISTS auth_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                )
            ''')
            # 기존 기록을 유지하며 회원 연결 컬럼을 인덱스 생성 전에 추가한다.
            columns = {row['name'] for row in db.execute('PRAGMA table_info(records)')}
            if columns and 'user_id' not in columns:
                db.execute('ALTER TABLE records ADD COLUMN user_id INTEGER REFERENCES auth_users(id)')
            db.executescript(SCHEMA)
            
            db.executemany('INSERT INTO songs VALUES (?,?,?) ON CONFLICT(song_id) DO UPDATE SET title=excluded.title, artist=excluded.artist',
                           [(s['id'], s['title'], s['artist']) for s in songs])

    def save(self, entry, summary, user_id=None):
        """사용자 및 종료 요약을 한 트랜잭션에서 저장하고 저장된 기록을 반환한다."""
        now = datetime.now(timezone.utc).isoformat()
        score = summary['mean_raw_final_score']
        score = float(score) if math.isfinite(score) else None
        with self.connect() as db:
            db.execute('INSERT INTO users VALUES (?,?,?) ON CONFLICT(guest_id) DO UPDATE SET nickname=excluded.nickname',
                       (entry.guest_id, entry.nickname, now))
            
            # [수정] 끝에 추가된 user_id 컬럼을 반영하여 10개의 값을 전달 (user_id가 없으면 None(NULL)으로 저장됨)
            db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(session_id) DO NOTHING',
                       (entry.session_id, entry.guest_id, entry.session.song_id, entry.playback_speed,
                        entry.mode, score, summary['valid_sample_count'], summary['recovery_trigger_count'], now, user_id))
            return dict(db.execute('SELECT * FROM records WHERE session_id=?', (entry.session_id,)).fetchone())

    def records(self, guest_id, song_id=None, user_id=None):
        """회원인 경우 user_id로, 비회원인 경우 guest_id를 기준으로 최근 기록부터 반환한다."""
        sql = 'SELECT r.*, s.title, s.artist FROM records r JOIN songs s USING(song_id) WHERE '
        args = []
        
        # [수정] user_id가 전달되면 회원의 기록만, 없으면 기기(guest_id)의 기록만 조회
        if user_id is not None:
            sql += 'user_id=?'
            args.append(user_id)
        else:
            sql += 'guest_id=? AND r.user_id IS NULL'
            args.append(guest_id)
            
        if song_id is not None:
            sql += ' AND song_id=?'
            args.append(song_id)
            
        with self.connect() as db:
            rows = db.execute(sql + ' ORDER BY ended_at DESC, session_id DESC', args).fetchall()
        return [dict(row, score_100=None if row['mean_raw_final_score'] is None else row['mean_raw_final_score'] * 100) for row in rows]
