"""SQLite 테이블 정의. 점수는 RAW 평균 0~1이며 평가 불가는 NULL이다."""
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
 guest_id TEXT PRIMARY KEY, nickname TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS songs (
 song_id TEXT PRIMARY KEY, title TEXT NOT NULL, artist TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS records (
 session_id TEXT PRIMARY KEY,
 guest_id TEXT NOT NULL REFERENCES users(guest_id),
 song_id TEXT NOT NULL REFERENCES songs(song_id),
 playback_speed REAL NOT NULL, mode TEXT NOT NULL CHECK(mode='video'),
 mean_raw_final_score REAL CHECK(mean_raw_final_score BETWEEN 0 AND 1),
 valid_sample_count INTEGER NOT NULL, recovery_trigger_count INTEGER NOT NULL,
 ended_at TEXT NOT NULL,
 user_id INTEGER REFERENCES auth_users(id)
);
CREATE INDEX IF NOT EXISTS records_guest_song ON records(guest_id, song_id, ended_at);
CREATE INDEX IF NOT EXISTS records_user_song ON records(user_id, song_id, ended_at);
"""