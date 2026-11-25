"""단일 프로세스의 독립된 DanceSession과 API 수명 주기를 관리한다."""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from threading import RLock
import time
from uuid import uuid4
import numpy as np
from app.services.dance_session import DanceSession, DEFAULT_DATA_DIR
from demo.desktop_demo import ReferenceImages


class ServiceError(Exception):
    """REST와 WebSocket에서 같은 코드로 전달하는 서비스 오류."""
    def __init__(self, code, status=409):
        self.code, self.status = code, status
        super().__init__(code)


@dataclass
class Entry:
    """동일 세션의 update/제어/저장 경쟁을 막는 잠금을 가진 상태."""
    session_id: str
    guest_id: str
    nickname: str
    playback_speed: float
    mode: str
    session: DanceSession
    touched: float
    last_playback: object = None
    record: object = None
    connected: bool = False
    lock: object = field(default_factory=RLock)


class SessionManager:
    """유휴 TTL과 최대 세션 수로 메모리를 제한한다. 만료는 저장하지 않는다."""
    def __init__(self, database, data_dir=DEFAULT_DATA_DIR, ttl=1800, max_sessions=100, clock=time.monotonic):
        self.database, self.data_dir = database, Path(data_dir).resolve()
        if not math.isfinite(ttl) or ttl <= 0 or max_sessions < 1:
            raise ValueError("TTL and capacity must be positive")
        self.ttl, self.max_sessions, self.clock = ttl, max_sessions, clock
        self.entries, self.images = {}, {}
        self.lock = RLock()
        with (self.data_dir / 'songs.json').open(encoding='utf-8') as handle:
            self.catalog = json.load(handle)

    def song(self, song_id):
        """확인된 catalog 항목만 선택한다."""
        for song in self.catalog:
            if song['id'] == song_id:
                return song
        raise ServiceError('song_not_found', 404)

    def public_song(self, song):
        """개인 경로 없이 곡 정보와 Reference 파일 준비 여부를 제공한다."""
        return dict(
            song_id=song['id'],
            title=song['title'],
            artist=song['artist'],
            duration_sec=song['duration_sec'],
            audio_url=song.get('audio_url', f"/audio/{song['id']}.mp3"),
            audio_start_sec=song.get('audio_start_sec', 0.0),
            reference_ready=all((self.data_dir / song[k]).is_file() for k in ('reference_npz', 'reference_metadata')),
        )

    def cleanup(self):
        """진행 중 작업 및 연결은 보존하고 유휴 세션을 저장 없이 제거한다."""
        with self.lock:
            for key, entry in list(self.entries.items()):
                with entry.lock:
                    if not entry.connected and self.clock() - entry.touched > self.ttl:
                        del self.entries[key]

    def get(self, session_id):
        """존재/만료를 검사하고 접근 시 유휴 시간을 갱신한다."""
        self.cleanup()
        with self.lock:
            entry = self.entries.get(session_id)
            if entry is None:
                raise ServiceError('session_not_found_or_expired', 404)
            entry.touched = self.clock()
            return entry

    def start(self, song_id, guest_id, nickname, playback_speed=1.0, mode='video'):
        """검증 후 새 UUID와 독립 계산 상태를 생성한다."""
        song = self.song(song_id)
        if (not isinstance(guest_id, str) or not guest_id.strip() or len(guest_id) > 128
                or not isinstance(nickname, str) or not nickname.strip() or len(nickname) > 80
                or isinstance(playback_speed, bool) or playback_speed not in (.5, .75, 1., 1.25, 1.5) or mode != 'video'):
            raise ServiceError('invalid_session_request', 422)
        with self.lock:
            self.cleanup()
            if len(self.entries) >= self.max_sessions:
                raise ServiceError('session_capacity_reached', 503)
            session = DanceSession(song_id, self.data_dir)
            entry = Entry(str(uuid4()), guest_id, nickname, playback_speed, mode, session, self.clock())
            self.entries[entry.session_id] = entry
        return self.info(entry)

    def info(self, entry):
        """현재 API 상태를 반환하며 배속으로 시간을 추측하지 않는다."""
        return dict(session_id=entry.session_id, song_id=entry.session.song_id, playback_speed=entry.playback_speed,
                    duration_sec=self.song(entry.session.song_id)['duration_sec'], status=entry.session.state)

    def update(self, session_id, message):
        """원본 World를 검증하여 기존 update에 전달하며 역행은 reset을 요구한다."""
        entry = self.get(session_id)
        with entry.lock:
            if entry.session.state == 'ended':
                raise ServiceError('session_ended')
            if not isinstance(message, dict) or message.get('type') != 'pose' or 'world_landmarks' not in message:
                raise ServiceError('invalid_pose_message', 422)
            playback = message.get('playback_time_sec')
            if isinstance(playback, bool) or not isinstance(playback, (int, float)) or not math.isfinite(playback) or playback < 0:
                raise ServiceError('invalid_playback_time', 422)
            world = message['world_landmarks']
            if world is not None:
                if (not isinstance(world, list) or len(world) != 33
                        or any(not isinstance(row, list) or len(row) != 3 for row in world)
                        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for row in world for v in row)):
                    raise ServiceError('invalid_world_landmarks', 422)
                world = np.asarray(world, dtype=np.float64)
            if entry.last_playback is not None and playback < entry.last_playback:
                raise ServiceError('playback_reset_required')
            result = entry.session.update(playback, world)
            if entry.session.state == 'running':
                entry.last_playback = playback
            return dict(result, session_id=session_id)

    # [수정] user_id를 선택적으로 받을 수 있도록 매개변수를 추가했습니다.
    def control(self, session_id, action, user_id=None):
        """상태 제어를 직렬화하고 End 재송에는 저장된 결과를 반환한다."""
        entry = self.get(session_id)
        with entry.lock:
            if action == 'end':
                if entry.record is None:
                    # [수정] database.save 호출 시 전달받은 user_id를 함께 넘겨줍니다.
                    entry.record = self.database.save(entry, entry.session.end(), user_id=user_id)
                return dict(self.info(entry), record=entry.record)
            if entry.session.state == 'ended':
                raise ServiceError('session_ended')
            if action not in ('pause', 'resume', 'reset'):
                raise ServiceError('invalid_control', 422)
            getattr(entry.session, action)()
            if action == 'reset':
                entry.last_playback = None
            return dict(self.info(entry), playback_time_sec=0.0) if action == 'reset' else self.info(entry)

    def reference_frame(self, song_id, seconds):
        """데모의 NFC/NFD 및 원본 frame_index 연결을 재사용한다."""
        song = self.song(song_id)
        with self.lock:
            if song_id not in self.images:
                reference = DanceSession(song_id, self.data_dir).reference
                self.images[song_id] = ReferenceImages(song, reference, self.data_dir)
            row, frame, path = self.images[song_id].at_playback(seconds)
        if path is None or not path.is_file():
            raise ServiceError('reference_frame_not_found', 404)
        try:
            path.resolve().relative_to((self.data_dir / 'raw').resolve())
        except ValueError:
            raise ServiceError('reference_frame_unavailable', 503)
        return row, frame, path