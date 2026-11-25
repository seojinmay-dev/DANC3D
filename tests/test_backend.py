"""임시 SQLite로 API 서비스 계약을 검증한다. 카메라/네트워크는 사용하지 않는다."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from uuid import UUID
import numpy as np
from app.api.serialization import json_safe
from app.db.database import Database
from app.services.session_manager import SessionManager, ServiceError

SONG = '033_D00_001_F'


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.temp.name) / 'test.db')
        self.manager = SessionManager(self.db)
        self.db.initialize(self.manager.catalog)

    def tearDown(self):
        self.temp.cleanup()

    def start(self, guest='guest-a', song=SONG):
        return self.manager.start(song, guest, '같은이름')['session_id']

    def pose(self, sid, seconds=0, world=None):
        return self.manager.update(sid, dict(type='pose', playback_time_sec=seconds, world_landmarks=world))

    def test_catalog_ten_and_private_paths(self):
        self.assertEqual(len(self.manager.catalog), 10)
        result = self.manager.public_song(self.manager.song(SONG))
        self.assertTrue(result['reference_ready'])
        self.assertNotIn('raw_source_dir', result)
        with self.assertRaises(ServiceError) as error:
            self.manager.song('missing')
        self.assertEqual(error.exception.status, 404)

    def test_uuid_independent_sessions(self):
        a, b = self.start(), self.start()
        UUID(a)
        self.assertNotEqual(a, b)
        self.assertIsNot(self.manager.get(a).session.alignment, self.manager.get(b).session.alignment)
        self.pose(a)
        self.assertEqual(self.manager.get(b).session.evaluation_sample_count, 0)

    def test_normal_world_and_json(self):
        sid = self.start()
        world = self.manager.get(sid).session.reference.raw_world[0].tolist()
        result = self.pose(sid, world=world)
        self.assertTrue(result['pose_detected'])
        self.assertEqual(result['session_id'], sid)
        self.assertEqual(result['valid_user_bone_count'], 8)
        json.dumps(json_safe(result), allow_nan=False)

    def test_shape_rejected_without_mutation(self):
        sid = self.start()
        for world in ([[0, 0, 0]], [['bad'] * 3] * 33, [[True, 0, 0]] * 33):
            with self.assertRaises(ServiceError):
                self.pose(sid, world=world)
        self.assertEqual(self.manager.get(sid).session.evaluation_sample_count, 0)

    def test_missing_pose_null_score(self):
        result = json_safe(self.pose(self.start()))
        self.assertFalse(result['pose_detected'])
        self.assertIsNone(result['raw_final_score'])

    def test_nested_numpy_serialization(self):
        result = json_safe({'v': [np.int64(3), np.float32(.5), np.bool_(True), np.array([np.nan, np.inf, -np.inf])]})
        self.assertEqual(result, {'v': [3, .5, True, [None] * 3]})
        json.dumps(result, allow_nan=False)

    def test_pause_resume_reset_and_seek(self):
        sid = self.start()
        entry = self.manager.get(sid)
        self.pose(sid, 1)
        alignment = entry.session.alignment
        self.manager.control(sid, 'pause')
        self.pose(sid, 1)
        self.assertEqual(entry.session.evaluation_sample_count, 1)
        self.manager.control(sid, 'resume')
        self.assertIs(entry.session.alignment, alignment)
        with self.assertRaises(ServiceError) as error:
            self.pose(sid, 0)
        self.assertEqual(error.exception.code, 'playback_reset_required')
        self.manager.control(sid, 'reset')
        self.assertEqual(entry.session.evaluation_sample_count, 0)
        self.assertIsNot(entry.session.alignment, alignment)
        self.pose(sid, 0)

    def test_idempotent_end_and_null_storage(self):
        sid = self.start()
        one = self.manager.control(sid, 'end')
        two = self.manager.control(sid, 'end')
        self.assertEqual(one, two)
        rows = self.db.records('guest-a')
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]['mean_raw_final_score'])
        with self.assertRaises(ServiceError):
            self.manager.control(sid, 'reset')

    def test_users_song_filter_and_best_score(self):
        for guest, song, score in [('a', SONG, .6), ('a', SONG, .9), ('b', SONG, .8), ('a', '024_C00_001_F', .7)]:
            sid = self.start(guest, song)
            entry = self.manager.get(sid)
            # 집계 저장 검증: 계산식 재구현 대신 종료 요약 입력을 통제한다.
            summary = entry.session.end()
            summary.update(mean_raw_final_score=score, valid_sample_count=1)
            self.db.save(entry, summary)
        rows = self.db.records('a', SONG)
        self.assertEqual(len(rows), 2)
        self.assertEqual(max(row['score_100'] for row in rows), 90)
        self.assertEqual(len(self.db.records('b')), 1)
        self.assertEqual(len(self.db.records('a')), 3)

    def test_unknown_session_and_invalid_requests(self):
        with self.assertRaises(ServiceError) as error:
            self.manager.get('missing')
        self.assertEqual(error.exception.status, 404)
        for speed, mode in [(2, 'video'), (True, 'video'), (1, 'character')]:
            with self.assertRaises(ServiceError):
                self.manager.start(SONG, 'a', 'name', speed, mode)

    def test_ttl_no_automatic_record(self):
        tick = [0]
        self.manager.clock = lambda: tick[0]
        sid = self.start()
        tick[0] = 1801
        with self.assertRaises(ServiceError):
            self.manager.get(sid)
        self.assertEqual(self.db.records('guest-a'), [])

    def test_reference_row_original_frame(self):
        row, frame, path = self.manager.reference_frame(SONG, 1.25)
        images = self.manager.images[SONG]
        self.assertEqual(frame, int(images.reference.frame_indices[row]))
        self.assertEqual(row, int(np.argmin(abs(images.reference.times - 1.25))))
        self.assertTrue(path.is_file())
        self.assertEqual(int(path.stem.rsplit('_', 1)[1]), frame)

    def test_unmatched_ringdingdong_excluded(self):
        self.manager.reference_frame('008_A00_001_F', 0)
        images = self.manager.images['008_A00_001_F']
        self.assertEqual(len(images.excluded_frame_indices), 2)
        self.assertTrue(all(frame not in images.paths for frame in images.excluded_frame_indices))


HAS_HTTP = all(importlib.util.find_spec(name) for name in ('fastapi', 'httpx', 'starlette'))


@unittest.skipUnless(HAS_HTTP, 'FastAPI/Starlette/httpx 미설치: HTTP/WebSocket 통합 검증 불가')
class HTTPTests(unittest.TestCase):
    def test_http_websocket_contract(self):
        from fastapi.testclient import TestClient
        from app.main import create_app
        with tempfile.TemporaryDirectory() as temp:
            with TestClient(create_app(db_path=Path(temp) / 'test.db')) as client:
                self.assertEqual(client.get('/health').json(), {'status': 'ok'})
                self.assertEqual(len(client.get('/songs').json()), 10)
                self.assertEqual(client.get('/songs/missing').status_code, 404)
                self.assertEqual(client.get('/songs/' + SONG).status_code, 200)
                self.assertEqual(client.get('/songs/' + SONG + '/reference-frame?time_sec=1.25').status_code, 200)
                response = client.post('/sessions/start', json=dict(song_id=SONG, guest_id='a', nickname='서진'))
                self.assertEqual(response.status_code, 201)
                sid = response.json()['session_id']
                with client.websocket_connect('/ws/sessions/' + sid) as ws:
                    ws.send_json(dict(type='pose', playback_time_sec=0, world_landmarks=None))
                    self.assertIsNone(ws.receive_json()['raw_final_score'])
                    ws.send_json(dict(type='pose', playback_time_sec=0, world_landmarks=[]))
                    self.assertEqual(ws.receive_json()['code'], 'invalid_world_landmarks')
                for action in ('pause', 'resume', 'reset', 'end', 'end'):
                    self.assertEqual(client.post('/sessions/' + sid + '/' + action).status_code, 200)
                self.assertEqual(len(client.get('/records/a').json()), 1)
                self.assertIsNone(client.get('/records/a/songs/' + SONG).json()['best_score_100'])
                self.assertEqual(client.post('/sessions/missing/pause').status_code, 404)
