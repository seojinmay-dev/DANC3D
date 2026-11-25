"""Account records span browsers; guest queries and invalid credentials remain separate."""
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from jose import jwt
from app.main import create_app
from app.services.auth import ALGORITHM, SECRET_KEY, create_access_token


class AuthRecordsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.app = create_app(db_path=Path(self.temp.name) / 'test.db')
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        db = self.app.state.manager.database
        with db.connect() as conn:
            conn.executemany('INSERT INTO auth_users (id,username,password_hash) VALUES (?,?,?)',
                             [(1,'one','unused'),(2,'two','unused')])
            conn.executemany('INSERT INTO users VALUES (?,?,?)', [('old','old','date'),('other','other','date')])
            for sid, guest, owner in [('one-old','old',1),('one-other','other',1),('two-old','old',2),('guest-old','old',None)]:
                conn.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)',
                             (sid,guest,'033_D00_001_F',1,'video',.8,1,0,'2026-11-17T00:00:00Z',owner))

    def headers(self, user_id):
        return {'Authorization':'Bearer '+create_access_token({'sub':str(user_id)})}

    def test_member_records_span_browsers_and_guest_excludes_members(self):
        rows = self.client.get('/records/new-browser',headers=self.headers(1)).json()
        self.assertEqual({row['session_id'] for row in rows}, {'one-old','one-other'})
        self.assertTrue(all(row['user_id']==1 for row in rows))
        rows = self.client.get('/records/old',headers=self.headers(2)).json()
        self.assertEqual([row['session_id'] for row in rows],['two-old'])
        rows = self.client.get('/records/old').json()
        self.assertEqual([row['session_id'] for row in rows],['guest-old'])
        by_song = self.client.get('/records/new-browser/songs/033_D00_001_F',headers=self.headers(1)).json()
        self.assertEqual(len(by_song['records']),2)
        self.assertEqual(by_song['best_score_100'],80)

    def test_invalid_expired_and_deleted_accounts_return_401_not_guest_records(self):
        expired = jwt.encode({'sub':'1','exp':1},SECRET_KEY,algorithm=ALGORITHM)
        missing_exp = jwt.encode({'sub':'1'},SECRET_KEY,algorithm=ALGORITHM)
        tokens = ['broken',expired,missing_exp,create_access_token({'sub':'999'})]
        for token in tokens:
            headers={'Authorization':'Bearer '+token}
            self.assertEqual(self.client.get('/records/old',headers=headers).status_code,401)
            response=self.client.post('/sessions/start',headers=headers,json={
                'song_id':'033_D00_001_F','guest_id':'old','nickname':'one'})
            self.assertEqual(response.status_code,401)
