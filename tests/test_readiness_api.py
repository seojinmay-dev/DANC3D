"""임시 SQLite/TestClient로 무상태 준비 API 및 기존 연습 연결 계약을 검증한다."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4
from fastapi.testclient import TestClient
from app.main import create_app
from app.core.readiness import evaluate_readiness, REQUIRED_LANDMARKS
from app.api.serialization import json_safe
from demo.desktop_demo import COUNTDOWN_DURATION_SEC


def image_pose():
    """신체 LEFT/RIGHT와 visibility를 보존하는 합성 브라우저 객체 배열이다."""
    points = [dict(x=.5, y=.1, z=0., visibility=.9, presence=.9) for _ in range(33)]
    coordinates = {0:(.5,.12),11:(.6,.3),12:(.4,.3),13:(.72,.3),14:(.28,.3),
                   15:(.85,.3),16:(.15,.3),23:(.56,.55),24:(.44,.55),25:(.56,.7),
                   26:(.44,.7),27:(.56,.85),28:(.44,.85),31:(.58,.9),32:(.42,.9)}
    for index, (x, y) in coordinates.items():
        points[index].update(x=x, y=y)
    return points


class ReadinessAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='danc3d_readiness_api_')
        cls.db_path = Path(cls.temp.name) / 'test.db'
        cls.app = create_app(db_path=cls.db_path)
        cls.client = TestClient(cls.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.temp.cleanup()

    def body(self, points=None):
        return dict(attempt_id=str(uuid4()), image_width=640, image_height=480, image_landmarks=points)

    def post(self, body):
        response = self.client.post('/readiness', json=body)
        # 응답은 엄격한 JSON으로 직렬화 가능해야 한다.
        json.dumps(response.json(), allow_nan=False)
        return response

    def test_ready_matches_core_exactly(self):
        points = image_pose()
        body = self.body(points)
        response = self.post(body)
        self.assertEqual(response.status_code, 200)
        expected = json_safe(evaluate_readiness(points,640,480))
        result = response.json()
        self.assertEqual({key:result[key] for key in expected},expected)
        self.assertEqual(result['status'],'READY_FOR_COUNTDOWN')
        self.assertTrue(result['ready'])
        self.assertEqual(result['attempt_id'],body['attempt_id'])
        self.assertEqual(result['countdown_duration_sec'],COUNTDOWN_DURATION_SEC)

    def test_no_pose_is_valid_waiting_response(self):
        response=self.post(self.body())
        self.assertEqual(response.status_code,200)
        self.assertFalse(response.json()['ready'])
        self.assertEqual(response.json()['status'],'TPOSE_WAITING')
        self.assertEqual(response.json()['reasons'],['NO_POSE'])

    def test_incomplete_pose_conditions_are_not_request_errors(self):
        for index,key,value,reason in [(15,'y',.5,'ARMS_NOT_HORIZONTAL'),(13,'y',.34,'ARMS_NOT_EXTENDED'),
                                      (15,'x',.5,'ARMS_NOT_SPREAD'),(31,'y',1.1,'BODY_OUT_OF_FRAME'),
                                      (0,'visibility',.49,'LOW_VISIBILITY')]:
            with self.subTest(reason=reason):
                points=image_pose();points[index][key]=value
                response=self.post(self.body(points))
                self.assertEqual(response.status_code,200)
                self.assertFalse(response.json()['ready'])
                self.assertIn(reason,response.json()['reasons'])

    def test_major_visibility_required_and_presence_not_substitute(self):
        for index in REQUIRED_LANDMARKS:
            points=image_pose();del points[index]['visibility']
            response=self.post(self.body(points))
            self.assertEqual(response.status_code,422)
            self.assertEqual(response.json()['detail']['code'],'invalid_image_landmarks')

    def test_nonessential_landmarks_not_forced_visible(self):
        points=image_pose()
        for index in set(range(33))-set(REQUIRED_LANDMARKS):points[index]=None
        self.assertTrue(self.post(self.body(points)).json()['ready'])

    def test_count_and_point_format_errors(self):
        for points in [[],image_pose()[:32],image_pose()+[{}],[[0,0,.9]]*33]:
            response=self.post(self.body(points))
            self.assertEqual(response.status_code,422)
            self.assertEqual(response.json()['detail']['code'],'invalid_readiness_request')

    def test_invalid_major_coordinates_and_visibility(self):
        for key,value in [('x',True),('y','0.3'),('visibility',None),('visibility',1.1),('visibility',-.1)]:
            points=image_pose();points[11][key]=value
            self.assertEqual(self.post(self.body(points)).status_code,422)

    def test_nonfinite_inputs_do_not_leak_into_error_json(self):
        for key in ('x','y','visibility'):
            for value in (float('nan'),float('inf'),float('-inf')):
                points=image_pose();points[11][key]=value
                response=self.client.post('/readiness',content=json.dumps(self.body(points)),headers={'Content-Type':'application/json'})
                self.assertEqual(response.status_code,422)
                json.dumps(response.json(),allow_nan=False)
        body=self.body();body['image_width']=float('inf')
        response=self.client.post('/readiness',content=json.dumps(body),headers={'Content-Type':'application/json'})
        self.assertEqual(response.status_code,422)
        json.dumps(response.json(),allow_nan=False)

    def test_positive_strict_dimensions_required(self):
        for key in ('image_width','image_height'):
            for value in [0,-1,True,'640',640.5]:
                body=self.body();body[key]=value
                self.assertEqual(self.post(body).status_code,422)

    def test_missing_fields_uuid_and_world_input_rejected(self):
        for key in self.body():
            body=self.body();del body[key]
            self.assertEqual(self.post(body).status_code,422)
        body=self.body();body['attempt_id']='not-uuid'
        self.assertEqual(self.post(body).status_code,422)
        body=self.body();body['world_landmarks']=[[0,0,0]]*33
        self.assertEqual(self.post(body).status_code,422)

    def test_reset_is_new_attempt_no_retained_readiness(self):
        first=self.body(image_pose())
        self.assertTrue(self.post(first).json()['ready'])
        reset=self.body()
        result=self.post(reset).json()
        self.assertNotEqual(result['attempt_id'],first['attempt_id'])
        self.assertFalse(result['ready'])
        first['image_landmarks']=None
        self.assertFalse(self.post(first).json()['ready'])

    def test_independent_attempts_and_sessions_do_not_mutate_evaluation(self):
        manager=self.app.state.manager
        ids=[self.client.post('/sessions/start',json=dict(song_id='033_D00_001_F',guest_id=str(uuid4()),nickname='테스트')).json()['session_id'] for _ in range(2)]
        entries=[manager.get(sid) for sid in ids]
        alignments=[entry.session.alignment for entry in entries]
        a,b=self.body(image_pose()),self.body()
        for body,ready in [(a,True),(b,False),(a,True),(b,False)]:
            result=self.post(body).json()
            self.assertEqual(result['ready'],ready)
            self.assertEqual(result['attempt_id'],body['attempt_id'])
        for entry,alignment in zip(entries,alignments):
            self.assertIs(entry.session.alignment,alignment)
            self.assertEqual(entry.session.evaluation_sample_count,0)
            self.assertEqual(entry.session.valid_sample_count,0)
            self.assertIsNone(entry.last_playback)

    def test_object_axes_indices_and_unused_fields_unchanged(self):
        points=image_pose();points[13]['y']=.4
        original=deepcopy(points)
        body=self.body(points)
        with patch('app.api.readiness.evaluate_readiness',wraps=evaluate_readiness) as evaluate:
            result=self.post(body).json()
        self.assertEqual(evaluate.call_args.args[0],original)
        self.assertEqual(points,original)
        self.assertLess(result['left_elbow_angle_deg'],155)
        self.assertEqual(result['right_elbow_angle_deg'],180)
        points=image_pose()
        for point in points:point['presence']=.01;point['z']=-3
        self.assertTrue(self.post(self.body(points)).json()['ready'])

    def test_aspect_ratio_is_passed_to_core(self):
        points=image_pose();points[13]['y']=.34
        body=self.body(points);body.update(image_width=1280,image_height=480)
        self.assertTrue(self.post(body).json()['ready'])
        body.update(image_width=480,image_height=1280)
        self.assertFalse(self.post(body).json()['ready'])

    def test_degenerate_geometry_returns_waiting_and_null_angle(self):
        points=image_pose();points[13]=points[11].copy()
        result=self.post(self.body(points)).json()
        self.assertFalse(result['ready'])
        self.assertIsNone(result['left_elbow_angle_deg'])

    def test_malformed_json_has_422(self):
        response=self.client.post('/readiness',content='{bad',headers={'Content-Type':'application/json'})
        self.assertEqual(response.status_code,422)
        self.assertEqual(response.json()['detail']['code'],'invalid_readiness_request')

    def test_cors_and_openapi_registration(self):
        response=self.client.options('/readiness',headers={'Origin':'http://localhost:5173','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'Content-Type'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.headers['access-control-allow-origin'],'http://localhost:5173')
        self.assertIn('post',self.client.get('/openapi.json').json()['paths']['/readiness'])

    def test_readiness_then_existing_websocket_starts_at_zero(self):
        ready=self.post(self.body(image_pose())).json()
        self.assertTrue(ready['ready'])
        # 카운트다운은 프론트 책임이다. 서버 테스트에서 sleep으로 추정하지 않는다.
        response=self.client.post('/sessions/start',json=dict(song_id='033_D00_001_F',guest_id=str(uuid4()),nickname='웹연결'))
        self.assertEqual(response.status_code,201)
        sid=response.json()['session_id']
        entry=self.app.state.manager.get(sid)
        world=entry.session.reference.raw_world[0].tolist()
        with self.client.websocket_connect('/ws/sessions/'+sid) as ws:
            ws.send_json(dict(type='pose',playback_time_sec=0.,world_landmarks=world))
            result=ws.receive_json()
        self.assertEqual(result['playback_time_sec'],0.)
        self.assertTrue(result['pose_detected'])
        self.assertEqual(entry.session.evaluation_sample_count,1)
        self.assertEqual(self.client.post('/sessions/'+sid+'/end').status_code,200)


if __name__=='__main__':
    unittest.main()
