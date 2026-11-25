"""임시 DB/127.0.0.1 서버로 실행하는 수동 HTTP/WebSocket 검증.

실행: python tests/smoke_test_api.py
기존 실행 서버를 사용하지 않으며 운영 DB, 카메라, 모델 추론에 접근하지 않는다.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SONG_ID = '033_D00_001_F'
REQUIRED_FIELDS = (
    'pose_detected', 'aligned_reference_index', 'overall_similarity',
    'bone_similarities', 'displayed_feedback', 'raw_final_score',
    'displayed_score', 'recovery_trigger_count', 'part_similarities',
    'valid_user_bone_count', 'aligned_reference_time_sec', 'timing_offset_sec',
    'bone_status', 'priority_error_bone', 'error_bone_count',
    'raw_final_feedback', 'torso_yaw_error_deg', 'recovery_triggered',
)


def check(condition, message):
    """최적화 옵션에서도 검증 실패를 명확하게 전달한다."""
    if not condition:
        raise AssertionError(message)


def strict_json(text):
    """비표준 NaN/Infinity뿐 아니라 과도한 지수의 Infinity도 거부한다."""
    def reject(value):
        raise ValueError('비표준 JSON 숫자: ' + value)

    def finite_float(value):
        number = float(value)
        check(math.isfinite(number), 'JSON 비유한 실수')
        return number

    return json.loads(text, parse_constant=reject, parse_float=finite_float)


def compare(actual, expected, path='result'):
    """기존 세션의 전체 결과와 비교하여 축/좌우/시간 변환 여부도 검증한다."""
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and actual.keys() == expected.keys(), path + ': 필드 불일치')
        for key, value in expected.items():
            compare(actual[key], value, path + '.' + key)
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), path + ': 배열 불일치')
        for index, value in enumerate(expected):
            compare(actual[index], value, '{}[{}]'.format(path, index))
    elif isinstance(expected, float):
        check(isinstance(actual, (int, float)) and not isinstance(actual, bool)
              and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), path + ': 계산값 불일치')
    else:
        check(type(actual) is type(expected) and actual == expected, path + ': 값 불일치')


def fingerprint():
    """소스/Reference/모델과 운영 DB의 지문을 읽기만 한다. raw는 쓰기 경로가 없다."""
    paths = list((ROOT / 'app').rglob('*.py')) + list((ROOT / 'data/references').glob('*'))
    paths += list((ROOT / 'models/mediapipe').glob('*.task'))
    paths += [ROOT / 'data/songs.json', ROOT / 'demo/desktop_demo.py']
    paths += list((ROOT / 'data').glob('danc3d.db*'))
    # 연구 코드 경로를 smoke 스크립트에 하드코딩하지 않는다.
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


async def exercise(base_url, db_path, report, client, connect, np):
    """실제 소켓을 통해 정상/미검출/오류/상태 제어/저장 계약을 검사한다."""
    from app.services.dance_session import DanceSession
    from app.api.serialization import json_safe

    endpoints = report['http_apis'] = []

    def request(method, path, body=None, status=200):
        response = client.request(method, base_url + path, json=body)
        check(response.status_code == status, '{} {}: {} {}'.format(method, path, response.status_code, response.text[:300]))
        data = strict_json(response.text)
        endpoints.append({'method': method, 'path': path, 'status': response.status_code})
        return data

    check(request('GET', '/health') == {'status': 'ok'}, 'health 실패')
    songs = request('GET', '/songs')
    check(len(songs) == 10, '10곡 목록 실패')
    song = request('GET', '/songs/' + SONG_ID)
    check(song['song_id'] == SONG_ID and song['reference_ready'], '곡 조회 실패')
    guest_id = str(uuid4())
    report['guest_id'] = guest_id
    start = request('POST', '/sessions/start', dict(song_id=SONG_ID, guest_id=guest_id,
                    nickname='실통신 검증', playback_speed=1.5, mode='video'), status=201)
    session_id = start['session_id']
    report['session_id'] = session_id
    check(start['status'] == 'running' and start['playback_speed'] == 1.5, '시작 상태 실패')
    oracle = DanceSession(SONG_ID)
    valid_rows = np.flatnonzero(np.isfinite(oracle.reference.raw_world).all(axis=(1, 2))
                               & oracle.reference.bone_valid_mask.all(axis=1))
    # 서로 다른 row와 원본 frame 번호를 확인할 수 있게 시간 간격을 두어 선택한다.
    rows = valid_rows[::5][:25]
    check(len(rows) == 25, '유효 NPZ 입력 25개 확보 실패')
    evaluations = report['evaluations'] = []
    raw_scores = []
    ws_url = base_url.replace('http://', 'ws://') + '/ws/sessions/' + session_id
    async with connect(ws_url, origin='http://localhost:5173', open_timeout=10,
                       close_timeout=5, max_size=2 ** 22) as ws:
        report['websocket_connected'] = True

        async def pose(seconds, world, kind):
            message = dict(type='pose', playback_time_sec=float(seconds), world_landmarks=world)
            await ws.send(json.dumps(message, allow_nan=False))
            result = strict_json(await asyncio.wait_for(ws.recv(), timeout=15))
            check(all(key in result for key in REQUIRED_FIELDS), '필수 평가 필드 누락')
            expected = json_safe(dict(oracle.update(float(seconds), world), session_id=session_id))
            compare(result, expected)
            if result['raw_final_score'] is not None:
                raw_scores.append(result['raw_final_score'])
            aligned = result['aligned_reference_index']
            if aligned >= 0:
                check(result['aligned_reference_frame_index'] == int(oracle.reference.frame_indices[aligned]), 'row/frame_index 혼동')
                check(result['aligned_reference_time_sec'] == float(oracle.reference.times[aligned]), '정렬 콘텐츠 시간 불일치')
            evaluations.append(dict(kind=kind, playback_time_sec=result['playback_time_sec'],
                                    **{key: result[key] for key in REQUIRED_FIELDS},
                                    aligned_reference_frame_index=result['aligned_reference_frame_index']))
            return result

        for index, row in enumerate(rows[:24]):
            seconds = oracle.reference.times[row]
            world = oracle.reference.raw_world[row].tolist()
            result = await pose(seconds, world, 'normal')
            check(result['pose_detected'] and result['valid_user_bone_count'] == 8, '정상 Pose 미검출')
            if index == 11:
                paused = request('POST', '/sessions/' + session_id + '/pause')
                check(paused['status'] == 'paused', 'pause 실패')
                oracle.pause()
                paused_result = await pose(seconds, world, 'paused')
                check(not paused_result['evaluation_performed'] and not paused_result['sample_recorded'], 'paused 샘플 집계')
                check(paused_result['raw_final_score'] is None and paused_result['aligned_reference_index'] == -1, 'paused 가짜 평가')
                resumed = request('POST', '/sessions/' + session_id + '/resume')
                check(resumed['status'] == 'running', 'resume 실패')
                oracle.resume()

        last_time = float(oracle.reference.times[rows[23]])
        missing = await pose(last_time + .001, None, 'missing')
        check(not missing['pose_detected'] and missing['valid_user_bone_count'] == 0, 'null Pose 검출 오류')
        check(missing['raw_final_score'] is None and missing['overall_similarity'] is None, '미검출 가짜 점수')
        check(missing['aligned_reference_index'] == -1 and missing['aligned_reference_frame_index'] == -1
              and missing['aligned_reference_time_sec'] is None, '미검출 과거 정렬 재사용')
        check(all(value is None for value in missing['bone_similarities'].values()), '미검출 유사도 null 실패')
        await ws.send(json.dumps(dict(type='pose', playback_time_sec=last_time + .002, world_landmarks=[[0, 0, 0]])))
        error = strict_json(await asyncio.wait_for(ws.recv(), timeout=15))
        check(error == {'type': 'error', 'code': 'invalid_world_landmarks', 'status': 422}, 'shape 오류 응답 불일치')
        report['invalid_shape_response'] = error
        await ws.send(json.dumps(dict(type='pose', playback_time_sec=0, world_landmarks=None)))
        error = strict_json(await asyncio.wait_for(ws.recv(), timeout=15))
        check(error == {'type': 'error', 'code': 'playback_reset_required', 'status': 409}, '역행 오류 응답 불일치')
        report['backward_time_response'] = error
        row = rows[24]
        await pose(oracle.reference.times[row], oracle.reference.raw_world[row].tolist(), 'normal_after_errors')

    # 연결 종료만으로 저장되지 않는지 검사한다.
    check(request('GET', '/records/' + guest_id) == [], '연결 해제로 조기 저장됨')
    ended = request('POST', '/sessions/' + session_id + '/end')
    check(ended['status'] == 'ended', 'end 실패')
    summary = oracle.end()
    record = ended['record']
    check(len(raw_scores) > 0, '유효 RAW 점수 없음')
    mean = sum(raw_scores) / len(raw_scores)
    check(record['valid_sample_count'] == len(raw_scores) == summary['valid_sample_count'], 'RAW 유효 표본 수 불일치')
    check(math.isclose(record['mean_raw_final_score'], mean, abs_tol=1e-12), 'RAW 평균 집계 실패')
    check(record['recovery_trigger_count'] == summary['recovery_trigger_count'], 'Recovery 집계 실패')
    compare(request('POST', '/sessions/' + session_id + '/end'), ended)
    records = request('GET', '/records/' + guest_id)
    check(len(records) == 1 and records[0]['session_id'] == session_id, '기록 조회/중복 방지 실패')
    compare(records[0]['score_100'], mean * 100)
    by_song = request('GET', '/records/' + guest_id + '/songs/' + SONG_ID)
    compare(by_song['records'], records)
    compare(by_song['best_score_100'], mean * 100)
    with sqlite3.connect('file:{}?mode=ro'.format(db_path), uri=True) as db:
        stored = db.execute('SELECT mean_raw_final_score, valid_sample_count, recovery_trigger_count FROM records WHERE session_id=?', (session_id,)).fetchall()
        check(len(stored) == 1, 'SQLite 중복 End 저장')
        check(db.execute('SELECT guest_id FROM users').fetchall() == [(guest_id,)], 'SQLite 사용자 저장')
        check(math.isclose(stored[0][0], mean, abs_tol=1e-12) and stored[0][1] == len(raw_scores), 'SQLite 집계 값 불일치')
        report['sqlite'] = dict(record_count=len(stored), mean_raw_final_score=stored[0][0],
                                score_100=mean * 100, valid_sample_count=stored[0][1], recovery_trigger_count=stored[0][2])
    report.update(normal_pose_frames=25, paused_pose_frames=1, missing_pose_frames=1,
                  invalid_shape_frames=1, backward_time_frames=1, duplicate_end_prevented=True,
                  coordinate_and_time_parity=True, strict_json=True,
                  feedback_values=sorted({item['displayed_feedback'] for item in evaluations}),
                  final_displayed_score=evaluations[-1]['displayed_score'],
                  raw_score_range=[min(raw_scores), max(raw_scores)],
                  pause_resume_end='PASS')


def run(report):
    """localhost 임의 포트를 선점하고 임시 DB의 서버만 생성/종료한다."""
    try:
        import httpx
        import numpy as np
        from websockets.legacy.client import connect
        import uvicorn  # 설치 여부만 확인하고 설치/다운로드하지 않는다.
        import fastapi
    except ImportError as exc:
        raise RuntimeError('필수 로컬 패키지 없음: ' + str(exc)) from exc
    report['versions'] = dict(fastapi=fastapi.__version__, uvicorn=uvicorn.__version__, httpx=httpx.__version__)
    before = fingerprint()
    with tempfile.TemporaryDirectory(prefix='danc3d-smoke-') as temp:
        db_path = Path(temp) / 'smoke.db'
        env = os.environ.copy()
        env.update(DANC3D_DB_PATH=str(db_path), DANC3D_CORS_ORIGINS='http://localhost:5173',
                   DANC3D_SESSION_TTL='1800', DANC3D_MAX_SESSIONS='100', PYTHONDONTWRITEBYTECODE='1')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen(128)
            base_url = 'http://127.0.0.1:{}'.format(listener.getsockname()[1])
            report.update(base_url=base_url, isolated_db=True, database_policy='temporary_database_deleted_after_run')
            with (Path(temp) / 'server.log').open('w+') as log:
                process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:app',
                                            '--fd', str(listener.fileno()), '--log-level', 'warning'],
                                           cwd=str(ROOT), env=env, stdout=log, stderr=log,
                                           pass_fds=(listener.fileno(),))
                try:
                    with httpx.Client(trust_env=False, timeout=15) as client:
                        deadline = time.monotonic() + 20
                        while True:
                            if process.poll() is not None:
                                raise RuntimeError('테스트 서버 조기 종료')
                            try:
                                ready = client.get(base_url + '/health')
                                if ready.status_code == 200:
                                    break
                            except httpx.TransportError:
                                pass
                            if time.monotonic() >= deadline:
                                raise TimeoutError('테스트 서버 시작 시간 초과')
                            time.sleep(.1)
                        asyncio.run(exercise(base_url, db_path, report, client, connect, np))
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    log.seek(0)
                    report['server_log'] = log.read()[-8000:]
    after = fingerprint()
    changed = sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
    report['protected_file_changes'] = changed
    check(not changed, '보호 파일 또는 운영 DB 변경: ' + ', '.join(changed))


def main():
    """FAIL이면 비영 종료하며 결과와 실제 응답 요약을 JSON 보고서로 남긴다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'results/api_smoke_report.json', help='검증 결과 JSON 저장 위치')
    args = parser.parse_args()
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), status='FAIL')
    try:
        run(report)
        report['status'] = 'PASS'
    except Exception as exc:
        report['error'] = '{}: {}'.format(type(exc).__name__, exc)
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in ('evaluations', 'http_apis', 'server_log')}, ensure_ascii=False, indent=2))
    print('보고서:', args.report)
    if report['status'] != 'PASS':
        print(report.get('server_log', ''))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
