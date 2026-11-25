"""곡 메타데이터 및 F-view JPG 라우터."""
import math
from fastapi import APIRouter, Request, Query
from fastapi.responses import FileResponse
from app.services.session_manager import ServiceError
router = APIRouter()
from app.services.dance_session import DanceSession
import numpy as np

@router.get('/songs')
def songs(request: Request):
    """확인된 10곡의 공개 정보를 반환한다."""
    manager = request.app.state.manager
    return [manager.public_song(song) for song in manager.catalog]


@router.get('/songs/{song_id}')
def song(song_id: str, request: Request):
    """곡 ID로 공개 정보를 조회한다."""
    manager = request.app.state.manager
    return manager.public_song(manager.song(song_id))


@router.get('/songs/{song_id}/reference-frame')
def reference_frame(song_id: str, request: Request, time_sec: float = Query(..., ge=0)):
    """가장 가까운 Reference row의 원본 JPG를 전송한다."""
    if not math.isfinite(time_sec):
        raise ServiceError('invalid_playback_time', 422)
    row, frame, path = request.app.state.manager.reference_frame(song_id, time_sec)
    return FileResponse(path, media_type='image/jpeg', headers={'X-Reference-Row': str(row), 'X-Frame-Index': str(frame)})

@router.get('/songs/{song_id}/reference-pose')
def reference_pose(song_id: str, request: Request):
    """3D 캐릭터 리타겟팅을 위한 기준 안무 3D 좌표 반환"""
    manager = request.app.state.manager
    
    # 곡 존재 여부 확인
    song = manager.song(song_id)
    
    # .npz 파일에서 3D 랜드마크(raw_world) 데이터 로드
    reference = DanceSession(song_id, manager.data_dir).reference
    raw_world = reference.raw_world
    
    # JSON 직렬화를 위해 numpy 배열의 NaN(결측치)을 None으로 치환 후 리스트로 변환
    safe_world = np.where(np.isnan(raw_world), None, raw_world).tolist()
    
    return safe_world