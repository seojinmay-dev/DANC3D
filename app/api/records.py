"""guest_id 또는 회원 user_id를 기준으로 분리된 연습 기록 조회."""
from fastapi import APIRouter, Request, Depends

# 공통 인증 의존성
from app.services.auth import get_current_user_id

router = APIRouter()

# 토큰 추출용 OAuth2 스킴 (토큰이 없어도 에러를 내지 않도록 auto_error=False 설정)
@router.get('/records/{guest_id}')
def records(guest_id: str, request: Request, user_id: int | None = Depends(get_current_user_id)):
    """전체 연습을 최근 종료일 순으로 반환한다."""
    # user_id가 존재하면 회원의 기록을, 없으면 guest_id의 기록을 조회하도록 파라미터 전달
    return request.app.state.manager.database.records(guest_id=guest_id, user_id=user_id)


@router.get('/records/{guest_id}/songs/{song_id}')
def song_records(guest_id: str, song_id: str, request: Request, user_id: int | None = Depends(get_current_user_id)):
    """특정 곡 기록과 유효 점수만의 최고 점수를 반환한다."""
    manager = request.app.state.manager
    song = manager.song(song_id)
    
    # user_id 파라미터 추가 전달
    rows = manager.database.records(guest_id=guest_id, song_id=song_id, user_id=user_id)
    valid = [row['score_100'] for row in rows if row['score_100'] is not None]
    
    return dict(
        song_id=song_id, 
        title=song['title'], 
        records=rows, 
        best_score_100=max(valid) if valid else None
    )