"""세션 REST 제어와 외부 World 입력 WebSocket."""
import json
from typing import Literal, Union
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel, ConfigDict, StrictStr, StrictFloat, StrictInt
from starlette.concurrency import run_in_threadpool

from app.api.serialization import json_safe
from app.services.session_manager import ServiceError
from app.services.auth import get_current_user_id

router = APIRouter()

from pydantic import Field # 상단 import에 Field 추가

class StartRequest(BaseModel):
    song_id: StrictStr = Field(..., min_length=1)
    guest_id: StrictStr = Field(..., min_length=1)
    nickname: StrictStr = Field(..., min_length=1, max_length=80) # 닉네임 최소 1자 이상 검증
    playback_speed: Union[StrictFloat, StrictInt] = 1.0
    mode: Literal['video'] = 'video'
    model_config = ConfigDict(extra='forbid')

@router.post('/sessions/start', status_code=201)
def start(body: StartRequest, request: Request, user_id: int | None = Depends(get_current_user_id)):
    """요청 검증 후 독립된 연습 세션을 생성한다."""
    return request.app.state.manager.start(**body.model_dump())

def make_control(action):
    """각 공개 제어 경로를 같은 서비스 동작에 연결한다."""
    def control(session_id: str, request: Request):
        return json_safe(request.app.state.manager.control(session_id, action))
    return control

# [수정] 'end'를 제외한 나머지 제어만 일괄 등록
for action in ('pause', 'resume', 'reset'):
    router.add_api_route('/sessions/{session_id}/' + action, make_control(action), methods=['POST'], name=action)

# [추가] 'end' 라우트를 분리하여 user_id 매개변수 주입
@router.post('/sessions/{session_id}/end', name='end')
def end_session(session_id: str, request: Request, user_id: int | None = Depends(get_current_user_id)):
    """종료 시 토큰을 확인하여 user_id를 manager에 전달한다."""
    return json_safe(request.app.state.manager.control(session_id, 'end', user_id=user_id))

@router.websocket('/ws/sessions/{session_id}')
async def poses(websocket: WebSocket, session_id: str):
    """한 세션에 한 연결만 허용하며 연결 해제는 저장/종료하지 않는다."""
    manager = websocket.app.state.manager
    origin = websocket.headers.get('origin')
    if origin is not None and origin not in websocket.app.state.allowed_origins:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    entry = None
    claimed = False
    try:
        entry = await run_in_threadpool(manager.get, session_id)
        with entry.lock:
            if entry.session.state == 'ended':
                raise ServiceError('session_ended')
            if entry.connected:
                raise ServiceError('session_already_connected')
            entry.connected = True
            claimed = True
        while True:
            payload = await websocket.receive_text()
            try:
                message = json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonstandard number')))
                result = await run_in_threadpool(manager.update, session_id, message)
                await websocket.send_json(json_safe(result))
            except (ValueError, TypeError):
                await websocket.send_json({'type': 'error', 'code': 'invalid_json', 'status': 422})
            except ServiceError as exc:
                await websocket.send_json({'type': 'error', 'code': exc.code, 'status': exc.status})
                if exc.code == 'session_ended':
                    await websocket.close(code=1008)
                    break
    except ServiceError as exc:
        await websocket.send_json({'type': 'error', 'code': exc.code, 'status': exc.status})
        await websocket.close(code=1008)
    except WebSocketDisconnect:
        pass
    finally:
        if claimed:
            with entry.lock:
                entry.connected = False
                entry.touched = manager.clock()