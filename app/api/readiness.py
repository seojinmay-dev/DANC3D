"""프레임별 T-Pose 준비 판정을 제공하는 무상태 REST API."""
from typing import Any, Annotated
from uuid import UUID
from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from app.api.serialization import json_safe
from app.core.readiness import evaluate_readiness

COUNTDOWN_DURATION_SEC = 5.0


class ReadinessRoute(APIRoute):
    """이 API의 검증 오류만 정리하여 비유한 입력을 오류 JSON에 재출력하지 않는다."""
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request):
            try:
                return await original(request)
            except RequestValidationError as exc:
                errors = [{key: item[key] for key in ('loc', 'type', 'msg')} for item in exc.errors()]
                return JSONResponse(status_code=422, content=json_safe(
                    {'detail': {'code': 'invalid_readiness_request', 'errors': errors}}))
        return handler


router = APIRouter(route_class=ReadinessRoute)


class ReadinessRequest(BaseModel):
    """브라우저의 원본 Image 점 객체 33개 또는 명시적인 미검출 null을 받는다.

    주요 점의 x/y/visibility 검증은 기존 core에 맡긴다. 점 객체의 z/presence 등
    추가 필드는 판정에 사용하지 않는다. 배열 행은 World XYZ와 혼동하지 않도록
    받지 않으며 비주요 점은 null로 보낼 수 있다. attempt_id는 클라이언트의
    준비 시도 UUID이며 서버 세션/인증 식별자가 아니다.
    """
    model_config = ConfigDict(extra='forbid')
    attempt_id: UUID
    image_width: Annotated[StrictInt, Field(gt=0)]
    image_height: Annotated[StrictInt, Field(gt=0)]
    image_landmarks: Annotated[list[dict[str, Any] | None], Field(min_length=33, max_length=33)] | None


@router.post('/readiness')
def readiness(body: ReadinessRequest):
    """기존 수식을 그대로 호출하고 현재 프레임 상태 및 카운트다운 길이를 반환한다."""
    result = evaluate_readiness(body.image_landmarks, body.image_width, body.image_height)
    if any(reason.startswith('INVALID_') for reason in result['reasons']):
        raise HTTPException(status_code=422, detail={
            'code': 'invalid_image_landmarks', 'reasons': result['reasons']})
    return json_safe(dict(result, attempt_id=str(body.attempt_id),
                          status='READY_FOR_COUNTDOWN' if result['ready'] else 'TPOSE_WAITING',
                          countdown_duration_sec=COUNTDOWN_DURATION_SEC))
