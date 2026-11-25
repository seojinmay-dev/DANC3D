"""FastAPI 진입점: 초기화, CORS, 라우터, 인증 및 health를 담당한다."""
import asyncio
from contextlib import asynccontextmanager, suppress
import os
from pathlib import Path
import sqlite3

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from starlette.concurrency import run_in_threadpool

from app.api import songs, sessions, records, readiness
from app.db.database import Database
from app.services.dance_session import DEFAULT_DATA_DIR
from app.services.session_manager import SessionManager, ServiceError

# 공통 인증 서비스
from app.services.auth import get_password_hash, verify_password, create_access_token


def create_app(data_dir=None, db_path=None):
    """테스트는 임시 DB 경로를 주입하며 운영 DB는 lifespan에서 생성한다."""
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    actual_db_path = db_path or os.environ.get('DANC3D_DB_PATH', str(data_dir / 'danc3d.db'))
    database = Database(actual_db_path)

    @asynccontextmanager
    async def lifespan(app):
        manager = SessionManager(database, data_dir, ttl=float(os.environ.get('DANC3D_SESSION_TTL', '1800')),
                                 max_sessions=int(os.environ.get('DANC3D_MAX_SESSIONS', '100')))
        database.initialize(manager.catalog)
        app.state.manager = manager

        async def cleanup_loop():
            while True:
                await asyncio.sleep(60)
                await run_in_threadpool(manager.cleanup)

        task = asyncio.create_task(cleanup_loop())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title='DANC3D', lifespan=lifespan)
    origins = [value.strip() for value in os.environ.get('DANC3D_CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173').split(',') if value.strip()]
    app.state.allowed_origins = origins
    if '*' in origins:
        raise ValueError('Explicit CORS origins required')
    
    # 🔑 [수정] 프론트엔드가 Authorization 헤더를 보낼 수 있도록 allow_headers 및 expose_headers 업데이트
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=['GET', 'POST'],
        allow_headers=['Content-Type', 'Authorization'],
        expose_headers=['X-Reference-Row', 'X-Frame-Index', 'Authorization']
    )

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        """경로나 내부 계산 세부 사항 없이 명확한 오류 코드를 반환한다."""
        return JSONResponse(status_code=exc.status, content={'detail': {'code': exc.code}})

    @app.get('/health')
    def health():
        """애플리케이션 생존 여부를 반환한다."""
        return {'status': 'ok'}

    # 🔑 [추가] 4. 회원가입 API
    @app.post("/register")
    def register_user(username: str, password: str):
        conn = sqlite3.connect(actual_db_path)
        cursor = conn.cursor()
        
        # 이미 존재하는 사용자인지 검사
        cursor.execute("SELECT id FROM auth_users WHERE username=?", (username,))
        if cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=400, detail="이미 존재하는 아이디입니다.")
            
        hashed_password = get_password_hash(password)
        cursor.execute("INSERT INTO auth_users (username, password_hash) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        conn.close()
        return {"message": "회원가입이 완료되었습니다."}

    # 🔑 [추가] 4. 로그인 API (JWT 토큰 발급)
    @app.post("/login")
    def login(form_data: OAuth2PasswordRequestForm = Depends()):
        conn = sqlite3.connect(actual_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash FROM auth_users WHERE username=?", (form_data.username,))
        user = cursor.fetchone()
        conn.close()

        if not user or not verify_password(form_data.password, user[2]):
            raise HTTPException(status_code=400, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
            
        # user[0]은 DB의 id(PK)입니다.
        access_token = create_access_token(data={"sub": str(user[0])})
        return {"access_token": access_token, "token_type": "bearer", "nickname": user[1]}

    for router in (songs.router, sessions.router, records.router, readiness.router):
        app.include_router(router)
        
    return app


app = create_app()
