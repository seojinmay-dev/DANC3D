from datetime import datetime, timedelta
import os
import secrets
from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer

# 비밀번호 암호화 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 토큰 생성용 암호 키 및 만료 시간 (24시간)
# 환경 변수 DANC3D_SECRET_KEY를 사용하고, 없으면 서버 시작마다 무작위로 생성한다.
SECRET_KEY = os.environ.get("DANC3D_SECRET_KEY") or secrets.token_urlsafe(32)
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)

def get_current_user_id(request: Request, token: str | None = Depends(oauth2_scheme)):
    """Only absent credentials mean guest; invalid credentials must require login."""
    if token is None and not request.headers.get('authorization'):
        return None
    try:
        if token is None:
            raise ValueError('invalid authorization header')
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"require_exp": True})
        subject = payload.get("sub", "")
        if not isinstance(subject, str) or not subject.isdecimal() or int(subject) < 1:
            raise ValueError('invalid subject')
        user_id = int(subject)
        with request.app.state.manager.database.connect() as db:
            exists = db.execute('SELECT id FROM auth_users WHERE id=?', (user_id,)).fetchone()
        if exists is None:
            raise ValueError('unknown user')
        return user_id
    except (JWTError, ValueError, OverflowError):
        raise HTTPException(status_code=401, detail='로그인이 만료되었거나 유효하지 않습니다.',
                            headers={'WWW-Authenticate':'Bearer'})

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=60 * 24)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
