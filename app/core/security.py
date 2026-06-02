import hashlib
import logging
import re
import time
from typing import Optional

import httpx
from auth import AuthenticatedUser, JWTValidator
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.revocation import is_user_access_revoked
from app.db.database import get_db
from app.db.models import User
from app.db.repositories import SocialAccountRepository, UserRepository

logger = logging.getLogger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")
AUTH_PROVIDER = "auth_service"


# 共享 SDK auth-client 的 JWTValidator（RS256 + JWKS 缓存）替代原 hand-rolled 实现。
# require_token_type="access" 复现 fusion 原有的「拒绝非 access 令牌」校验，避免 refresh
# 令牌被用在受保护路由——该能力已 upstream 进 auth-client（默认 None 不影响其它消费方）。
# 仍叫 jwt_validator：get_current_user / files.py 沿用此名，仅返回类型由 dict 变为
# AuthenticatedUser（下方按 .sub/.email/.scopes 读取）。
jwt_validator = JWTValidator(
    jwks_url=settings.RESOLVED_AUTH_SERVICE_JWKS_URL,
    issuer=settings.AUTH_SERVICE_BASE_URL.rstrip("/"),
    audience=settings.AUTH_SERVICE_CLIENT_ID,
    require_token_type="access",
)


def _build_username_seed(email: Optional[str], subject: str) -> str:
    if email and email.strip():
        local_part = email.split("@")[0]
        slug = re.sub(r"[^a-zA-Z0-9_]+", "-", local_part).strip("-").lower()
        if slug:
            return slug
    return f"user-{subject[:8]}"


# 用户信息缓存，key 为 token 的 SHA256 哈希前缀，value 为 (userinfo, timestamp)
_userinfo_cache: dict[str, tuple[dict, float]] = {}
_USERINFO_CACHE_TTL = 300  # 5分钟，与 JWKS 缓存 TTL 保持一致


def _fetch_auth_service_userinfo(token: str) -> dict:
    token_key = hashlib.sha256(token.encode()).hexdigest()[:32]
    now = time.time()

    cached = _userinfo_cache.get(token_key)
    if cached and (now - cached[1]) < _USERINFO_CACHE_TTL:
        return cached[0]

    response = httpx.get(
        settings.AUTH_SERVICE_USERINFO_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0,
    )
    response.raise_for_status()
    userinfo = response.json()

    _userinfo_cache[token_key] = (userinfo, now)
    return userinfo


def _sync_user_from_claims(db: Session, auth_user: AuthenticatedUser, token: str) -> User:
    subject = auth_user.sub
    email = (auth_user.email or "").strip() or None
    try:
        userinfo = _fetch_auth_service_userinfo(token)
    except httpx.HTTPError as exc:
        logger.warning("Auth userinfo fetch failed: %s", exc)
        userinfo = {}

    email = (userinfo.get("email") or email or "").strip() or None
    nickname = (userinfo.get("name") or "").strip() or None
    avatar = (userinfo.get("avatar_url") or "").strip() or None

    # auth-service JWT 用 scopes=['admin'|'user']，没有 is_superuser claim
    # 这里把 'admin' scope 映射成 is_superuser=True，同步到 fusion-api 本地 users 表
    scopes = auth_user.scopes or []
    is_superuser = "admin" in scopes

    user_repo = UserRepository(db)
    social_repo = SocialAccountRepository(db)

    social_account = social_repo.get_by_provider(AUTH_PROVIDER, subject)
    user = social_account.user if social_account else None
    if not user:
        user = user_repo.get(subject)
    if not user and email:
        user = user_repo.get_by_email(email)

    if not user:
        user = user_repo.create(
            {
                "id": subject,
                "username": user_repo.build_unique_username(
                    _build_username_seed(email, subject),
                    subject,
                ),
                "email": email,
                "nickname": nickname,
                "avatar": avatar,
                "is_superuser": is_superuser,
            }
        )
        db.commit()
        db.refresh(user)
    else:
        should_commit = False
        if email and user.email != email:
            user.email = email
            should_commit = True
        if nickname != user.nickname:
            user.nickname = nickname
            should_commit = True
        if avatar != user.avatar:
            user.avatar = avatar
            should_commit = True
        if not user.username:
            user.username = user_repo.build_unique_username(
                _build_username_seed(email, subject),
                subject,
            )
            should_commit = True
        if user.is_superuser != is_superuser:
            user.is_superuser = is_superuser
            should_commit = True
        if should_commit:
            db.commit()
            db.refresh(user)

    if not social_account:
        social_repo.create(
            {
                "user_id": user.id,
                "provider": AUTH_PROVIDER,
                "provider_user_id": subject,
            }
        )
        db.commit()
        db.refresh(user)

    return user


def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        auth_user = jwt_validator.verify(token)
    except (InvalidTokenError, httpx.HTTPError) as exc:
        logger.warning("Auth token verification failed: %s", exc)
        raise credentials_exception

    # 跨应用单点登出：签名/类型校验通过后，再查共享 Redis 的吊销标记。放在 _sync_user_from_claims
    # 之前，吊销令牌即可快速失败，省去 userinfo 拉取与 DB 写。
    if is_user_access_revoked(auth_user.sub, auth_user.raw_payload.get("iat")):
        logger.info("Access token revoked via SLO for user %s", auth_user.sub)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = _sync_user_from_claims(db, auth_user, token)
    if user is None:
        raise credentials_exception
    return user
