import hashlib
import logging
import re
import time
from typing import Dict, Optional

import httpx
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError, PyJWK
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db
from app.db.models import User
from app.db.repositories import SocialAccountRepository, UserRepository

logger = logging.getLogger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")
AUTH_PROVIDER = "auth_service"


class AuthServiceJWTValidator:
    def __init__(
        self,
        jwks_url: str,
        issuer: str,
        audience: Optional[str] = None,
        cache_ttl: int = 300,
    ):
        self.jwks_url = jwks_url
        self.issuer = issuer
        self.audience = audience
        self.cache_ttl = cache_ttl
        self._jwks_cache: Optional[Dict] = None
        self._cache_time = 0.0

    def _fetch_jwks(self) -> dict:
        now = time.time()
        if self._jwks_cache and (now - self._cache_time) < self.cache_ttl:
            return self._jwks_cache

        response = httpx.get(self.jwks_url, timeout=10.0)
        response.raise_for_status()
        self._jwks_cache = response.json()
        self._cache_time = now
        return self._jwks_cache

    def _get_signing_key(self, token: str):
        jwks = self._fetch_jwks()
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")

        for key_data in jwks.get("keys", []):
            if key_data.get("kid") == kid:
                return PyJWK(key_data).key

        raise InvalidTokenError(f"No matching key found for kid: {kid}")

    def verify(self, token: str) -> dict:
        signing_key = self._get_signing_key(token)
        options = {"verify_aud": bool(self.audience)}
        decode_kwargs = {
            "algorithms": ["RS256"],
            "issuer": self.issuer,
            "options": options,
        }
        if self.audience:
            decode_kwargs["audience"] = self.audience

        payload = jwt.decode(token, signing_key, **decode_kwargs)
        if payload.get("type") != "access":
            raise InvalidTokenError("Invalid access token type")
        return payload


jwt_validator = AuthServiceJWTValidator(
    jwks_url=settings.RESOLVED_AUTH_SERVICE_JWKS_URL,
    issuer=settings.AUTH_SERVICE_BASE_URL.rstrip("/"),
    audience=settings.AUTH_SERVICE_CLIENT_ID,
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


def _sync_user_from_claims(db: Session, payload: dict, token: str) -> User:
    subject = payload["sub"]
    email = (payload.get("email") or "").strip() or None
    try:
        userinfo = _fetch_auth_service_userinfo(token)
    except httpx.HTTPError as exc:
        logger.warning("Auth userinfo fetch failed: %s", exc)
        userinfo = {}

    email = (userinfo.get("email") or email or "").strip() or None
    nickname = (userinfo.get("name") or "").strip() or None
    avatar = (userinfo.get("avatar_url") or "").strip() or None

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
        payload = jwt_validator.verify(token)
    except (InvalidTokenError, httpx.HTTPError) as exc:
        logger.warning("Auth token verification failed: %s", exc)
        raise credentials_exception

    user = _sync_user_from_claims(db, payload, token)
    if user is None:
        raise credentials_exception
    return user
