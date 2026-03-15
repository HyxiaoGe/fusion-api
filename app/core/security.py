import logging
import re
import time

import httpx
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jwt import InvalidTokenError, PyJWK

from app.core.config import settings
from app.db.database import get_db
from app.db.repositories import SocialAccountRepository, UserRepository
from app.db.models import User

logger = logging.getLogger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")
AUTH_PROVIDER = "auth_service"


class AuthServiceJWTValidator:
    def __init__(
        self,
        jwks_url: str,
        issuer: str,
        audience: str | None = None,
        cache_ttl: int = 300,
    ):
        self.jwks_url = jwks_url
        self.issuer = issuer
        self.audience = audience
        self.cache_ttl = cache_ttl
        self._jwks_cache: dict | None = None
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


def _build_username_seed(email: str | None, subject: str) -> str:
    if email and email.strip():
        local_part = email.split("@")[0]
        slug = re.sub(r"[^a-zA-Z0-9_]+", "-", local_part).strip("-").lower()
        if slug:
            return slug
    return f"user-{subject[:8]}"


def _sync_user_from_claims(db: Session, payload: dict) -> User:
    subject = payload["sub"]
    email = (payload.get("email") or "").strip() or None

    user_repo = UserRepository(db)
    social_repo = SocialAccountRepository(db)

    social_account = social_repo.get_by_provider(AUTH_PROVIDER, subject)
    if social_account:
        return social_account.user

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
            }
        )
        db.commit()
        db.refresh(user)
    else:
        should_commit = False
        if email and user.email != email:
            user.email = email
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

    if not social_repo.get_by_provider(AUTH_PROVIDER, subject):
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

    user = _sync_user_from_claims(db, payload)
    if user is None:
        raise credentials_exception
    return user
