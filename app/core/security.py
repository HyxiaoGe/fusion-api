import hashlib
import logging
import re
import time
from typing import Optional

import httpx
from auth_service_client import AuthenticatedUser, JWTValidator
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.revocation import extract_session_id, is_session_access_revoked, is_user_access_revoked
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


def _sync_existing_user_fields(
    db: Session,
    user: User,
    user_repo: UserRepository,
    *,
    subject: str,
    email: str | None,
    nickname: str | None,
    avatar: str | None,
    is_superuser: bool,
) -> None:
    """同步已有用户资料；空 userinfo 不覆盖已有昵称和头像。"""
    should_commit = False
    if email and user.email != email:
        user.email = email
        should_commit = True
    if nickname and nickname != user.nickname:
        user.nickname = nickname
        should_commit = True
    if avatar and avatar != user.avatar:
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


def _is_social_account_identity_conflict(error: IntegrityError) -> bool:
    """仅识别 auth provider identity 唯一约束，绝不吞其他完整性错误。"""
    original = getattr(error, "orig", None)
    diagnostic = getattr(original, "diag", None)
    if getattr(diagnostic, "constraint_name", None) == "uix_provider_user_id":
        return True
    # SQLite 回归测试/本地工具没有 PostgreSQL diag，仅接受精确列组合。
    message = str(original or "")
    return "UNIQUE constraint failed: social_accounts.provider, social_accounts.provider_user_id" in message


def _is_user_identity_conflict(error: IntegrityError) -> bool:
    """只允许可能来自同一 subject 并发 INSERT 的用户唯一约束。"""
    original = getattr(error, "orig", None)
    diagnostic = getattr(original, "diag", None)
    constraint_name = getattr(diagnostic, "constraint_name", None)
    if constraint_name in {"users_pkey", "ix_users_email", "ix_users_username"}:
        return True
    message = str(original or "")
    return any(
        marker in message
        for marker in (
            "UNIQUE constraint failed: users.id",
            "UNIQUE constraint failed: users.email",
            "UNIQUE constraint failed: users.username",
        )
    )


def _ensure_social_account(
    db: Session,
    social_repo: SocialAccountRepository,
    *,
    user: User,
    subject: str,
) -> User:
    """并发安全地创建 auth-service 关联；冲突后回滚并读取胜者。"""
    social_repo.create(
        {
            "user_id": user.id,
            "provider": AUTH_PROVIDER,
            "provider_user_id": subject,
        }
    )
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        if not _is_social_account_identity_conflict(error):
            raise
        recovered = social_repo.get_by_provider(AUTH_PROVIDER, subject)
        if recovered is None:
            raise
        return recovered.user
    db.refresh(user)
    return user


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
        try:
            db.commit()
            db.refresh(user)
        except IntegrityError as error:
            # 两个首请求可能同时观察到用户不存在。失败事务必须先 rollback，
            # 再按同一 auth identity / subject 读取并发胜者。
            db.rollback()
            if not _is_user_identity_conflict(error):
                raise
            social_account = social_repo.get_by_provider(AUTH_PROVIDER, subject)
            recovered_user = social_account.user if social_account else user_repo.get(subject)
            # 即使命中了 email/username 唯一索引，也必须以相同 subject 为最终证明；
            # 不能仅凭 email 把本次认证错误绑定到另一个本地用户。
            if recovered_user is None or str(recovered_user.id) != subject:
                raise
            user = recovered_user
            _sync_existing_user_fields(
                db,
                user,
                user_repo,
                subject=subject,
                email=email,
                nickname=nickname,
                avatar=avatar,
                is_superuser=is_superuser,
            )
    else:
        # 仅在确凿拿到新值时才更新昵称/头像，绝不用空值覆盖既有 profile（与上面 email 的
        # `email and ...` 守卫同款）。userinfo 拉取失败（慢 cloudflared 隧道超时 → 上面 except
        # 分支 userinfo={}）或返回里缺 name/avatar_url 时，nickname/avatar 会是 None；旧逻辑无
        # 条件写入即把既有头像抹成 NULL 并 commit → /api/auth/me 返回 avatar:null → 前端头像
        # 回退单字母。此处每个鉴权请求都会跑（get_current_user），故空值覆盖会被高频触发。
        _sync_existing_user_fields(
            db,
            user,
            user_repo,
            subject=subject,
            email=email,
            nickname=nickname,
            avatar=avatar,
            is_superuser=is_superuser,
        )

    if not social_account:
        resolved_user = _ensure_social_account(
            db,
            social_repo,
            user=user,
            subject=subject,
        )
        if resolved_user is not user:
            user = resolved_user
            _sync_existing_user_fields(
                db,
                user,
                user_repo,
                subject=subject,
                email=email,
                nickname=nickname,
                avatar=avatar,
                is_superuser=is_superuser,
            )

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

    try:
        session_id = extract_session_id(auth_user.raw_payload)
    except ValueError:
        logger.warning("Auth token verification failed: invalid sid claim")
        raise credentials_exception

    # 跨应用单点登出：签名/类型校验通过后，再查共享 Redis 的吊销标记。放在 _sync_user_from_claims
    # 之前，吊销令牌即可快速失败，省去 userinfo 拉取与 DB 写。
    if is_session_access_revoked(session_id) or is_user_access_revoked(
        auth_user.sub,
        auth_user.raw_payload.get("iat"),
    ):
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
