from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from app.api.deps import get_user_repo
from app.core import security
from app.core.avatar_proxy import AvatarProxyError, fetch_avatar
from app.db.models import User as UserModel
from app.db.repositories import UserRepository
from app.schemas.auth import UserSettingsUpdate
from app.schemas.response import success

router = APIRouter()


@router.get("/avatar")
def proxy_avatar(url: str = Query(..., min_length=1)):
    """同源头像代理：按白名单抓取第三方头像并缓存，前端用同源 src 加载（绕开国内直连图床慢）。

    不鉴权——头像 URL 本就是公开的，且 ``<img>`` 请求不会带 Bearer；安全由 avatar_proxy 的
    https + host 白名单（防 SSRF）保证。``Cache-Control`` 让浏览器再缓存一天，后续秒开。
    """
    try:
        body, content_type = fetch_avatar(url)
    except AvatarProxyError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return Response(
        content=body,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/me")
async def read_current_user(request: Request, current_user: UserModel = Depends(security.get_current_user)):
    user_data = {
        "id": current_user.id,
        "username": current_user.username,
        "nickname": current_user.nickname,
        "avatar": current_user.avatar,
        "system_prompt": current_user.system_prompt or "",
        "is_superuser": bool(current_user.is_superuser),
    }
    return success(data=user_data, request_id=request.state.request_id)


@router.patch("/me")
async def update_current_user(
    body: UserSettingsUpdate,
    request: Request,
    current_user: UserModel = Depends(security.get_current_user),
    user_repo: UserRepository = Depends(get_user_repo),
):
    """更新当前用户的个性化设置（system_prompt）"""
    updated = user_repo.update_system_prompt(current_user, body.system_prompt)
    return success(
        data={"system_prompt": updated.system_prompt},
        request_id=request.state.request_id,
    )
