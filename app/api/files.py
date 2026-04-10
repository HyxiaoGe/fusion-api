import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.core.file_token import verify_file_token
from app.core.security import get_current_user, jwt_validator
from app.db.database import get_db
from app.db.models import User
from app.db.repositories import UserRepository
from app.services.file_service import FileService

logger = logging.getLogger(__name__)
router = APIRouter()


def _resolve_user_from_bearer(request: Request, db: Session) -> Optional[User]:
    """
    从 Authorization header 中提取 Bearer token 并验证用户。
    成功返回 User，失败返回 None（不抛异常）。
    """
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header[len("Bearer ") :]
    try:
        payload = jwt_validator.verify(token)
        subject = payload.get("sub")
        if not subject:
            return None
        user_repo = UserRepository(db)
        return user_repo.get(subject)
    except Exception:
        return None


@router.post("/upload")
async def upload_files(
    provider: str = Form(...),
    model: str = Form(...),
    conversation_id: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """上传文件到指定对话"""
    try:
        file_service = FileService(db)
        results = await file_service.upload_files(files, current_user.id, conversation_id, provider, model)
        # 兼容旧前端：返回 file_ids 列表 + 新增 files 详情
        file_ids = [r["file_id"] for r in results]
        return {"status": "success", "file_ids": file_ids, "files": results}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@router.get("/{file_id}/url")
async def get_file_url(
    file_id: str,
    variant: str = Query("thumbnail", pattern="^(processed|thumbnail)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取文件访问 URL（presigned URL 或 API 代理路径）"""
    file_service = FileService(db)
    url = await file_service.get_file_url(file_id, current_user.id, variant)
    if not url:
        raise HTTPException(status_code=404, detail="文件不存在或无权访问")
    return {"url": url}


@router.get("/{file_id}/content")
async def get_file_content(
    file_id: str,
    variant: str = Query("thumbnail", pattern="^(processed|thumbnail)$"),
    token: Optional[str] = Query(None),
    request: Request = None,
    db: Session = Depends(get_db),
):
    """
    直接返回文件内容（用于本地存储模式的代理访问）。

    认证方式（二选一）：
    - Bearer token（Authorization header）
    - 签名 token（?token=xxx query 参数，本地存储签名 URL 使用）
    """
    user_id: Optional[str] = None

    # 优先尝试 Bearer token 认证
    current_user = _resolve_user_from_bearer(request, db)
    if current_user:
        user_id = current_user.id
    elif token:
        # 签名 token 认证：验证签名和过期时间，匹配 file_id
        verified_file_id = verify_file_token(token)
        if not verified_file_id or verified_file_id != file_id:
            raise HTTPException(status_code=401, detail="无效或过期的文件访问令牌")
        # 签名 token 已验证 file_id，跳过 user_id 过滤（token 本身即授权凭证）
        user_id = None
    else:
        raise HTTPException(status_code=401, detail="需要认证才能访问文件")

    file_service = FileService(db)
    result = await file_service.get_file_content(file_id, user_id, variant)
    if not result:
        raise HTTPException(status_code=404, detail="文件不存在或无权访问")

    data, mime_type = result
    return Response(
        content=data,
        media_type=mime_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/")
def get_user_files(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """获取当前用户的所有文件"""
    file_service = FileService(db)
    files = file_service.get_files_by_user(current_user.id)
    return {"files": files}


@router.get("/conversation/{conversation_id}")
def get_conversation_files(
    conversation_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """获取对话关联的所有文件"""
    file_service = FileService(db)
    files = file_service.get_conversation_files_for_user(conversation_id, current_user.id)
    if files is None:
        raise HTTPException(status_code=404, detail="对话不存在或无权访问")
    return {"files": files}


@router.get("/{file_id}/status")
def get_file_status(file_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """获取文件处理状态"""
    file_service = FileService(db)
    file = file_service.get_file_status(file_id, user_id=current_user.id)
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在或无权访问")
    return file


@router.delete("/{file_id}")
async def delete_file(file_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """删除文件"""
    file_service = FileService(db)
    success = await file_service.delete_file(file_id, user_id=current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="文件不存在或删除失败")
    return {"status": "success", "message": "文件已删除"}
