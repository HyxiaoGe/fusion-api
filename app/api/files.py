from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User
from app.services.file_service import FileService
from app.core.security import get_current_user

router = APIRouter()


@router.post("/upload")
async def upload_files(
        provider: str = Form(...),
        model: str = Form(...),
        conversation_id: str = Form(...),
        files: List[UploadFile] = File(...),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """上传文件到指定对话"""
    try:
        file_service = FileService(db)
        # 验证用户是否有权访问此对话
        # (这部分逻辑在ChatService中，这里可以简化或调用)
        file_ids = await file_service.upload_files(files, current_user.id, conversation_id, provider, model)
        return {"status": "success", "file_ids": file_ids}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@router.get("/")
def get_user_files(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """获取当前用户的所有文件"""
    file_service = FileService(db)
    files = file_service.get_files_by_user(current_user.id)
    return {"files": files}


@router.get("/conversation/{conversation_id}")
def get_conversation_files(
        conversation_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """获取对话关联的所有文件"""
    file_service = FileService(db)
    # TODO: 验证用户是否有权访问此对话
    files = file_service.get_conversation_files(conversation_id)
    return {"files": files}


@router.get("/{file_id}/status")
def get_file_status(
        file_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """获取文件处理状态"""
    file_service = FileService(db)
    file = file_service.get_file_status(file_id, user_id=current_user.id)
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在或无权访问")
    return file


@router.delete("/{file_id}")
def delete_file(
        file_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """删除文件"""
    file_service = FileService(db)
    success = file_service.delete_file(file_id, user_id=current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="文件不存在或删除失败")
    return {"status": "success", "message": "文件已删除"}
