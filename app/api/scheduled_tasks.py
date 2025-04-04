from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Body, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories import ScheduledTaskRepository
from app.services.hot_topic_service import HotTopicService
from app.schemas.scheduled_tasks import TaskResponse, TaskUpdateRequest

router = APIRouter()

@router.get("/", response_model=List[TaskResponse])
def get_all_tasks(
    db: Session = Depends(get_db)
):
    """获取所有定时任务"""
    repo = ScheduledTaskRepository(db)
    return repo.get_all_active_tasks()

@router.get("/{task_name}", response_model=TaskResponse)
def get_task(
    task_name: str,
    db: Session = Depends(get_db)
):
    """获取特定定时任务详情"""
    repo = ScheduledTaskRepository(db)
    task = repo.get_task_by_name(task_name)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task

@router.post("/{task_name}/run")
async def run_task(
    task_name: str,
    force: bool = Query(False, description="是否强制执行，忽略时间间隔限制"),
    db: Session = Depends(get_db)
):
    """手动执行特定任务"""
    if task_name == HotTopicService.TASK_NAME:
        service = HotTopicService(db)
        await service.update_hot_topics(force=force)
        return {"status": "success", "message": f"任务 {task_name} 执行完成"}
    else:
        raise HTTPException(status_code=404, detail="未知的任务类型")

@router.patch("/{task_name}")
def update_task(
    task_name: str,
    update_data: TaskUpdateRequest,
    db: Session = Depends(get_db)
):
    """更新定时任务配置"""
    repo = ScheduledTaskRepository(db)
    task = repo.get_task_by_name(task_name)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
        
    # 转换为字典并移除空值
    update_dict = update_data.dict(exclude_unset=True)
    
    success = repo.update_task(task_name, update_dict)
    if not success:
        raise HTTPException(status_code=500, detail="更新任务失败")
        
    return {"status": "success", "message": f"任务 {task_name} 已更新"}