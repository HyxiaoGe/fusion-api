"""普通用户的历史轨迹读取端点。"""

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_current_user, get_trajectory_query_service
from app.db.models import User
from app.schemas.response import ApiException, success
from app.services.trajectory_query_service import TrajectoryQueryService

router = APIRouter()

_NOT_FOUND_MESSAGE = "会话或轨迹不存在，或无权访问"


@router.get("/conversations/{conversation_id}/runs")
def list_conversation_runs(
    conversation_id: str,
    request: Request,
    service: TrajectoryQueryService = Depends(get_trajectory_query_service),
    current_user: User = Depends(get_current_user),
):
    result = service.list_runs(conversation_id, current_user.id)
    if result is None:
        raise ApiException.not_found(_NOT_FOUND_MESSAGE)
    return success(data=result, request_id=request.state.request_id)


@router.get("/conversations/{conversation_id}/runs/{run_id}/trajectory")
def get_run_trajectory(
    conversation_id: str,
    run_id: str,
    request: Request,
    service: TrajectoryQueryService = Depends(get_trajectory_query_service),
    current_user: User = Depends(get_current_user),
):
    result = service.get_user_snapshot(conversation_id, run_id, current_user.id)
    if result is None:
        raise ApiException.not_found(_NOT_FOUND_MESSAGE)
    return success(data=result, request_id=request.state.request_id)


@router.get("/conversations/{conversation_id}/runs/{run_id}/node-detail/tool/{node_id}")
def get_tool_node_detail(
    conversation_id: str,
    run_id: str,
    node_id: str,
    request: Request,
    service: TrajectoryQueryService = Depends(get_trajectory_query_service),
    current_user: User = Depends(get_current_user),
):
    result = service.get_user_tool_node_detail(conversation_id, run_id, node_id, current_user.id)
    if result is None:
        raise ApiException.not_found(_NOT_FOUND_MESSAGE)
    return success(data=result, request_id=request.state.request_id)
