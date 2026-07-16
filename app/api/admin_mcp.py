from fastapi import APIRouter, Depends

from app.api.deps import get_current_admin_user, get_mcp_server_service
from app.db.models import User as UserModel
from app.schemas.mcp import (
    McpServerCreate,
    McpServerResponse,
    McpServerStatusRequest,
    McpServerUpdate,
)
from app.schemas.response import success
from app.services.mcp.server_service import McpServerService

router = APIRouter()


@router.get("/servers")
async def list_mcp_servers(
    _admin: UserModel = Depends(get_current_admin_user),
    service: McpServerService = Depends(get_mcp_server_service),
):
    return success([_serialize(row) for row in service.list_servers()])


@router.post("/servers")
async def create_mcp_server(
    request: McpServerCreate,
    _admin: UserModel = Depends(get_current_admin_user),
    service: McpServerService = Depends(get_mcp_server_service),
):
    return success(_serialize(service.create_server(request)))


@router.patch("/servers/{server_id}")
async def update_mcp_server(
    server_id: str,
    request: McpServerUpdate,
    _admin: UserModel = Depends(get_current_admin_user),
    service: McpServerService = Depends(get_mcp_server_service),
):
    return success(_serialize(service.update_server(server_id, request)))


@router.post("/servers/{server_id}/status")
async def update_mcp_server_status(
    server_id: str,
    request: McpServerStatusRequest,
    _admin: UserModel = Depends(get_current_admin_user),
    service: McpServerService = Depends(get_mcp_server_service),
):
    return success(_serialize(service.set_status(server_id, request)))


@router.post("/servers/{server_id}/test")
async def test_mcp_server(
    server_id: str,
    _admin: UserModel = Depends(get_current_admin_user),
    service: McpServerService = Depends(get_mcp_server_service),
):
    return success(_serialize(await service.test_server(server_id)))


@router.post("/servers/{server_id}/tools/refresh")
async def refresh_mcp_server_tools(
    server_id: str,
    _admin: UserModel = Depends(get_current_admin_user),
    service: McpServerService = Depends(get_mcp_server_service),
):
    return success(_serialize(await service.refresh_tools(server_id)))


def _serialize(row) -> McpServerResponse:
    return McpServerResponse.model_validate(row)
