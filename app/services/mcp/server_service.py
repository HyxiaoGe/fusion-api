from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.exc import IntegrityError

from app.db.mcp_server_repository import McpServerRepository
from app.schemas.mcp import McpServerCreate, McpServerStatusRequest, McpServerUpdate
from app.schemas.response import ApiException
from app.services.mcp.client import McpClientError, McpClientManager, McpConnectionConfig
from app.utils.time import utc_now

_CONNECTION_IDENTITY_FIELDS = {
    "provider",
    "endpoint_url",
    "transport",
    "auth_type",
    "auth_name",
    "credential_ref",
}


class McpServerService:
    def __init__(
        self,
        repository: McpServerRepository,
        client_manager: McpClientManager,
        *,
        clock: Callable = utc_now,
    ):
        self.repository = repository
        self.client_manager = client_manager
        self.clock = clock

    def list_servers(self):
        return self.repository.list_all()

    def create_server(self, request: McpServerCreate):
        self._ensure_unique_name(request.name)
        if request.allowed_tools:
            raise ApiException.bad_request("allowed_tools 只能选择已发现工具")
        values = request.model_dump()
        values.update(
            {
                "is_enabled": False,
                "discovered_tools": [],
                "health_status": "disabled",
                "last_checked_at": None,
                "last_error_code": None,
                "last_error_message": None,
            }
        )
        self._validate_client_configuration(values, server_id="pending")
        try:
            return self.repository.create(values)
        except IntegrityError:
            raise ApiException.conflict("MCP 服务名称已存在") from None

    def update_server(self, server_id: str, request: McpServerUpdate):
        row = self._require_server(server_id)
        changes = request.model_dump(exclude_unset=True)
        if "name" in changes and changes["name"] != row.name:
            self._ensure_unique_name(changes["name"], exclude_id=row.id)

        candidate = self._build_candidate(row, changes)
        try:
            validated = McpServerCreate(**candidate)
        except ValueError as exc:
            raise ApiException.bad_request("MCP 服务配置无效") from exc
        candidate = validated.model_dump()
        self._validate_client_configuration(candidate, server_id=row.id)

        identity_changed = any(
            field in changes and changes[field] != getattr(row, field) for field in _CONNECTION_IDENTITY_FIELDS
        )
        if identity_changed:
            changes.update(self._reset_connection_state(row.is_enabled))
        else:
            allowed_tools = changes.get("allowed_tools", row.allowed_tools or [])
            self._ensure_allowed_subset(allowed_tools, row.discovered_tools or [])
        try:
            return self.repository.update(row, changes)
        except IntegrityError:
            raise ApiException.conflict("MCP 服务名称已存在") from None

    def set_status(self, server_id: str, request: McpServerStatusRequest):
        row = self._require_server(server_id)
        values: dict[str, Any] = {
            "is_enabled": request.is_enabled,
            "health_status": "unknown" if request.is_enabled else "disabled",
            "last_error_code": None,
            "last_error_message": None,
        }
        if request.is_enabled:
            values["last_checked_at"] = None
        return self.repository.update(row, values)

    async def test_server(self, server_id: str):
        row = self._require_server(server_id)
        expected_version = row.config_version
        config = self._to_connection_config(row)
        try:
            await self.client_manager.test_connection(config)
        except McpClientError as exc:
            return self._save_remote_failure(row.id, expected_version, exc)
        return self._save_remote_result(
            row.id,
            expected_version,
            {
                "health_status": "healthy",
                "last_checked_at": self.clock(),
                "last_error_code": None,
                "last_error_message": None,
            },
        )

    async def refresh_tools(self, server_id: str):
        row = self._require_server(server_id)
        expected_version = row.config_version
        config = self._to_connection_config(row)
        try:
            tools = await self.client_manager.list_tools(config)
        except McpClientError as exc:
            return self._save_remote_failure(row.id, expected_version, exc)

        discovered_names = {tool["name"] for tool in tools}
        allowed_tools = [name for name in (row.allowed_tools or []) if name in discovered_names]
        return self._save_remote_result(
            row.id,
            expected_version,
            {
                "discovered_tools": tools,
                "allowed_tools": allowed_tools,
                "health_status": "healthy",
                "last_checked_at": self.clock(),
                "last_error_code": None,
                "last_error_message": None,
            },
        )

    def _save_remote_failure(self, server_id: str, expected_version: int, error: McpClientError):
        return self._save_remote_result(
            server_id,
            expected_version,
            {
                "health_status": "unhealthy",
                "last_checked_at": self.clock(),
                "last_error_code": error.code,
                "last_error_message": error.safe_message,
            },
        )

    def _save_remote_result(self, server_id: str, expected_version: int, values: dict[str, Any]):
        updated = self.repository.update_if_version(server_id, expected_version, values)
        if updated is not None:
            return updated
        current = self.repository.get(server_id)
        if current is None:
            raise ApiException.not_found("MCP 服务不存在")
        return current

    def _validate_client_configuration(self, values: dict[str, Any], *, server_id: str) -> None:
        config = McpConnectionConfig(
            server_id=server_id,
            provider=values["provider"],
            endpoint_url=values["endpoint_url"],
            auth_type=values["auth_type"],
            auth_name=values.get("auth_name"),
            credential_ref=values.get("credential_ref"),
            allowed_tools=values.get("allowed_tools", []),
        )
        try:
            self.client_manager.validate_configuration(config)
        except McpClientError as exc:
            raise ApiException.bad_request(exc.safe_message) from None

    def _to_connection_config(self, row) -> McpConnectionConfig:
        return McpConnectionConfig(
            server_id=row.id,
            provider=row.provider,
            endpoint_url=row.endpoint_url,
            auth_type=row.auth_type,
            auth_name=row.auth_name,
            credential_ref=row.credential_ref,
            allowed_tools=list(row.allowed_tools or []),
        )

    def _require_server(self, server_id: str):
        row = self.repository.get(server_id)
        if row is None:
            raise ApiException.not_found("MCP 服务不存在")
        return row

    def _ensure_unique_name(self, name: str, *, exclude_id: str | None = None) -> None:
        existing = self.repository.get_by_name(name)
        if existing is not None and existing.id != exclude_id:
            raise ApiException.conflict("MCP 服务名称已存在")

    @staticmethod
    def _ensure_allowed_subset(allowed_tools: list[str], discovered_tools: list[dict[str, Any]]) -> None:
        discovered_names = {tool.get("name") for tool in discovered_tools}
        if any(tool not in discovered_names for tool in allowed_tools):
            raise ApiException.bad_request("allowed_tools 只能选择已发现工具")

    @staticmethod
    def _build_candidate(row, changes: dict[str, Any]) -> dict[str, Any]:
        fields = McpServerCreate.model_fields
        return {field: changes.get(field, getattr(row, field)) for field in fields}

    @staticmethod
    def _reset_connection_state(is_enabled: bool) -> dict[str, Any]:
        return {
            "allowed_tools": [],
            "discovered_tools": [],
            "health_status": "unknown" if is_enabled else "disabled",
            "last_checked_at": None,
            "last_error_code": None,
            "last_error_message": None,
        }
