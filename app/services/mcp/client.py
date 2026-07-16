from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)

# SDK 传输层会记录原始 session id、URL 和异常；Fusion 只保留下方固定字段的脱敏审计日志。
_sdk_transport_logger = logging.getLogger("mcp.client.streamable_http")
_sdk_transport_logger.handlers.clear()
_sdk_transport_logger.addHandler(logging.NullHandler())
_sdk_transport_logger.propagate = False
_sdk_transport_logger.disabled = True

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.:/-]+$")
_AUTH_NAME_PATTERN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_FORBIDDEN_AUTH_HEADERS = {
    "accept",
    "connection",
    "content-length",
    "content-type",
    "host",
    "mcp-protocol-version",
    "mcp-session-id",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


@dataclass(frozen=True)
class McpClientPolicy:
    allowed_hosts: frozenset[str]
    allowed_credential_refs: frozenset[str]
    connect_timeout_seconds: float = 5.0
    call_timeout_seconds: float = 15.0
    max_discovery_pages: int = 5
    max_discovered_tools: int = 50
    max_tool_description_chars: int = 2_000
    max_tool_schema_bytes: int = 32_768
    max_response_bytes: int = 262_144


@dataclass(frozen=True)
class McpConnectionConfig:
    server_id: str
    provider: str
    endpoint_url: str
    auth_type: str
    auth_name: str | None
    credential_ref: str | None
    allowed_tools: list[str]


@dataclass(frozen=True)
class _ResolvedConnection:
    endpoint_url: str
    headers: dict[str, str]
    query_params: dict[str, str]


class McpClientError(Exception):
    """只携带可持久化、可返回的固定脱敏错误。"""

    def __init__(self, code: str, safe_message: str):
        self.code = code
        self.safe_message = safe_message
        super().__init__(safe_message)


class McpSession(Protocol):
    async def initialize(self) -> Any: ...

    async def list_tools(self, cursor: str | None = None) -> Any: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        read_timeout_seconds: timedelta | None = None,
    ) -> Any: ...


class McpConnector(Protocol):
    @asynccontextmanager
    async def connect(
        self,
        *,
        endpoint_url: str,
        headers: dict[str, str],
        query_params: dict[str, str],
        connect_timeout_seconds: float,
        call_timeout_seconds: float,
    ) -> AsyncIterator[McpSession]: ...

    async def close(self) -> None: ...


class _QueryParameterTransport(httpx.AsyncBaseTransport):
    """在 HTTPX 日志层之后注入 query 凭证，避免最终 URL 出现在日志和异常中。"""

    def __init__(self, query_params: dict[str, str]):
        self._query_params = query_params
        self._transport = httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        credential_url = request.url.copy_merge_params(self._query_params)
        credential_request = httpx.Request(
            request.method,
            credential_url,
            headers=request.headers,
            stream=request.stream,
            extensions=request.extensions,
        )
        return await self._transport.handle_async_request(credential_request)

    async def aclose(self) -> None:
        await self._transport.aclose()


class StreamableHttpMcpConnector:
    """官方 MCP Python SDK 的安全 Streamable HTTP 连接器。"""

    @asynccontextmanager
    async def connect(
        self,
        *,
        endpoint_url: str,
        headers: dict[str, str],
        query_params: dict[str, str],
        connect_timeout_seconds: float,
        call_timeout_seconds: float,
    ) -> AsyncIterator[McpSession]:
        transport = _QueryParameterTransport(query_params) if query_params else httpx.AsyncHTTPTransport(retries=0)
        timeout = httpx.Timeout(call_timeout_seconds, connect=connect_timeout_seconds)
        async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        ) as http_client:
            async with streamable_http_client(
                endpoint_url,
                http_client=http_client,
                terminate_on_close=True,
            ) as (read_stream, write_stream, _get_session_id):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=call_timeout_seconds),
                ) as session:
                    yield session

    async def close(self) -> None:
        """连接按操作关闭；保留统一生命周期接口供应用退出时调用。"""


class McpClientManager:
    def __init__(
        self,
        *,
        policy: McpClientPolicy,
        connector: McpConnector | None = None,
        environ: Mapping[str, str] | None = None,
    ):
        self.policy = policy
        self.connector = connector or StreamableHttpMcpConnector()
        self.environ = environ

    def validate_configuration(self, config: McpConnectionConfig) -> None:
        """只校验静态安全边界，不要求部署环境此刻已有凭证值。"""

        _validate_endpoint(config.endpoint_url, self.policy.allowed_hosts)
        auth_type = config.auth_type
        if auth_type == "none":
            if config.auth_name or config.credential_ref:
                raise McpClientError("invalid_auth", "MCP 鉴权配置无效")
            return
        if auth_type not in {"bearer", "header", "query"}:
            raise McpClientError("invalid_auth", "MCP 鉴权配置无效")
        if not config.credential_ref or config.credential_ref not in self.policy.allowed_credential_refs:
            raise McpClientError("credential_not_allowed", "MCP 凭证引用未获授权")
        if auth_type == "bearer":
            if config.auth_name:
                raise McpClientError("invalid_auth", "MCP 鉴权配置无效")
            return
        auth_name = config.auth_name or ""
        if not _AUTH_NAME_PATTERN.fullmatch(auth_name):
            raise McpClientError("invalid_auth", "MCP 鉴权配置无效")
        if auth_type == "header" and auth_name.lower() in _FORBIDDEN_AUTH_HEADERS:
            raise McpClientError("invalid_auth", "MCP 鉴权配置无效")

    async def test_connection(self, config: McpConnectionConfig) -> None:
        async def operation(_session: McpSession) -> None:
            return None

        await self._run(config, "initialize", operation)

    async def list_tools(self, config: McpConnectionConfig) -> list[dict[str, Any]]:
        async def operation(session: McpSession) -> list[dict[str, Any]]:
            return await self._list_tools(session)

        return await self._run(config, "tools_list", operation)

    async def call_tool(
        self,
        config: McpConnectionConfig,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if not _TOOL_NAME_PATTERN.fullmatch(tool_name) or tool_name not in config.allowed_tools:
            raise McpClientError("tool_not_allowed", "MCP 工具未获授权")
        normalized_arguments = _normalize_json_payload(arguments)
        if (
            not isinstance(normalized_arguments, dict)
            or len(_json_bytes(normalized_arguments)) > self.policy.max_response_bytes
        ):
            raise McpClientError("invalid_arguments", "MCP 工具参数无效")

        async def operation(session: McpSession) -> dict[str, Any]:
            async with asyncio.timeout(self.policy.call_timeout_seconds):
                result = await session.call_tool(
                    tool_name,
                    normalized_arguments,
                    read_timeout_seconds=timedelta(seconds=self.policy.call_timeout_seconds),
                )
            payload = _normalize_json_payload(result.model_dump(by_alias=True, mode="json", exclude_none=True))
            if not isinstance(payload, dict) or len(_json_bytes(payload)) > self.policy.max_response_bytes:
                raise McpClientError("invalid_response", "MCP 服务返回无效响应")
            if payload.get("isError") is True:
                raise McpClientError("tool_error", "MCP 工具执行失败")
            return payload

        return await self._run(config, "tools_call", operation)

    async def close(self) -> None:
        await self.connector.close()

    async def _run(self, config: McpConnectionConfig, operation_name: str, operation):
        started_at = time.perf_counter()
        try:
            connection = self._resolve_connection(config)
            async with self.connector.connect(
                endpoint_url=connection.endpoint_url,
                headers=connection.headers,
                query_params=connection.query_params,
                connect_timeout_seconds=self.policy.connect_timeout_seconds,
                call_timeout_seconds=self.policy.call_timeout_seconds,
            ) as session:
                async with asyncio.timeout(self.policy.connect_timeout_seconds):
                    await session.initialize()
                result = await operation(session)
                logger.info(
                    "MCP 操作完成 server_id=%s provider=%s operation=%s duration_ms=%s",
                    config.server_id,
                    config.provider,
                    operation_name,
                    int((time.perf_counter() - started_at) * 1_000),
                )
                return result
        except McpClientError as exc:
            error = exc
        except Exception as exc:
            error = _classify_exception(exc, operation_name)

        logger.warning(
            "MCP 操作失败 server_id=%s provider=%s operation=%s error_code=%s duration_ms=%s",
            config.server_id,
            config.provider,
            operation_name,
            error.code,
            int((time.perf_counter() - started_at) * 1_000),
        )
        raise error from None

    async def _list_tools(self, session: McpSession) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        seen_cursors: set[str] = set()
        cursor: str | None = None

        for _page in range(self.policy.max_discovery_pages):
            async with asyncio.timeout(self.policy.call_timeout_seconds):
                result = await session.list_tools(cursor=cursor)
            page_tools = getattr(result, "tools", None)
            if not isinstance(page_tools, list):
                raise McpClientError("invalid_response", "MCP 服务返回无效响应")
            for tool in page_tools:
                normalized = self._normalize_tool(tool)
                if normalized["name"] in seen_names:
                    raise McpClientError("invalid_response", "MCP 服务返回无效响应")
                seen_names.add(normalized["name"])
                tools.append(normalized)
                if len(tools) > self.policy.max_discovered_tools:
                    raise McpClientError("invalid_response", "MCP 服务返回无效响应")

            cursor = getattr(result, "nextCursor", None)
            if cursor is None:
                if len(_json_bytes(tools)) > self.policy.max_response_bytes:
                    raise McpClientError("invalid_response", "MCP 服务返回无效响应")
                return tools
            if not isinstance(cursor, str) or not cursor or len(cursor) > 1_024 or cursor in seen_cursors:
                raise McpClientError("invalid_response", "MCP 服务返回无效响应")
            seen_cursors.add(cursor)

        raise McpClientError("invalid_response", "MCP 服务返回无效响应")

    def _normalize_tool(self, tool: Any) -> dict[str, Any]:
        name = getattr(tool, "name", None)
        description = getattr(tool, "description", None)
        input_schema = getattr(tool, "inputSchema", None)
        if not isinstance(name, str) or len(name) > 128 or not _TOOL_NAME_PATTERN.fullmatch(name):
            raise McpClientError("invalid_response", "MCP 服务返回无效响应")
        if description is not None and not isinstance(description, str):
            raise McpClientError("invalid_response", "MCP 服务返回无效响应")
        if not isinstance(input_schema, dict):
            raise McpClientError("invalid_response", "MCP 服务返回无效响应")

        normalized_schema = _normalize_json_payload(input_schema)
        if len(_json_bytes(normalized_schema)) > self.policy.max_tool_schema_bytes:
            raise McpClientError("invalid_response", "MCP 服务返回无效响应")
        normalized_description = None
        if description is not None:
            normalized_description = _strip_unsafe_controls(description.strip())[
                : self.policy.max_tool_description_chars
            ]
        return {
            "name": name,
            "description": normalized_description or None,
            "input_schema": normalized_schema,
        }

    def _resolve_connection(self, config: McpConnectionConfig) -> _ResolvedConnection:
        self.validate_configuration(config)
        endpoint_url = config.endpoint_url
        auth_type = config.auth_type
        if auth_type == "none":
            return _ResolvedConnection(endpoint_url, {}, {})

        credential = self._resolve_credential(config.credential_ref)
        if auth_type == "bearer":
            return _ResolvedConnection(endpoint_url, {"Authorization": f"Bearer {credential}"}, {})

        auth_name = config.auth_name or ""
        if auth_type == "header":
            return _ResolvedConnection(endpoint_url, {auth_name: credential}, {})
        return _ResolvedConnection(endpoint_url, {}, {auth_name: credential})

    def _resolve_credential(self, credential_ref: str | None) -> str:
        if not credential_ref or credential_ref not in self.policy.allowed_credential_refs:
            raise McpClientError("credential_not_allowed", "MCP 凭证引用未获授权")
        source = self.environ if self.environ is not None else __import__("os").environ
        credential = source.get(credential_ref, "").strip()
        if not credential:
            raise McpClientError("credential_unavailable", "MCP 凭证不可用")
        return credential


def _validate_endpoint(endpoint_url: str, allowed_hosts: frozenset[str]) -> str:
    try:
        parsed = urlsplit(endpoint_url)
        port = parsed.port
    except ValueError:
        raise McpClientError("invalid_endpoint", "MCP 服务地址无效") from None
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port not in {None, 443}
    ):
        raise McpClientError("invalid_endpoint", "MCP 服务地址无效")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise McpClientError("invalid_endpoint", "MCP 服务地址无效")
    if not _host_is_allowed(hostname, allowed_hosts):
        raise McpClientError("endpoint_not_allowed", "MCP 服务地址未获授权")
    return endpoint_url


def _host_is_allowed(hostname: str, allowed_hosts: frozenset[str]) -> bool:
    for pattern in allowed_hosts:
        normalized = pattern.strip().lower().rstrip(".")
        if normalized.startswith("*."):
            suffix = normalized[1:]
            if hostname.endswith(suffix) and hostname != suffix[1:]:
                return True
        elif hostname == normalized:
            return True
    return False


def _classify_exception(exc: Exception, operation_name: str) -> McpClientError:
    errors = list(_iter_exceptions(exc))
    for error in errors:
        if isinstance(error, McpClientError):
            return error
    for error in errors:
        if isinstance(error, httpx.HTTPStatusError):
            status_code = error.response.status_code
            if status_code in {401, 403}:
                return McpClientError("auth_failed", "MCP 服务鉴权失败")
            if status_code in {301, 302, 303, 307, 308}:
                return McpClientError("redirect_blocked", "MCP 服务重定向已被安全策略阻止")
            if status_code == 404:
                return McpClientError("endpoint_not_found", "MCP 服务端点不存在")
            return McpClientError("upstream_error", "MCP 服务调用失败")
    if any(isinstance(error, (TimeoutError, httpx.TimeoutException)) for error in errors):
        if operation_name == "initialize":
            return McpClientError("connect_timeout", "连接 MCP 服务超时")
        return McpClientError("call_timeout", "MCP 服务调用超时")
    if any(isinstance(error, (httpx.NetworkError, httpx.LocalProtocolError)) for error in errors):
        return McpClientError("network_error", "无法连接 MCP 服务")
    return McpClientError("protocol_error", "MCP 协议交互失败")


def _iter_exceptions(exc: BaseException):
    yield exc
    if isinstance(exc, BaseExceptionGroup):
        for nested in exc.exceptions:
            yield from _iter_exceptions(nested)


def _normalize_json_payload(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError):
        raise McpClientError("invalid_response", "MCP 服务返回无效响应") from None


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _strip_unsafe_controls(value: str) -> str:
    return "".join(char for char in value if char in "\n\t" or ord(char) >= 32)
