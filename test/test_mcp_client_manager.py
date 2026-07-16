import asyncio
import logging
import os
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx

os.environ["DATABASE_URL"] = "sqlite:///./fusion-test.db"

from app.services.mcp.client import (  # noqa: E402
    McpClientError,
    McpClientManager,
    McpClientPolicy,
    McpConnectionConfig,
    _QueryParameterTransport,
)


class FakeSession:
    def __init__(self, *, tools=None, call_result=None, initialize_error=None, list_error=None):
        self.tools = tools or []
        self.call_result = call_result
        self.initialize_error = initialize_error
        self.list_error = list_error
        self.list_cursors: list[str | None] = []
        self.calls: list[tuple[str, dict]] = []

    async def initialize(self):
        if self.initialize_error:
            raise self.initialize_error
        return SimpleNamespace(serverInfo=SimpleNamespace(name="fake", version="1.0"))

    async def list_tools(self, cursor=None):
        if self.list_error:
            raise self.list_error
        self.list_cursors.append(cursor)
        return SimpleNamespace(tools=self.tools, nextCursor=None)

    async def call_tool(self, name, arguments, read_timeout_seconds=None):
        self.calls.append((name, arguments))
        return self.call_result or SimpleNamespace(
            isError=False,
            model_dump=lambda **_: {"content": [{"type": "text", "text": "ok"}], "isError": False},
        )


class FakeConnector:
    def __init__(self, session):
        self.session = session
        self.connections: list[dict] = []
        self.closed = False

    @asynccontextmanager
    async def connect(self, **kwargs):
        self.connections.append(kwargs)
        yield self.session

    async def close(self):
        self.closed = True


def build_policy(**overrides):
    values = {
        "allowed_hosts": frozenset({"dashscope.aliyuncs.com"}),
        "allowed_credential_refs": frozenset({"DASHSCOPE_API_KEY"}),
        "connect_timeout_seconds": 1.0,
        "call_timeout_seconds": 2.0,
        "idempotent_max_attempts": 2,
        "idempotent_total_timeout_seconds": 2.0,
        "retry_backoff_seconds": 0,
        "max_discovery_pages": 2,
        "max_discovered_tools": 3,
        "max_tool_description_chars": 100,
        "max_tool_schema_bytes": 1024,
        "max_response_bytes": 4096,
    }
    values.update(overrides)
    return McpClientPolicy(**values)


def build_config(**overrides):
    values = {
        "server_id": "server-1",
        "provider": "aliyun",
        "endpoint_url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
        "auth_type": "bearer",
        "auth_name": None,
        "credential_ref": "DASHSCOPE_API_KEY",
        "allowed_tools": [],
    }
    values.update(overrides)
    return McpConnectionConfig(**values)


class McpClientManagerTests(unittest.TestCase):
    def test_non_json_arguments_are_invalid_without_opening_connector(self):
        connector = FakeConnector(FakeSession())
        manager = McpClientManager(
            policy=build_policy(),
            connector=connector,
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )

        for arguments in ({"value": float("nan")}, {"value": float("inf")}, {"value": object()}):
            with self.subTest(arguments=arguments):
                with self.assertRaises(McpClientError) as raised:
                    asyncio.run(manager.call_tool(build_config(allowed_tools=["search"]), "search", arguments))
                self.assertEqual(raised.exception.code, "invalid_arguments")

        self.assertEqual(connector.connections, [])

    def test_idempotent_discovery_retries_once_after_nested_transient_timeout(self):
        request = httpx.Request("POST", "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp")
        wrapped_timeout = RuntimeError("sdk transport failed")
        wrapped_timeout.__cause__ = httpx.ConnectTimeout("connect timeout", request=request)

        class FlakyDiscoverySession(FakeSession):
            def __init__(self):
                super().__init__(
                    tools=[SimpleNamespace(name="search", description="搜索", inputSchema={"type": "object"})]
                )
                self.list_attempts = 0

            async def list_tools(self, cursor=None):
                self.list_attempts += 1
                if self.list_attempts == 1:
                    raise wrapped_timeout
                return await super().list_tools(cursor)

        session = FlakyDiscoverySession()
        connector = FakeConnector(session)
        manager = McpClientManager(
            policy=build_policy(),
            connector=connector,
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )

        tools = asyncio.run(manager.list_tools(build_config()))

        self.assertEqual([tool["name"] for tool in tools], ["search"])
        self.assertEqual(session.list_attempts, 2)
        self.assertEqual(len(connector.connections), 2)

    def test_idempotent_discovery_retries_once_after_protocol_error(self):
        class FlakyProtocolSession(FakeSession):
            def __init__(self):
                super().__init__(
                    tools=[SimpleNamespace(name="search", description="搜索", inputSchema={"type": "object"})]
                )
                self.list_attempts = 0

            async def list_tools(self, cursor=None):
                self.list_attempts += 1
                if self.list_attempts == 1:
                    raise RuntimeError("transient protocol failure")
                return await super().list_tools(cursor)

        session = FlakyProtocolSession()
        connector = FakeConnector(session)
        manager = McpClientManager(
            policy=build_policy(),
            connector=connector,
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )

        tools = asyncio.run(manager.list_tools(build_config()))

        self.assertEqual([tool["name"] for tool in tools], ["search"])
        self.assertEqual(session.list_attempts, 2)
        self.assertEqual(len(connector.connections), 2)

    def test_idempotent_discovery_retries_exception_group_remote_protocol_error(self):
        class FlakyRemoteProtocolSession(FakeSession):
            def __init__(self):
                super().__init__(
                    tools=[SimpleNamespace(name="search", description="搜索", inputSchema={"type": "object"})]
                )
                self.list_attempts = 0

            async def list_tools(self, cursor=None):
                self.list_attempts += 1
                if self.list_attempts == 1:
                    raise ExceptionGroup("task group cleanup", [httpx.RemoteProtocolError("peer closed")])
                return await super().list_tools(cursor)

        session = FlakyRemoteProtocolSession()
        connector = FakeConnector(session)
        manager = McpClientManager(
            policy=build_policy(),
            connector=connector,
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )

        tools = asyncio.run(manager.list_tools(build_config()))

        self.assertEqual([tool["name"] for tool in tools], ["search"])
        self.assertEqual(session.list_attempts, 2)
        self.assertEqual(len(connector.connections), 2)

    def test_idempotent_discovery_stops_after_second_transient_failure(self):
        class AlwaysFailingDiscoverySession(FakeSession):
            def __init__(self):
                super().__init__()
                self.list_attempts = 0

            async def list_tools(self, cursor=None):
                self.list_attempts += 1
                raise RuntimeError("transient protocol failure")

        session = AlwaysFailingDiscoverySession()
        connector = FakeConnector(session)
        manager = McpClientManager(
            policy=build_policy(),
            connector=connector,
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )

        with self.assertRaises(McpClientError) as raised:
            asyncio.run(manager.list_tools(build_config()))

        self.assertEqual(raised.exception.code, "protocol_error")
        self.assertEqual(raised.exception.safe_message, "MCP 协议交互失败")
        self.assertEqual(session.list_attempts, 2)
        self.assertEqual(len(connector.connections), 2)

    def test_idempotent_discovery_total_budget_returns_safe_error(self):
        class BudgetedDiscoverySession(FakeSession):
            def __init__(self):
                super().__init__()
                self.list_attempts = 0

            async def list_tools(self, cursor=None):
                self.list_attempts += 1
                if self.list_attempts == 1:
                    raise RuntimeError("transient protocol failure")
                await asyncio.sleep(0.2)
                return await super().list_tools(cursor)

        session = BudgetedDiscoverySession()
        connector = FakeConnector(session)
        manager = McpClientManager(
            policy=build_policy(idempotent_total_timeout_seconds=0.02),
            connector=connector,
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )

        with self.assertRaises(McpClientError) as raised:
            asyncio.run(manager.list_tools(build_config()))

        self.assertEqual(raised.exception.code, "call_timeout")
        self.assertEqual(raised.exception.safe_message, "MCP 服务调用超时")
        self.assertEqual(session.list_attempts, 2)
        self.assertEqual(len(connector.connections), 2)

    def test_tool_execution_never_retries_transient_failure(self):
        request = httpx.Request("POST", "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp")

        class FailingToolSession(FakeSession):
            def __init__(self):
                super().__init__()
                self.call_attempts = 0

            async def call_tool(self, name, arguments, read_timeout_seconds=None):
                self.call_attempts += 1
                raise httpx.ReadTimeout("read timeout", request=request)

        session = FailingToolSession()
        connector = FakeConnector(session)
        manager = McpClientManager(
            policy=build_policy(),
            connector=connector,
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )

        with self.assertRaises(McpClientError) as raised:
            asyncio.run(manager.call_tool(build_config(allowed_tools=["search"]), "search", {"query": "Fusion"}))

        self.assertEqual(raised.exception.code, "call_timeout")
        self.assertEqual(session.call_attempts, 1)
        self.assertEqual(len(connector.connections), 1)

    def test_non_retryable_discovery_error_is_not_retried(self):
        connector = FakeConnector(FakeSession(tools=[SimpleNamespace(name="bad name", inputSchema={})]))
        manager = McpClientManager(
            policy=build_policy(),
            connector=connector,
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )

        with self.assertRaises(McpClientError) as raised:
            asyncio.run(manager.list_tools(build_config()))

        self.assertEqual(raised.exception.code, "invalid_response")
        self.assertEqual(len(connector.connections), 1)

    def test_bearer_credential_strips_crlf_before_building_header(self):
        connector = FakeConnector(FakeSession())
        manager = McpClientManager(
            policy=build_policy(),
            connector=connector,
            environ={"DASHSCOPE_API_KEY": "  test-secret\r\n"},
        )

        asyncio.run(manager.test_connection(build_config()))

        self.assertEqual(connector.connections[0]["headers"], {"Authorization": "Bearer test-secret"})

    def test_exception_message_and_log_never_leak_bearer_secret(self):
        secret = "test-secret"
        session = FakeSession(initialize_error=httpx.LocalProtocolError(f"Illegal Authorization: Bearer {secret}"))
        manager = McpClientManager(
            policy=build_policy(),
            connector=FakeConnector(session),
            environ={"DASHSCOPE_API_KEY": f"{secret}\r"},
        )

        with self.assertLogs("app.services.mcp.client", level="WARNING") as captured:
            with self.assertRaises(McpClientError) as raised:
                asyncio.run(manager.test_connection(build_config()))

        self.assertEqual(raised.exception.code, "network_error")
        self.assertEqual(raised.exception.safe_message, "无法连接 MCP 服务")
        self.assertNotIn(secret, raised.exception.safe_message)
        self.assertNotIn(secret, "\n".join(captured.output))

    def test_sdk_transport_logger_cannot_emit_session_id_or_raw_exception(self):
        sdk_logger = logging.getLogger("mcp.client.streamable_http")
        session_id = "mcp-session-secret"
        raw_exception = "raw exception with Authorization Bearer bearer-secret"
        records: list[str] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                records.append(self.format(record))

        root_logger = logging.getLogger()
        handler = CaptureHandler()
        root_logger.addHandler(handler)
        try:
            sdk_logger.info("Received session ID: %s", session_id)
            sdk_logger.exception(raw_exception)
        finally:
            root_logger.removeHandler(handler)

        self.assertTrue(sdk_logger.disabled)
        self.assertFalse(sdk_logger.propagate)
        self.assertEqual(records, [])

    def test_nested_exception_group_maps_401_without_exposing_original_error(self):
        secret = "test-secret"
        request = httpx.Request("POST", "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp")
        response = httpx.Response(401, request=request)
        http_error = httpx.HTTPStatusError(
            f"401 with Authorization Bearer {secret}",
            request=request,
            response=response,
        )
        session = FakeSession(initialize_error=ExceptionGroup("task group", [http_error]))
        manager = McpClientManager(
            policy=build_policy(),
            connector=FakeConnector(session),
            environ={"DASHSCOPE_API_KEY": secret},
        )

        with self.assertRaises(McpClientError) as raised:
            asyncio.run(manager.test_connection(build_config()))

        self.assertEqual(raised.exception.code, "auth_failed")
        self.assertEqual(raised.exception.safe_message, "MCP 服务鉴权失败")
        self.assertNotIn(secret, str(raised.exception))

    def test_http_429_maps_to_safe_rate_limited_error_without_retry_or_raw_response(self):
        secret = "amap-secret-response"
        request = httpx.Request("POST", "https://mcp.amap.com/mcp")
        response = httpx.Response(429, text=f"quota exceeded: {secret}", request=request)
        http_error = httpx.HTTPStatusError(
            f"429 upstream body={secret}",
            request=request,
            response=response,
        )

        class RateLimitedSession(FakeSession):
            def __init__(self):
                super().__init__()
                self.call_attempts = 0

            async def call_tool(self, name, arguments, read_timeout_seconds=None):
                self.call_attempts += 1
                raise http_error

        session = RateLimitedSession()
        manager = McpClientManager(
            policy=build_policy(allowed_hosts=frozenset({"mcp.amap.com"})),
            connector=FakeConnector(session),
            environ={"DASHSCOPE_API_KEY": "credential-secret"},
        )
        config = build_config(
            endpoint_url="https://mcp.amap.com/mcp",
            allowed_tools=["search"],
        )

        with self.assertRaises(McpClientError) as raised:
            asyncio.run(manager.call_tool(config, "search", {"keywords": "民治烤肉"}))

        self.assertEqual(raised.exception.code, "rate_limited")
        self.assertEqual(raised.exception.safe_message, "MCP 服务请求过于频繁")
        self.assertNotIn(secret, str(raised.exception))
        self.assertEqual(session.call_attempts, 1)

    def test_redirect_is_blocked_and_classified_without_following_target(self):
        request = httpx.Request("POST", "https://dashscope.aliyuncs.com/mcp")
        response = httpx.Response(307, headers={"location": "http://127.0.0.1/internal"}, request=request)
        redirect_error = httpx.HTTPStatusError("redirect", request=request, response=response)
        manager = McpClientManager(
            policy=build_policy(),
            connector=FakeConnector(FakeSession(initialize_error=redirect_error)),
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )

        with self.assertRaises(McpClientError) as raised:
            asyncio.run(manager.test_connection(build_config()))

        self.assertEqual(raised.exception.code, "redirect_blocked")
        self.assertNotIn("127.0.0.1", str(raised.exception))

    def test_endpoint_and_credential_reference_must_hit_independent_allowlists(self):
        manager = McpClientManager(
            policy=build_policy(),
            connector=FakeConnector(FakeSession()),
            environ={"DATABASE_URL": "postgresql://secret"},
        )

        with self.assertRaises(McpClientError) as host_error:
            asyncio.run(
                manager.test_connection(
                    build_config(endpoint_url="https://evil.example/mcp", credential_ref="DASHSCOPE_API_KEY")
                )
            )
        self.assertEqual(host_error.exception.code, "endpoint_not_allowed")

        with self.assertRaises(McpClientError) as credential_error:
            asyncio.run(manager.test_connection(build_config(credential_ref="DATABASE_URL")))
        self.assertEqual(credential_error.exception.code, "credential_not_allowed")

    def test_endpoint_rejects_credentials_query_fragment_and_non_https(self):
        manager = McpClientManager(
            policy=build_policy(),
            connector=FakeConnector(FakeSession()),
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )

        invalid_urls = [
            "http://dashscope.aliyuncs.com/mcp",
            "https://user:pass@dashscope.aliyuncs.com/mcp",
            "https://dashscope.aliyuncs.com/mcp?key=secret",
            "https://dashscope.aliyuncs.com/mcp#fragment",
        ]
        for endpoint_url in invalid_urls:
            with self.subTest(endpoint_url=endpoint_url):
                with self.assertRaises(McpClientError) as raised:
                    asyncio.run(manager.test_connection(build_config(endpoint_url=endpoint_url)))
                self.assertEqual(raised.exception.code, "invalid_endpoint")

    def test_query_credential_is_injected_only_into_transport_parameters(self):
        connector = FakeConnector(FakeSession())
        manager = McpClientManager(
            policy=build_policy(),
            connector=connector,
            environ={"DASHSCOPE_API_KEY": "query-secret\r\n"},
        )

        asyncio.run(
            manager.test_connection(
                build_config(auth_type="query", auth_name="api_key", credential_ref="DASHSCOPE_API_KEY")
            )
        )

        connection = connector.connections[0]
        self.assertEqual(connection["endpoint_url"], build_config().endpoint_url)
        self.assertEqual(connection["query_params"], {"api_key": "query-secret"})
        self.assertEqual(connection["headers"], {})

    def test_amap_missing_query_credential_fails_before_network_without_secret_detail(self):
        connector = FakeConnector(FakeSession())
        manager = McpClientManager(
            policy=build_policy(
                allowed_hosts=frozenset({"mcp.amap.com"}),
                allowed_credential_refs=frozenset({"AMAP_MCP_API_KEY"}),
            ),
            connector=connector,
            environ={},
        )
        config = build_config(
            provider="amap",
            endpoint_url="https://mcp.amap.com/mcp",
            auth_type="query",
            auth_name="key",
            credential_ref="AMAP_MCP_API_KEY",
        )

        with self.assertRaises(McpClientError) as raised:
            asyncio.run(manager.test_connection(config))

        self.assertEqual(raised.exception.code, "credential_unavailable")
        self.assertEqual(raised.exception.safe_message, "MCP 凭证不可用")
        self.assertNotIn("AMAP_MCP_API_KEY", str(raised.exception))
        self.assertEqual(connector.connections, [])

    def test_query_transport_keeps_secret_out_of_httpx_log_and_response_url(self):
        secret = "query-secret"
        captured_urls = []

        async def handler(request):
            captured_urls.append(str(request.url))
            return httpx.Response(200, json={"ok": True})

        transport = _QueryParameterTransport({"api_key": secret})
        transport._transport = httpx.MockTransport(handler)

        async def request_once():
            async with httpx.AsyncClient(transport=transport) as client:
                return await client.get("https://dashscope.aliyuncs.com/mcp")

        with self.assertLogs("httpx", level="INFO") as captured_logs:
            response = asyncio.run(request_once())

        self.assertIn(secret, captured_urls[0])
        self.assertNotIn(secret, "\n".join(captured_logs.output))
        self.assertNotIn(secret, str(response.request.url))

    def test_configurable_allowlist_supports_public_no_auth_mcp(self):
        connector = FakeConnector(FakeSession())
        manager = McpClientManager(
            policy=build_policy(allowed_hosts=frozenset({"learn.microsoft.com"})),
            connector=connector,
            environ={},
        )
        config = build_config(
            provider="microsoft",
            endpoint_url="https://learn.microsoft.com/api/mcp",
            auth_type="none",
            credential_ref=None,
        )

        asyncio.run(manager.test_connection(config))

        self.assertEqual(connector.connections[0]["endpoint_url"], "https://learn.microsoft.com/api/mcp")

    def test_discovery_normalizes_tools_and_rejects_unbounded_pagination(self):
        first_page = SimpleNamespace(
            tools=[SimpleNamespace(name="search", description="搜索文档", inputSchema={"type": "object"})],
            nextCursor="next",
        )
        second_page = SimpleNamespace(
            tools=[SimpleNamespace(name="read", description=None, inputSchema={"type": "object"})],
            nextCursor=None,
        )

        class PaginatedSession(FakeSession):
            async def list_tools(self, cursor=None):
                return first_page if cursor is None else second_page

        manager = McpClientManager(
            policy=build_policy(),
            connector=FakeConnector(PaginatedSession()),
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )

        tools = asyncio.run(manager.list_tools(build_config()))

        self.assertEqual(
            tools,
            [
                {"name": "search", "description": "搜索文档", "input_schema": {"type": "object"}},
                {"name": "read", "description": None, "input_schema": {"type": "object"}},
            ],
        )

        endless_page = SimpleNamespace(tools=[], nextCursor="again")
        endless_manager = McpClientManager(
            policy=build_policy(max_discovery_pages=1),
            connector=FakeConnector(FakeSession()),
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )
        endless_manager.connector.session.list_tools = lambda cursor=None: None

        class EndlessSession(FakeSession):
            async def list_tools(self, cursor=None):
                return endless_page

        endless_manager.connector = FakeConnector(EndlessSession())
        with self.assertRaises(McpClientError) as raised:
            asyncio.run(endless_manager.list_tools(build_config()))
        self.assertEqual(raised.exception.code, "invalid_response")

    def test_discovery_enforces_tool_count_schema_size_and_description_limit(self):
        oversized_count = [
            SimpleNamespace(name=f"tool_{index}", description=None, inputSchema={"type": "object"})
            for index in range(4)
        ]
        count_manager = McpClientManager(
            policy=build_policy(max_discovered_tools=3),
            connector=FakeConnector(FakeSession(tools=oversized_count)),
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )
        with self.assertRaises(McpClientError) as count_error:
            asyncio.run(count_manager.list_tools(build_config()))
        self.assertEqual(count_error.exception.code, "invalid_response")

        oversized_schema = {"type": "object", "description": "x" * 2_000}
        schema_manager = McpClientManager(
            policy=build_policy(max_tool_schema_bytes=100),
            connector=FakeConnector(
                FakeSession(tools=[SimpleNamespace(name="search", description=None, inputSchema=oversized_schema)])
            ),
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )
        with self.assertRaises(McpClientError) as schema_error:
            asyncio.run(schema_manager.list_tools(build_config()))
        self.assertEqual(schema_error.exception.code, "invalid_response")

        description_manager = McpClientManager(
            policy=build_policy(max_tool_description_chars=10),
            connector=FakeConnector(
                FakeSession(
                    tools=[SimpleNamespace(name="search", description="a" * 100, inputSchema={"type": "object"})]
                )
            ),
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )
        tools = asyncio.run(description_manager.list_tools(build_config()))
        self.assertEqual(tools[0]["description"], "a" * 10)

    def test_allowed_tools_empty_denies_call_and_subset_allows_call(self):
        session = FakeSession()
        manager = McpClientManager(
            policy=build_policy(),
            connector=FakeConnector(session),
            environ={"DASHSCOPE_API_KEY": "test-secret"},
        )

        with self.assertRaises(McpClientError) as denied:
            asyncio.run(manager.call_tool(build_config(allowed_tools=[]), "search", {"query": "Fusion"}))
        self.assertEqual(denied.exception.code, "tool_not_allowed")

        result = asyncio.run(manager.call_tool(build_config(allowed_tools=["search"]), "search", {"query": "Fusion"}))
        self.assertFalse(result["isError"])
        self.assertEqual(session.calls, [("search", {"query": "Fusion"})])

    def test_close_delegates_to_connector(self):
        connector = FakeConnector(FakeSession())
        manager = McpClientManager(policy=build_policy(), connector=connector, environ={})

        asyncio.run(manager.close())

        self.assertTrue(connector.closed)


if __name__ == "__main__":
    unittest.main()
