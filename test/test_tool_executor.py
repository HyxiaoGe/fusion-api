"""tool_executor 重试策略测试"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.mcp.agent_tools import (
    Context7McpAgentToolHandler,
    McpAgentToolBinding,
)
from app.services.mcp.context7_guard import Context7RunLibraryRegistry
from app.services.mcp.tool_contract import (
    MAX_TOOL_ARGUMENT_ARRAY_ITEMS,
    MAX_TOOL_ARGUMENT_JSON_BYTES,
    MAX_TOOL_ARGUMENT_NODES,
    validate_tool_argument_resource_limits,
)
from app.services.source_evidence_ledger import stable_web_evidence_id
from app.services.stream import tool_executor as tool_executor_module
from app.services.stream.tool_execution_result import ToolExecutionRecord
from app.services.stream.tool_executor import (
    AgentEventCompositeWriter,
    _should_retry_tool_result,
    build_successful_call_signature,
    execute_tools_parallel,
)
from app.services.tool_handlers.base import ToolResult


class ToolRetryPolicyTests(unittest.TestCase):
    def test_degraded_result_is_terminal_and_not_retried(self):
        """degraded 是已接受的降级结果，不应再触发 backoff 重试/放弃日志"""
        result = ToolResult(status="degraded", error_message="reader-service 暂时未返回内容")

        self.assertFalse(_should_retry_tool_result(result))

    def test_retry_policy_uses_structured_fields_instead_of_error_text(self):
        self.assertTrue(
            _should_retry_tool_result(
                ToolResult(
                    status="failed",
                    error_message="固定安全文案",
                    data={"error_code": "search_unavailable", "retryable": True},
                )
            )
        )
        self.assertFalse(
            _should_retry_tool_result(
                ToolResult(
                    status="failed",
                    error_message="timeout Authorization: Bearer dummy-secret",
                    data={},
                )
            )
        )


class DynamicToolExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_context7_unresolved_library_id_preserves_recovery_instruction(self):
        handler = Context7McpAgentToolHandler(
            binding=McpAgentToolBinding(
                alias="mcp_context7_query",
                server_id="context7-server",
                server_name="Context7",
                provider="context7",
                remote_tool_name="query-docs",
                config_version=1,
                tool_label="Context7 query-docs",
                definition_sha256="a" * 64,
            ),
            remote_executor=MagicMock(),
            input_schema={
                "type": "object",
                "properties": {
                    "libraryId": {"type": "string", "minLength": 3, "maxLength": 200},
                    "query": {"type": "string", "minLength": 1, "maxLength": 500},
                },
                "required": ["libraryId", "query"],
                "additionalProperties": False,
            },
            library_registry=Context7RunLibraryRegistry(),
        )
        handler.log = AsyncMock()

        records = await execute_tools_parallel(
            [
                {
                    "id": "call-context7-unresolved",
                    "name": handler.tool_name,
                    "arguments": json.dumps(
                        {
                            "libraryId": "/websites/fastapi_tiangolo",
                            "query": "How do I define lifespan events?",
                        }
                    ),
                }
            ],
            "conv-context7",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state={}, step_number=1),
        )

        record = records[0]
        self.assertEqual(record.result.data["error_code"], "context7_library_id_unresolved")
        self.assertTrue(record.result.data["local_preflight"])
        self.assertNotIn("repair", record.result.data)
        self.assertIn("请先调用库解析工具", record.format_llm_context())
        self.assertNotIn("改写成不含代码块", record.format_llm_context())
        handler.log.assert_awaited_once()

    def test_success_signature_failure_log_does_not_leak_raw_exception(self):
        secret = "Authorization: Bearer SIGNATURE_SECRET"

        class SignatureHandler:
            tool_name = "mcp_signature_probe"

            def build_successful_call_signature(self, _args):
                raise RuntimeError(secret)

        with self.assertLogs("app", level="WARNING") as captured:
            signature = build_successful_call_signature(
                {
                    "id": "call-signature-probe",
                    "name": SignatureHandler.tool_name,
                    "arguments": '{"query":"Fusion"}',
                },
                {SignatureHandler.tool_name: SignatureHandler()},
            )

        self.assertIsNone(signature)
        logs = "\n".join(captured.output)
        self.assertNotIn(secret, logs)
        self.assertNotIn("RuntimeError:", logs)
        self.assertIn("error_type=RuntimeError", logs)

    async def test_generic_argument_resource_limits_fail_local_preflight_without_execution(self):
        class Handler:
            tool_name = "mcp_resource_probe"
            supports_automatic_retry = False

            def validate_arguments(self, _args):
                return []

        handler = Handler()
        handler.execute = AsyncMock()
        handler.log = AsyncMock()
        handler._build_result_summary = MagicMock(return_value={"kind": "external_tool", "truncated": False})
        cases = {
            "bytes": (
                {"payload": "x" * MAX_TOOL_ARGUMENT_JSON_BYTES},
                [{"field": "$", "code": "max_bytes"}],
            ),
            "nodes": (
                {f"k{index}": 0 for index in range(MAX_TOOL_ARGUMENT_NODES)},
                [{"field": "$", "code": "max_nodes"}],
            ),
            "array": (
                {"items": list(range(MAX_TOOL_ARGUMENT_ARRAY_ITEMS + 1))},
                [{"field": "$", "code": "max_items"}],
            ),
        }

        for label, (arguments, expected_errors) in cases.items():
            with self.subTest(label=label):
                self.assertEqual(
                    validate_tool_argument_resource_limits(arguments),
                    expected_errors,
                )
                records = await execute_tools_parallel(
                    [
                        {
                            "id": f"call-resource-{label}",
                            "name": handler.tool_name,
                            "arguments": json.dumps(arguments),
                        }
                    ],
                    "conv-1",
                    "user-1",
                    "model-1",
                    "openai",
                    tool_handlers={handler.tool_name: handler},
                    runtime_context=SimpleNamespace(argument_repair_state={}, step_number=1),
                )
                self.assertEqual(
                    records[0].result.data["error_code"],
                    "arguments_schema_invalid",
                )
                self.assertTrue(records[0].result.data["local_preflight"])

        handler.execute.assert_not_awaited()

    async def test_structured_transient_failure_retries_once_then_succeeds(self):
        handler = SimpleNamespace(
            tool_name="web_search",
            execute=AsyncMock(
                side_effect=[
                    ToolResult(
                        status="failed",
                        error_message="搜索服务暂时不可用",
                        data={"error_code": "search_unavailable", "retryable": True},
                    ),
                    ToolResult(status="success", data={"sources": []}),
                ]
            ),
        )

        with patch("backoff._async.asyncio.sleep", new=AsyncMock()):
            result = await tool_executor_module.execute_tool_with_retry(handler, {})

        self.assertEqual(result.status, "success")
        self.assertEqual(handler.execute.await_count, 2)

    async def test_invalid_argument_json_returns_repair_context_without_calling_remote_handler(self):
        handler = MagicMock()
        handler.tool_name = "mcp_docs_a1b2c3d4"
        handler.supports_automatic_retry = False
        handler.execute = AsyncMock()
        handler.log = AsyncMock()
        handler._build_result_summary.return_value = {
            "kind": "external_tool",
            "title": "文档搜索",
            "truncated": False,
        }
        emitter = AsyncMock()
        runtime_context = SimpleNamespace(argument_repair_state={}, step_number=1)

        records = await execute_tools_parallel(
            [
                {
                    "id": "call-invalid-json",
                    "name": "mcp_docs_a1b2c3d4",
                    "arguments": '{"query":"MCP"',
                }
            ],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            emitter=emitter,
            tool_handlers={"mcp_docs_a1b2c3d4": handler},
            runtime_context=runtime_context,
        )

        record = records[0]
        self.assertEqual(record.result.status, "failed")
        self.assertEqual(record.result.data["error_code"], "arguments_json_invalid")
        self.assertTrue(record.result.data["repair"]["retryable"])
        self.assertFalse(record.result.data["repair"]["requires_user_input"])
        self.assertIn('"tool_result": "argument_repair_required"', record.format_llm_context())
        self.assertIn("有效 JSON 对象", record.format_llm_context())
        handler.execute.assert_not_awaited()
        handler.log.assert_awaited_once()
        self.assertEqual(handler.log.await_args.kwargs["input_params"], {})
        emitter.tool_call_started.assert_awaited_once()
        emitter.tool_call_completed.assert_awaited_once()
        self.assertEqual(emitter.tool_call_completed.await_args.kwargs["status"], "degraded")
        self.assertIsNone(emitter.tool_call_completed.await_args.kwargs["error"])

    async def test_repeated_invalid_argument_json_is_not_retryable_and_never_calls_handler(self):
        handler = MagicMock()
        handler.tool_name = "mcp_docs_a1b2c3d4"
        handler.supports_automatic_retry = False
        handler.execute = AsyncMock()
        handler.log = AsyncMock()
        handler._build_result_summary.return_value = {"kind": "external_tool", "truncated": False}
        repair_state = {}
        tool_call = {
            "id": "call-invalid-json",
            "name": "mcp_docs_a1b2c3d4",
            "arguments": "{invalid",
        }

        first = await execute_tools_parallel(
            [tool_call],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={"mcp_docs_a1b2c3d4": handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=1),
        )
        second = await execute_tools_parallel(
            [{**tool_call, "id": "call-invalid-json-2"}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={"mcp_docs_a1b2c3d4": handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=2),
        )

        self.assertTrue(first[0].result.data["repair"]["retryable"])
        self.assertFalse(second[0].result.data["repair"]["retryable"])
        self.assertFalse(second[0].result.data["repair"]["requires_user_input"])
        self.assertTrue(second[0].result.data["repair"]["retry_exhausted"])
        handler.execute.assert_not_awaited()

    async def test_invalid_parsable_invalid_sequence_never_reopens_repair_budget(self):
        class SchemaAwareHandler:
            tool_name = "mcp_docs_a1b2c3d4"
            supports_automatic_retry = False

            def validate_arguments(self, args):
                return [] if isinstance(args.get("query"), str) else ["query:required"]

        handler = SchemaAwareHandler()
        handler.execute = AsyncMock()
        handler.log = AsyncMock()
        handler._build_result_summary = MagicMock(return_value={"kind": "external_tool", "truncated": False})
        repair_state = {}

        first = await execute_tools_parallel(
            [{"id": "invalid-json-1", "name": handler.tool_name, "arguments": "{invalid"}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=1),
        )
        second = await execute_tools_parallel(
            [{"id": "schema-invalid", "name": handler.tool_name, "arguments": "{}"}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=2),
        )
        third = await execute_tools_parallel(
            [{"id": "invalid-json-2", "name": handler.tool_name, "arguments": "{invalid"}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=3),
        )

        self.assertTrue(first[0].result.data["repair"]["retryable"])
        self.assertFalse(second[0].result.data["repair"]["retryable"])
        self.assertEqual(second[0].result.data["error_code"], "arguments_schema_invalid")
        self.assertFalse(third[0].result.data["repair"]["retryable"])
        handler.execute.assert_not_awaited()

    async def test_structured_remote_argument_error_enters_same_one_shot_repair_protocol(self):
        class RemoteSchemaHandler:
            tool_name = "mcp_docs_a1b2c3d4"
            supports_automatic_retry = False

            def validate_arguments(self, _args):
                return []

        handler = RemoteSchemaHandler()
        handler.execute = AsyncMock(
            return_value=ToolResult(
                status="failed",
                data={
                    "error_code": "remote_invalid_arguments",
                    "validation_errors": ["query:required"],
                },
            )
        )
        handler.log = AsyncMock()
        handler._build_result_summary = MagicMock(return_value={"kind": "external_tool", "truncated": False})
        repair_state = {}

        records = await execute_tools_parallel(
            [{"id": "remote-invalid", "name": handler.tool_name, "arguments": '{"query":"Fusion"}'}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=1),
        )

        repair = records[0].result.data["repair"]
        self.assertEqual(records[0].result.data["error_code"], "remote_arguments_invalid")
        self.assertFalse(records[0].result.data["local_preflight"])
        self.assertEqual(repair["required_fields"], ["query"])
        self.assertTrue(repair["retryable"])
        self.assertFalse(repair["requires_user_input"])
        handler.execute.assert_awaited_once_with({"query": "Fusion"})

    async def test_schema_root_and_array_paths_block_execution_before_network(self):
        for error in (
            {"field": "$", "code": "additional_properties"},
            {"field": "segments[].kind", "code": "enum"},
        ):
            with self.subTest(error=error):

                class SchemaPathHandler:
                    tool_name = "mcp_docs_a1b2c3d4"
                    supports_automatic_retry = False

                    def validate_arguments(self, _args):
                        return [error]

                handler = SchemaPathHandler()
                handler.execute = AsyncMock()
                handler.log = AsyncMock()
                handler._build_result_summary = MagicMock(return_value={"kind": "external_tool", "truncated": False})

                records = await execute_tools_parallel(
                    [{"id": f"call-{error['code']}", "name": handler.tool_name, "arguments": "{}"}],
                    "conv-1",
                    "user-1",
                    "model-1",
                    "openai",
                    tool_handlers={handler.tool_name: handler},
                    runtime_context=SimpleNamespace(argument_repair_state={}, step_number=1),
                )

                self.assertEqual(records[0].result.data["error_code"], "arguments_schema_invalid")
                handler.execute.assert_not_awaited()

    async def test_repair_success_allows_later_valid_call_without_reopening_repair_budget(self):
        class RepairHandler:
            tool_name = "mcp_docs_a1b2c3d4"
            supports_automatic_retry = False

            def validate_arguments(self, args):
                return [] if isinstance(args.get("query"), str) else [{"field": "query", "code": "required"}]

        handler = RepairHandler()
        handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"payload": "ok"}))
        handler.log = AsyncMock()
        handler._build_result_summary = MagicMock(return_value={"kind": "external_tool", "truncated": False})
        repair_state = {}

        await execute_tools_parallel(
            [{"id": "invalid", "name": handler.tool_name, "arguments": "{}"}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=1),
        )
        repaired = await execute_tools_parallel(
            [{"id": "repaired", "name": handler.tool_name, "arguments": '{"query":"first"}'}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=2),
        )
        later = await execute_tools_parallel(
            [{"id": "later", "name": handler.tool_name, "arguments": '{"query":"second"}'}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=3),
        )
        later_invalid = await execute_tools_parallel(
            [{"id": "later-invalid", "name": handler.tool_name, "arguments": "{invalid"}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=4),
        )

        self.assertEqual(repaired[0].result.status, "success")
        self.assertEqual(later[0].result.status, "success")
        self.assertEqual(handler.execute.await_count, 2)
        self.assertFalse(later_invalid[0].result.data["repair"]["retryable"])

    async def test_runtime_repair_drops_unrequested_extra_arguments_before_execution(self):
        class RuntimeRepairHandler:
            tool_name = "route_compare"
            supports_automatic_retry = False

            def __init__(self):
                self.calls = []

            def validate_arguments(self, _args):
                return []

            def build_runtime_argument_repair(self, args, _runtime_context):
                if "origin_city" in args:
                    return None
                return {
                    "error_code": "route_city_required",
                    "required_fields": ["origin_city"],
                    "allowed_values": {"origin_city": ["厦门"]},
                }

            async def execute_with_runtime_context(self, args, _runtime_context):
                self.calls.append(dict(args))
                return ToolResult(status="success", data={"routes": []})

            async def log(self, **_kwargs):
                return None

            def _build_result_summary(self, _result):
                return {"kind": "external_tool", "truncated": False}

        handler = RuntimeRepairHandler()
        repair_state = {}
        base_arguments = {
            "origin": "厦门北站",
            "destination": "邮轮中心厦鼓码头",
            "modes": ["transit"],
        }

        first = await execute_tools_parallel(
            [
                {
                    "id": "route-invalid",
                    "name": handler.tool_name,
                    "arguments": json.dumps(base_arguments, ensure_ascii=False),
                }
            ],
            "conv-1",
            "user-1",
            "model-1",
            "moonshot",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=1),
        )
        repaired = await execute_tools_parallel(
            [
                {
                    "id": "route-repaired",
                    "name": handler.tool_name,
                    "arguments": json.dumps(
                        {
                            **base_arguments,
                            "origin_city": "厦门",
                            "destination_city": "厦门",
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
            "conv-1",
            "user-1",
            "model-1",
            "moonshot",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=2),
        )

        self.assertEqual(first[0].result.data["error_code"], "route_city_required")
        self.assertEqual(repaired[0].result.status, "success")
        self.assertEqual(handler.calls, [{**base_arguments, "origin_city": "厦门"}])

    async def test_later_invalid_episode_uses_a_new_repair_id_after_resolution(self):
        class RepairHandler:
            tool_name = "mcp_docs_a1b2c3d4"
            supports_automatic_retry = False

            def validate_arguments(self, args):
                return [] if isinstance(args.get("query"), str) else [{"field": "query", "code": "required"}]

        handler = RepairHandler()
        handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"payload": "ok"}))
        handler.log = AsyncMock()
        handler._build_result_summary = MagicMock(return_value={"kind": "external_tool", "truncated": False})
        repair_state = {}

        first = await execute_tools_parallel(
            [{"id": "invalid-1", "name": handler.tool_name, "arguments": "{}"}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=1),
        )
        repaired = await execute_tools_parallel(
            [{"id": "repaired", "name": handler.tool_name, "arguments": '{"query":"first"}'}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=2),
        )
        second_invalid = await execute_tools_parallel(
            [{"id": "invalid-2", "name": handler.tool_name, "arguments": "{}"}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=3),
        )
        second_valid = await execute_tools_parallel(
            [{"id": "valid-2", "name": handler.tool_name, "arguments": '{"query":"second"}'}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=4),
        )

        first_id = first[0].result.data["repair"]["repair_id"]
        second_id = second_invalid[0].result.data["repair"]["repair_id"]
        self.assertEqual(repaired[0].result.data["resolves_repair_id"], first_id)
        self.assertNotEqual(second_id, first_id)
        self.assertEqual(
            second_valid[0].result.data["repair"]["repair_id"],
            second_id,
        )
        self.assertEqual(second_valid[0].result.data["error_code"], "argument_repair_exhausted")

    async def test_last_agent_step_does_not_promise_an_unavailable_repair_round(self):
        handler = MagicMock()
        handler.tool_name = "mcp_docs_a1b2c3d4"
        handler.execute = AsyncMock()
        handler.log = AsyncMock()
        handler._build_result_summary.return_value = {"kind": "external_tool", "truncated": False}

        records = await execute_tools_parallel(
            [{"id": "last-step-invalid", "name": handler.tool_name, "arguments": "{invalid"}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(
                argument_repair_state={},
                step_number=8,
                remaining_agent_steps=0,
                remaining_agent_tool_calls_before_round=1,
            ),
        )

        repair = records[0].result.data["repair"]
        self.assertFalse(repair["retryable"])
        self.assertTrue(repair["retry_exhausted"])
        self.assertFalse(repair["requires_user_input"])
        handler.execute.assert_not_awaited()

    async def test_same_round_valid_variant_waits_until_next_round_before_execution(self):
        handler = MagicMock()
        handler.tool_name = "mcp_docs_a1b2c3d4"
        handler.supports_automatic_retry = False
        handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"text": "ok"}))
        handler.execute_with_runtime_context = None
        handler.log = AsyncMock()
        handler._build_result_summary.return_value = {"kind": "external_tool", "truncated": False}
        repair_state = {}

        same_round = await execute_tools_parallel(
            [
                {
                    "id": "call-invalid",
                    "name": "mcp_docs_a1b2c3d4",
                    "arguments": "{invalid",
                },
                {
                    "id": "call-preplanned-valid",
                    "name": "mcp_docs_a1b2c3d4",
                    "arguments": '{"query":"MCP"}',
                },
            ],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={"mcp_docs_a1b2c3d4": handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=1),
        )
        next_round = await execute_tools_parallel(
            [
                {
                    "id": "call-repaired",
                    "name": "mcp_docs_a1b2c3d4",
                    "arguments": '{"query":"MCP"}',
                }
            ],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={"mcp_docs_a1b2c3d4": handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=2),
        )

        self.assertEqual(same_round[1].result.data["error_code"], "argument_repair_deferred_to_next_round")
        self.assertTrue(same_round[1].result.data["repair"]["retryable"])
        self.assertEqual(next_round[0].result.status, "success")
        handler.execute.assert_awaited_once_with({"query": "MCP"})

    async def test_same_round_schema_invalid_call_runs_before_valid_variant_in_both_orders(self):
        class SchemaAwareHandler:
            tool_name = "mcp_schema_order"
            supports_automatic_retry = False

            def validate_arguments(self, args):
                return [] if isinstance(args.get("query"), str) else [{"field": "query", "code": "required"}]

        for order in ("valid_first", "invalid_first"):
            with self.subTest(order=order):
                handler = SchemaAwareHandler()
                handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"payload": "ok"}))
                handler.log = AsyncMock()
                handler._build_result_summary = MagicMock(return_value={"kind": "external_tool", "truncated": False})
                valid = {"id": "valid", "name": handler.tool_name, "arguments": '{"query":"MCP"}'}
                invalid = {"id": "invalid", "name": handler.tool_name, "arguments": "{}"}
                calls = [valid, invalid] if order == "valid_first" else [invalid, valid]

                records = await execute_tools_parallel(
                    calls,
                    "conv-1",
                    "user-1",
                    "model-1",
                    "openai",
                    tool_handlers={handler.tool_name: handler},
                    runtime_context=SimpleNamespace(argument_repair_state={}, step_number=1),
                )

                results_by_id = {record.tool_call["id"]: record.result for record in records}
                self.assertEqual(
                    results_by_id["invalid"].data["error_code"],
                    "arguments_schema_invalid",
                )
                self.assertEqual(
                    results_by_id["valid"].data["error_code"],
                    "argument_repair_deferred_to_next_round",
                )
                handler.execute.assert_not_awaited()

    async def test_multiple_invalid_calls_in_same_round_only_promise_one_repair(self):
        handler = MagicMock()
        handler.tool_name = "mcp_same_round_invalid"
        handler.supports_automatic_retry = False
        handler.execute = AsyncMock()
        handler.log = AsyncMock()
        handler._build_result_summary.return_value = {"kind": "external_tool", "truncated": False}
        repair_state = {}

        records = await execute_tools_parallel(
            [
                {"id": "invalid-1", "name": handler.tool_name, "arguments": "{bad-one"},
                {"id": "invalid-2", "name": handler.tool_name, "arguments": "{bad-two"},
            ],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state=repair_state, step_number=1),
        )

        self.assertTrue(records[0].result.data["repair"]["retryable"])
        self.assertEqual(records[1].result.data["error_code"], "argument_repair_coalesced")
        self.assertNotIn("repair", records[1].result.data)
        self.assertEqual(len(repair_state), 1)
        handler.execute.assert_not_awaited()

    async def test_independent_repairs_atomically_share_remaining_call_slots(self):
        class RepairHandler:
            supports_automatic_retry = False

            def __init__(self, tool_name):
                self.tool_name = tool_name

            def validate_arguments(self, args):
                return [] if isinstance(args.get("query"), str) else [{"field": "query", "code": "required"}]

        handlers = {}
        for tool_name in ("mcp_repair_one", "mcp_repair_two"):
            handler = RepairHandler(tool_name)
            handler.execute = AsyncMock()
            handler.log = AsyncMock()
            handler._build_result_summary = MagicMock(return_value={"kind": "external_tool", "truncated": False})
            handlers[tool_name] = handler

        repair_state = {}
        records = await execute_tools_parallel(
            [
                {"id": "repair-one", "name": "mcp_repair_one", "arguments": "{}"},
                {"id": "repair-two", "name": "mcp_repair_two", "arguments": "{}"},
            ],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers=handlers,
            runtime_context=SimpleNamespace(
                argument_repair_state=repair_state,
                step_number=1,
                remaining_agent_steps=1,
                remaining_agent_tool_calls_before_round=1,
            ),
        )

        repairs = [record.result.data["repair"] for record in records]
        self.assertEqual(sum(repair["retryable"] is True for repair in repairs), 1)
        self.assertEqual(sum(repair["retry_exhausted"] is True for repair in repairs), 1)
        self.assertEqual(
            sum(state["status"] == "pending" for state in repair_state.values()),
            1,
        )
        self.assertEqual(
            sum(state["status"] == "consumed" for state in repair_state.values()),
            1,
        )
        for handler in handlers.values():
            handler.execute.assert_not_awaited()

    async def test_batch_capacity_accounts_for_other_real_tool_calls(self):
        invalid_handler = MagicMock()
        invalid_handler.tool_name = "mcp_invalid"
        invalid_handler.supports_automatic_retry = False
        invalid_handler.execute = AsyncMock()
        invalid_handler.execute_with_runtime_context = None
        invalid_handler.log = AsyncMock()
        invalid_handler._build_result_summary.return_value = {"kind": "external_tool", "truncated": False}

        valid_handler = MagicMock()
        valid_handler.tool_name = "mcp_valid"
        valid_handler.supports_automatic_retry = False
        valid_handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"payload": "ok"}))
        valid_handler.execute_with_runtime_context = None
        valid_handler.log = AsyncMock()
        valid_handler._build_result_summary.return_value = {"kind": "external_tool", "truncated": False}

        records = await execute_tools_parallel(
            [
                {"id": "invalid", "name": invalid_handler.tool_name, "arguments": "{bad"},
                {"id": "valid", "name": valid_handler.tool_name, "arguments": "{}"},
            ],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={
                invalid_handler.tool_name: invalid_handler,
                valid_handler.tool_name: valid_handler,
            },
            runtime_context=SimpleNamespace(
                argument_repair_state={},
                step_number=1,
                remaining_agent_steps=1,
                remaining_agent_tool_calls_before_round=1,
            ),
        )

        results_by_id = {record.tool_call["id"]: record.result for record in records}
        self.assertFalse(results_by_id["invalid"].data["repair"]["retryable"])
        self.assertTrue(results_by_id["invalid"].data["repair"]["retry_exhausted"])
        valid_handler.execute.assert_awaited_once()

    async def test_batch_capacity_counts_missing_handler_as_consumed_tool_call(self):
        invalid_handler = MagicMock()
        invalid_handler.tool_name = "mcp_invalid_with_missing"
        invalid_handler.supports_automatic_retry = False
        invalid_handler.execute = AsyncMock()
        invalid_handler.log = AsyncMock()
        invalid_handler._build_result_summary.return_value = {"kind": "external_tool", "truncated": False}

        records = await execute_tools_parallel(
            [
                {"id": "invalid", "name": invalid_handler.tool_name, "arguments": "{bad"},
                {"id": "missing", "name": "announced_but_missing", "arguments": "{}"},
            ],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={invalid_handler.tool_name: invalid_handler},
            runtime_context=SimpleNamespace(
                argument_repair_state={},
                step_number=1,
                remaining_agent_steps=1,
                remaining_agent_tool_calls_before_round=1,
            ),
        )

        results_by_id = {record.tool_call["id"]: record.result for record in records}
        self.assertFalse(results_by_id["invalid"].data["repair"]["retryable"])
        self.assertEqual(results_by_id["missing"].status, "failed")

    async def test_remote_argument_failure_consuming_last_call_is_not_retryable(self):
        class RemoteHandler:
            tool_name = "mcp_remote_last_call"
            supports_automatic_retry = False

            def validate_arguments(self, _args):
                return []

        handler = RemoteHandler()
        handler.execute = AsyncMock(
            return_value=ToolResult(
                status="failed",
                data={
                    "error_code": "remote_invalid_arguments",
                    "validation_errors": ["query:required"],
                },
            )
        )
        handler.log = AsyncMock()
        handler._build_result_summary = MagicMock(return_value={"kind": "external_tool", "truncated": False})

        records = await execute_tools_parallel(
            [{"id": "remote-invalid", "name": handler.tool_name, "arguments": "{}"}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(
                argument_repair_state={},
                step_number=1,
                remaining_agent_steps=1,
                remaining_agent_tool_calls_before_round=1,
            ),
        )

        self.assertFalse(records[0].result.data["repair"]["retryable"])
        self.assertTrue(records[0].result.data["repair"]["retry_exhausted"])
        handler.execute.assert_awaited_once()

    async def test_unverified_remote_argument_failure_is_not_turned_into_repair_context(self):
        class RemoteHandler:
            tool_name = "mcp_remote_unverified"
            supports_automatic_retry = False

            def validate_arguments(self, _args):
                return []

        handler = RemoteHandler()
        handler.execute = AsyncMock(
            return_value=ToolResult(
                status="failed",
                data={"error_code": "remote_invalid_arguments"},
                error_message="外部工具调用失败",
            )
        )
        handler.log = AsyncMock()
        handler._build_result_summary = MagicMock(return_value={"kind": "external_tool", "truncated": False})

        records = await execute_tools_parallel(
            [{"id": "remote-invalid", "name": handler.tool_name, "arguments": "{}"}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={handler.tool_name: handler},
            runtime_context=SimpleNamespace(argument_repair_state={}, step_number=1),
        )

        self.assertNotIn("repair", records[0].result.data)

    async def test_same_batch_remote_invalid_and_success_do_not_leave_tool_level_pending_repair(self):
        class RemoteHandler:
            tool_name = "mcp_remote_mixed_batch"
            supports_automatic_retry = False

            def validate_arguments(self, _args):
                return []

        for order in ("invalid_first", "success_first"):
            with self.subTest(order=order):
                handler = RemoteHandler()

                async def execute(args):
                    if args["query"] == "bad":
                        return ToolResult(
                            status="failed",
                            data={
                                "error_code": "remote_invalid_arguments",
                                "validation_errors": ["query:required"],
                            },
                        )
                    return ToolResult(status="success", data={"payload": "ok"})

                handler.execute = AsyncMock(side_effect=execute)
                handler.log = AsyncMock()
                handler._build_result_summary = MagicMock(return_value={"kind": "external_tool", "truncated": False})
                invalid = {
                    "id": "remote-invalid",
                    "name": handler.tool_name,
                    "arguments": '{"query":"bad"}',
                }
                success = {
                    "id": "remote-success",
                    "name": handler.tool_name,
                    "arguments": '{"query":"good"}',
                }
                calls = [invalid, success] if order == "invalid_first" else [success, invalid]
                repair_state = {}

                records = await execute_tools_parallel(
                    calls,
                    "conv-1",
                    "user-1",
                    "model-1",
                    "openai",
                    tool_handlers={handler.tool_name: handler},
                    runtime_context=SimpleNamespace(
                        argument_repair_state=repair_state,
                        step_number=1,
                    ),
                )

                results = {record.tool_call["id"]: record.result for record in records}
                self.assertEqual(results["remote-success"].status, "success")
                self.assertEqual(
                    results["remote-invalid"].data["error_code"],
                    "remote_invalid_arguments",
                )
                self.assertNotIn("repair", results["remote-invalid"].data)
                self.assertEqual(repair_state, {})
                self.assertEqual(handler.execute.await_count, 2)

    async def test_identical_successful_travel_query_is_reused_without_second_execution(self):
        from app.services.mcp.flyai_travel_tools import (
            FLYAI_SEARCH_TRAINS,
            FlyAiTravelToolHandler,
            build_flyai_travel_binding,
        )

        handler = FlyAiTravelToolHandler(
            binding=build_flyai_travel_binding(FLYAI_SEARCH_TRAINS),
            client=MagicMock(),
            controls=MagicMock(),
            user_scope="scope",
        )
        handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"result_count": 1}))
        handler.log = AsyncMock()
        emitter = AsyncMock()
        successful_signatures: set[str] = set()
        first_call = {
            "id": "call-train-1",
            "name": FLYAI_SEARCH_TRAINS,
            "arguments": json.dumps(
                {
                    "origin": " 深圳 ",
                    "destination": "广州",
                    "departure_date": "2026-07-25",
                }
            ),
        }
        repeated_call = {
            "id": "call-train-2",
            "name": FLYAI_SEARCH_TRAINS,
            "arguments": {
                "departure_date": "2026-07-25",
                "destination": "广州",
                "origin": "深圳",
                "sort_by": "recommended",
                "limit": 5,
            },
        }

        records = await execute_tools_parallel(
            [first_call, repeated_call],
            "conv-1",
            "user-1",
            "model-1",
            "qwen",
            emitter=emitter,
            tool_handlers={FLYAI_SEARCH_TRAINS: handler},
            successful_tool_call_signatures=successful_signatures,
        )

        self.assertEqual(handler.execute.await_count, 1)
        self.assertEqual(handler.log.await_count, 1)
        self.assertFalse(records[0].reused)
        self.assertTrue(records[1].reused)
        self.assertIn("复用上一条成功结果", records[1].format_llm_context())
        self.assertIn("直接回答", records[1].format_llm_context())
        self.assertIsNone(records[1].build_content_block())
        self.assertEqual(emitter.tool_call_started.await_count, 1)
        self.assertEqual(emitter.tool_call_completed.await_count, 1)
        self.assertEqual(emitter.tool_result_digest.await_count, 1)

    async def test_different_travel_time_ranges_are_not_reused(self):
        from app.services.mcp.flyai_travel_tools import (
            FLYAI_SEARCH_TRAINS,
            FlyAiTravelToolHandler,
            build_flyai_travel_binding,
        )

        handler = FlyAiTravelToolHandler(
            binding=build_flyai_travel_binding(FLYAI_SEARCH_TRAINS),
            client=MagicMock(),
            controls=MagicMock(),
            user_scope="scope",
        )
        handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"result_count": 1}))
        handler.log = AsyncMock()
        successful_signatures: set[str] = set()
        calls = [
            {
                "id": f"call-train-{hour}",
                "name": FLYAI_SEARCH_TRAINS,
                "arguments": {
                    "origin": "深圳",
                    "destination": "广州",
                    "departure_date": "2026-07-25",
                    "departure_hour_start": hour,
                },
            }
            for hour in (8, 9)
        ]

        records = await execute_tools_parallel(
            calls,
            "conv-1",
            "user-1",
            "model-1",
            "qwen",
            tool_handlers={FLYAI_SEARCH_TRAINS: handler},
            successful_tool_call_signatures=successful_signatures,
        )

        self.assertEqual(handler.execute.await_count, 2)
        self.assertTrue(all(not record.reused for record in records))

    async def test_failed_identical_travel_query_can_retry_until_one_succeeds(self):
        from app.services.mcp.flyai_travel_tools import (
            FLYAI_SEARCH_TRAINS,
            FlyAiTravelToolHandler,
            build_flyai_travel_binding,
        )

        handler = FlyAiTravelToolHandler(
            binding=build_flyai_travel_binding(FLYAI_SEARCH_TRAINS),
            client=MagicMock(),
            controls=MagicMock(),
            user_scope="scope",
        )
        handler.execute = AsyncMock(
            side_effect=[
                ToolResult(status="failed", error_message="临时失败"),
                ToolResult(status="success", data={"result_count": 1}),
            ]
        )
        handler.log = AsyncMock()
        successful_signatures: set[str] = set()
        arguments = {
            "origin": "深圳",
            "destination": "广州",
            "departure_date": "2026-07-25",
        }
        calls = [{"id": f"call-train-{index}", "name": FLYAI_SEARCH_TRAINS, "arguments": arguments} for index in (1, 2)]

        records = await execute_tools_parallel(
            calls,
            "conv-1",
            "user-1",
            "model-1",
            "qwen",
            tool_handlers={FLYAI_SEARCH_TRAINS: handler},
            successful_tool_call_signatures=successful_signatures,
        )

        self.assertEqual(handler.execute.await_count, 2)
        self.assertTrue(all(not record.reused for record in records))
        self.assertEqual([record.result.status for record in records], ["failed", "success"])

    async def test_verified_research_same_batch_rejects_duplicate_canonical_read_source(self):
        from app.services.stream.network_budget import NetworkToolBudget
        from app.services.tool_handlers.url_read import UrlReadHandler

        handler = UrlReadHandler()
        handler.execute = AsyncMock(
            return_value=ToolResult(
                status="success",
                data={"url": "https://example.com/report?a=1&b=2"},
            )
        )
        handler.log = AsyncMock()
        calls = [
            {
                "id": "call-read-one",
                "name": "url_read",
                "arguments": {"url": "https://www.example.com/report?b=2&a=1&utm_source=test#section"},
                "plan_item_id": "read-one",
            },
            {
                "id": "call-read-two",
                "name": "url_read",
                "arguments": {"url": "https://example.com/report?a=1&b=2"},
                "plan_item_id": "read-two",
            },
        ]

        records = await execute_tools_parallel(
            calls,
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            network_budget=NetworkToolBudget(require_distinct_read_urls=True),
            tool_handlers={"url_read": handler},
        )

        self.assertEqual(handler.execute.await_count, 1)
        self.assertEqual([record.result.status for record in records], ["success", "degraded"])
        self.assertTrue(records[1].result.data["duplicate_read_source"])
        self.assertTrue(records[1].result.data["retryable"])

    async def test_runtime_location_is_only_forwarded_to_handler_not_events_or_logs(self):
        handler = MagicMock()
        handler.tool_name = "local_place_search"
        handler.supports_automatic_retry = False
        handler.execute = AsyncMock()
        handler.execute_with_runtime_context = AsyncMock(return_value=ToolResult(status="success", data={}))
        handler.log = AsyncMock()
        handler._build_result_summary.return_value = {"kind": "external_tool", "truncated": False}
        emitter = AsyncMock()
        runtime_context = SimpleNamespace(
            geolocation=SimpleNamespace(latitude=22.616123, longitude=114.031456, accuracy_m=10)
        )
        args = {"query": "咖啡", "anchor_source": "current_location"}

        await execute_tools_parallel(
            [{"id": "call-current", "name": "local_place_search", "arguments": args}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            emitter=emitter,
            tool_handlers={"local_place_search": handler},
            runtime_context=runtime_context,
        )

        handler.execute.assert_not_awaited()
        handler.execute_with_runtime_context.assert_awaited_once_with(args, runtime_context)
        self.assertEqual(handler.log.await_args.kwargs["input_params"], args)
        self.assertEqual(emitter.tool_call_started.await_args.kwargs["arguments"], args)
        self.assertNotIn("114.031456", str(handler.log.await_args))
        self.assertNotIn("114.031456", str(emitter.method_calls))

    async def test_run_scoped_handler_is_resolved_without_global_registration(self):
        handler = AsyncMock()
        handler.tool_name = "mcp_docs_a1b2c3d4"
        handler.supports_automatic_retry = False
        handler.execute.return_value = ToolResult(status="success", data={"text": "结果"})
        handler._build_result_summary.return_value = {
            "kind": "external_tool",
            "title": "Microsoft Learn 文档搜索",
            "truncated": False,
        }

        records = await execute_tools_parallel(
            [{"id": "call-mcp", "name": "mcp_docs_a1b2c3d4", "arguments": {"query": "MCP"}}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={"mcp_docs_a1b2c3d4": handler},
        )

        self.assertEqual(records[0].tool_name, "mcp_docs_a1b2c3d4")
        handler.execute.assert_awaited_once_with({"query": "MCP"})

    async def test_mcp_handler_failure_is_never_automatically_retried(self):
        handler = AsyncMock()
        handler.tool_name = "mcp_calendar_a1b2c3d4"
        handler.supports_automatic_retry = False
        handler.execute.return_value = ToolResult(status="failed", error_message="MCP 工具暂时不可用")

        records = await execute_tools_parallel(
            [{"id": "call-mcp", "name": "mcp_calendar_a1b2c3d4", "arguments": {"title": "聚餐"}}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={"mcp_calendar_a1b2c3d4": handler},
        )

        self.assertEqual(records[0].result.status, "failed")
        handler.execute.assert_awaited_once()

    async def test_stable_amap_product_tool_reuses_single_product_lifecycle(self):
        handler = MagicMock()
        handler.tool_name = "local_place_search"
        handler.supports_automatic_retry = False
        handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"result": {"places": []}}))
        handler.log = AsyncMock()
        handler._build_result_summary.return_value = {
            "kind": "external_tool",
            "title": "高德地点搜索",
            "result_count": 0,
            "truncated": False,
        }
        emitter = AsyncMock()

        records = await execute_tools_parallel(
            [{"id": "call-amap", "name": "local_place_search", "arguments": {"query": "咖啡"}}],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            emitter=emitter,
            tool_handlers={"local_place_search": handler},
        )

        self.assertEqual(records[0].tool_name, "local_place_search")
        handler.execute.assert_awaited_once_with({"query": "咖啡"})
        emitter.tool_call_started.assert_awaited_once()
        emitter.tool_call_completed.assert_awaited_once()
        self.assertEqual(emitter.tool_call_completed.await_args.kwargs["tool_name"], "local_place_search")

    async def test_amap_route_runs_before_place_calls_and_results_keep_model_order(self):
        request = tool_executor_module.ToolExecutionBatchRequest(
            conversation_id="conv-1",
            user_id="user-1",
            model_id="deepseek-chat",
            provider="deepseek",
        )
        place_first = {"id": "call-place-1", "name": "local_place_search", "arguments": {"query": "咖啡"}}
        route = {
            "id": "call-route",
            "name": "route_compare",
            "arguments": {"origin": "民治地铁站", "destination": "深圳市民中心"},
        }
        place_second = {
            "id": "call-place-2",
            "name": "local_place_search",
            "arguments": {"query": "安静的咖啡店"},
        }
        started = []

        async def execute_one_tool_call(_request, tool_call):
            started.append(tool_call["id"])
            await asyncio.sleep(0)
            handler = MagicMock(tool_name=tool_call["name"])
            return ToolExecutionRecord(
                tool_call=tool_call,
                result=ToolResult(status="success", data={}),
                handler=handler,
                block_id=f"blk-{tool_call['id']}",
                log_id=f"log-{tool_call['id']}",
            )

        with patch("app.services.stream.tool_executor.execute_one_tool_call", side_effect=execute_one_tool_call):
            records = await tool_executor_module.execute_tool_batch(request, [place_first, route, place_second])

        self.assertEqual(started, ["call-route", "call-place-1", "call-place-2"])
        self.assertEqual(
            [record.tool_call["id"] for record in records],
            ["call-place-1", "call-route", "call-place-2"],
        )

    async def test_same_round_travel_defers_untrusted_route_without_consuming_network_call(self):
        travel_handler = MagicMock()
        travel_handler.tool_name = "search_trains"
        travel_handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"result": {}}))
        travel_handler.execute_with_runtime_context = None
        travel_handler.log = AsyncMock()
        route_handler = MagicMock()
        route_handler.tool_name = "route_compare"
        route_handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"result": {}}))
        route_handler.execute_with_runtime_context = None
        route_handler.log = AsyncMock()
        runtime_context = SimpleNamespace(
            argument_repair_state={},
            route_city_hints={},
            step_number=1,
            remaining_agent_steps=7,
            remaining_agent_tool_calls_before_round=20,
        )

        records = await execute_tools_parallel(
            [
                {
                    "id": "call-train",
                    "name": "search_trains",
                    "arguments": {"origin": "广州", "destination": "杭州"},
                },
                {
                    "id": "call-route",
                    "name": "route_compare",
                    "arguments": {
                        "origin": "杭州东站",
                        "origin_source": "named",
                        "destination": "西湖",
                        "destination_source": "named",
                    },
                },
            ],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={
                "search_trains": travel_handler,
                "route_compare": route_handler,
            },
            runtime_context=runtime_context,
        )

        self.assertEqual(records[0].result.status, "success")
        self.assertEqual(records[1].result.status, "failed")
        self.assertEqual(
            records[1].result.data["error_code"],
            "route_requires_prior_travel_results",
        )
        self.assertTrue(records[1].result.data["local_preflight"])
        travel_handler.execute.assert_awaited_once()
        route_handler.execute.assert_not_awaited()

    async def test_same_round_travel_allows_route_from_prior_trusted_station(self):
        travel_handler = MagicMock()
        travel_handler.tool_name = "search_trains"
        travel_handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"result": {}}))
        travel_handler.execute_with_runtime_context = None
        travel_handler.log = AsyncMock()
        route_handler = MagicMock()
        route_handler.tool_name = "route_compare"
        route_handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"result": {}}))
        route_handler.execute_with_runtime_context = None
        route_handler.log = AsyncMock()
        runtime_context = SimpleNamespace(
            argument_repair_state={},
            route_city_hints={"杭州东站": ("杭州",)},
            step_number=2,
            remaining_agent_steps=6,
            remaining_agent_tool_calls_before_round=18,
        )

        records = await execute_tools_parallel(
            [
                {
                    "id": "call-train",
                    "name": "search_trains",
                    "arguments": {"origin": "广州", "destination": "杭州"},
                },
                {
                    "id": "call-route",
                    "name": "route_compare",
                    "arguments": {
                        "origin": "杭州东站",
                        "origin_source": "named",
                        "destination": "西湖",
                        "destination_source": "named",
                    },
                },
            ],
            "conv-1",
            "user-1",
            "model-1",
            "openai",
            tool_handlers={
                "search_trains": travel_handler,
                "route_compare": route_handler,
            },
            runtime_context=runtime_context,
        )

        self.assertEqual([record.result.status for record in records], ["success", "success"])
        route_handler.execute.assert_awaited_once()


class WebSearchRedirectPresentationTests(unittest.TestCase):
    def test_read_alternative_redirect_has_safe_context_without_search_block(self):
        from app.services.tool_handlers.web_search import WebSearchHandler

        handler = WebSearchHandler()
        result = ToolResult(
            status="degraded",
            error_message="已有未读取候选来源，已暂停继续搜索",
            data={
                "query": "继续搜索同一问题",
                "read_alternatives_available": True,
                "unread_candidate_count": 1,
            },
        )

        context = handler.format_llm_context(result)
        summary = handler._build_result_summary(result)

        self.assertIsNone(handler.build_content_block(result, "blk-search", "log-search"))
        self.assertIn("优先", context)
        self.assertIn("候选来源", context)
        self.assertNotIn("url_read", context)
        self.assertNotIn("reader-service", context)
        self.assertEqual(summary["title"], "优先读取已有候选")


class AgentEventCompositeWriterTests(unittest.IsolatedAsyncioTestCase):
    async def test_writes_redis_then_progress_then_trajectory(self):
        calls = []

        class RedisWriter:
            async def append_chunk(self, conversation_id, task_id, chunk_type, payload):
                calls.append(("redis", conversation_id, task_id, chunk_type, payload))

        class Recorder:
            def record_chunk(self, conversation_id, chunk_type, payload):
                calls.append(("progress", conversation_id, chunk_type, payload))

        class TrajectoryRecorder:
            async def record_chunk(self, conversation_id, chunk_type, payload):
                calls.append(("trajectory", conversation_id, chunk_type, payload))

        writer = AgentEventCompositeWriter(
            redis_writer=RedisWriter(),
            recorder=Recorder(),
            trajectory_recorder=TrajectoryRecorder(),
        )

        await writer.append_chunk("c1", "task-1", "agent_event", {"type": "run_progress_updated"})

        self.assertEqual(
            calls,
            [
                ("redis", "c1", "task-1", "agent_event", {"type": "run_progress_updated"}),
                ("progress", "c1", "agent_event", {"type": "run_progress_updated"}),
                ("trajectory", "c1", "agent_event", {"type": "run_progress_updated"}),
            ],
        )

    async def test_required_redis_failure_is_raised_unchanged_before_auxiliary_sinks(self):
        failure = RuntimeError("redis failed")
        progress = MagicMock()
        trajectory = AsyncMock()
        redis_writer = AsyncMock()
        redis_writer.append_chunk.side_effect = failure
        writer = AgentEventCompositeWriter(
            redis_writer=redis_writer,
            recorder=progress,
            trajectory_recorder=trajectory,
        )

        with self.assertRaises(RuntimeError) as raised:
            await writer.append_chunk("c1", "task-1", "agent_event", {"type": "plan_snapshot"})

        self.assertIs(raised.exception, failure)
        progress.record_chunk.assert_not_called()
        trajectory.record_chunk.assert_not_awaited()

    async def test_progress_failure_is_fail_open_and_does_not_skip_trajectory(self):
        progress = MagicMock()
        progress.record_chunk.side_effect = RuntimeError("progress failed")
        trajectory = AsyncMock()
        writer = AgentEventCompositeWriter(
            redis_writer=AsyncMock(),
            recorder=progress,
            trajectory_recorder=trajectory,
        )

        await writer.append_chunk("c1", "task-1", "agent_event", {"type": "step_started"})

        trajectory.record_chunk.assert_awaited_once()

    async def test_trajectory_failure_is_fail_open(self):
        trajectory = AsyncMock()
        trajectory.record_chunk.side_effect = RuntimeError("trajectory failed")
        writer = AgentEventCompositeWriter(
            redis_writer=AsyncMock(),
            recorder=MagicMock(),
            trajectory_recorder=trajectory,
        )

        await writer.append_chunk("c1", "task-1", "agent_event", {"type": "step_started"})


class ToolExecutorMessageIdTests(unittest.IsolatedAsyncioTestCase):
    async def test_progress_digest_does_not_swallow_stream_ownership_lost(self):
        from app.services.stream.tool_executor import emit_progress_digest_events
        from app.services.stream_state_service import StreamOwnershipLostError

        emitter = AsyncMock()
        emitter.tool_result_digest.side_effect = StreamOwnershipLostError("ownership lost")
        request = tool_executor_module.ToolExecutionBatchRequest(
            conversation_id="conv-1",
            user_id="user-1",
            model_id="gpt-4",
            provider="openai",
            emitter=emitter,
        )
        record = ToolExecutionRecord(
            tool_call={"id": "call-1", "name": "web_search", "arguments": {}},
            result=ToolResult(status="success", data={"sources": []}),
            handler=MagicMock(tool_name="web_search"),
            block_id="blk-1",
            log_id="log-1",
        )

        with self.assertRaises(StreamOwnershipLostError):
            await emit_progress_digest_events(request=request, record=record)

    async def test_progress_digest_does_not_swallow_stream_write_unavailable(self):
        from app.services.stream.tool_executor import emit_progress_digest_events
        from app.services.stream_state_service import StreamWriteUnavailableError

        emitter = AsyncMock()
        emitter.tool_result_digest.side_effect = StreamWriteUnavailableError("Redis write failed")
        request = tool_executor_module.ToolExecutionBatchRequest(
            conversation_id="conv-1",
            user_id="user-1",
            model_id="gpt-4",
            provider="openai",
            emitter=emitter,
        )
        record = ToolExecutionRecord(
            tool_call={"id": "call-1", "name": "web_search", "arguments": {}},
            result=ToolResult(status="success", data={"sources": []}),
            handler=MagicMock(tool_name="web_search"),
            block_id="blk-1",
            log_id="log-1",
        )

        with self.assertRaises(StreamWriteUnavailableError):
            await emit_progress_digest_events(request=request, record=record)

    async def test_execute_tool_batch_accepts_request_and_preserves_log_context(self):
        request_cls = getattr(tool_executor_module, "ToolExecutionBatchRequest")
        execute_tool_batch = getattr(tool_executor_module, "execute_tool_batch")
        handler = AsyncMock()
        handler.tool_name = "web_search"
        handler.execute.return_value = ToolResult(status="success", data={"items": []})
        tool_call = {
            "id": "call-1",
            "name": "web_search",
            "arguments": {"query": "redis stream"},
        }

        request = request_cls(
            conversation_id="conv-1",
            user_id="user-1",
            model_id="gpt-4",
            provider="openai",
            trace_id="trace-1",
            step_number=2,
            message_id="assistant-1",
            emitter=None,
            network_budget=None,
        )

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            records = await execute_tool_batch(request, [tool_call])

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertIsInstance(record, ToolExecutionRecord)
        self.assertIs(record.tool_call, tool_call)
        self.assertIs(record.handler, handler)
        handler.execute.assert_awaited_once_with({"query": "redis stream"})
        handler.log.assert_awaited_once()
        self.assertEqual(handler.log.await_args.kwargs["message_id"], "assistant-1")
        self.assertEqual(handler.log.await_args.kwargs["trace_id"], "trace-1")
        self.assertEqual(handler.log.await_args.kwargs["step_number"], 2)
        self.assertEqual(handler.log.await_args.kwargs["input_params"], {"query": "redis stream"})

    async def test_execute_tool_batch_preserves_input_order_when_calls_complete_out_of_order(self):
        request_cls = getattr(tool_executor_module, "ToolExecutionBatchRequest")
        execute_tool_batch = getattr(tool_executor_module, "execute_tool_batch")
        slow_handler = AsyncMock()
        slow_handler.tool_name = "web_search"
        slow_handler.execute.side_effect = lambda _args: ToolResult(status="success", data={"order": "slow"})
        fast_handler = AsyncMock()
        fast_handler.tool_name = "url_read"
        fast_handler.execute.return_value = ToolResult(status="success", data={"order": "fast"})
        slow_call = {"id": "call-slow", "name": "web_search", "arguments": {"query": "redis"}}
        fast_call = {"id": "call-fast", "name": "url_read", "arguments": {"url": "https://example.com"}}
        both_started = asyncio.Event()
        started = []
        completed = []

        async def execute_one_tool_call(_request, tool_call):
            started.append(tool_call["id"])
            if len(started) == 2:
                both_started.set()
            if tool_call is slow_call:
                await asyncio.wait_for(both_started.wait(), timeout=1)
                await asyncio.sleep(0.01)
                completed.append(tool_call["id"])
                return ToolExecutionRecord(
                    tool_call=slow_call,
                    result=ToolResult(status="success", data={"order": "slow"}),
                    handler=slow_handler,
                    block_id="blk_slow",
                    log_id="log_slow",
                )
            await asyncio.wait_for(both_started.wait(), timeout=1)
            completed.append(tool_call["id"])
            return ToolExecutionRecord(
                tool_call=fast_call,
                result=ToolResult(status="success", data={"order": "fast"}),
                handler=fast_handler,
                block_id="blk_fast",
                log_id="log_fast",
            )

        request = request_cls(
            conversation_id="conv-1",
            user_id="user-1",
            model_id="gpt-4",
            provider="openai",
        )
        with patch("app.services.stream.tool_executor.execute_one_tool_call", side_effect=execute_one_tool_call):
            records = await execute_tool_batch(request, [slow_call, fast_call])

        self.assertEqual([record.tool_call for record in records], [slow_call, fast_call])
        self.assertEqual(completed, ["call-fast", "call-slow"])

    async def test_execute_tools_parallel_returns_record_with_named_access_and_handler_helpers(self):
        """成功执行后返回命名记录，并通过记录方法委派 handler 格式化逻辑。"""
        handler = MagicMock()
        handler.tool_name = "web_search"
        handler.execute = AsyncMock(return_value=ToolResult(status="success", data={"items": []}))
        handler.log = AsyncMock()
        handler.format_llm_context.return_value = "LLM 可见上下文"
        handler.build_content_block.return_value = {"type": "tool_result"}

        tool_call = {
            "id": "call-1",
            "name": "web_search",
            "arguments": {"query": "redis stream"},
        }

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            records = await execute_tools_parallel(
                [tool_call],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertIsInstance(record, ToolExecutionRecord)
        self.assertIs(record.tool_call, tool_call)
        self.assertEqual(record.tool_name, "web_search")
        self.assertEqual(record.result.status, "success")
        self.assertIs(record.handler, handler)
        self.assertTrue(record.block_id.startswith("blk_"))
        self.assertIsInstance(record.log_id, str)

        self.assertEqual(record.format_llm_context(), "LLM 可见上下文")
        handler.format_llm_context.assert_called_once_with(record.result)
        self.assertEqual(record.build_content_block(), {"type": "tool_result"})
        handler.build_content_block.assert_called_once_with(record.result, record.block_id, record.log_id)

    async def test_execute_tools_parallel_returns_record_for_unknown_handler_with_safe_fallbacks(self):
        """未知工具仍返回记录，但不会产生 content block，LLM context 使用安全兜底文案。"""
        tool_call = {"id": "call-404", "name": "missing_tool", "arguments": {}}

        with patch("app.services.tool_handlers.get_handler", return_value=None):
            records = await execute_tools_parallel(
                [tool_call],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertIsInstance(record, ToolExecutionRecord)
        self.assertIs(record.tool_call, tool_call)
        self.assertEqual(record.tool_name, "missing_tool")
        self.assertIsNone(record.handler)
        self.assertEqual(record.result.status, "failed")
        self.assertEqual(record.result.error_message, "未知工具: missing_tool")
        self.assertEqual(record.format_llm_context(), "工具未取得可用结果，不能把该工具结果作为依据。")
        self.assertIsNone(record.build_content_block())

    async def test_execute_tools_parallel_unknown_handler_does_not_emit_events(self):
        tool_call = {"id": "call-404", "name": "missing_tool", "arguments": {"query": "x"}}
        emitter = AsyncMock()

        with patch("app.services.tool_handlers.get_handler", return_value=None):
            records = await execute_tools_parallel(
                [tool_call],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
                emitter=emitter,
            )

        self.assertEqual(records[0].result.status, "failed")
        emitter.tool_call_started.assert_not_awaited()
        emitter.tool_call_completed.assert_not_awaited()

    async def test_execute_tools_parallel_passes_message_id_to_handler_log(self):
        """工具调用日志必须关联最终 assistant message"""
        handler = AsyncMock()
        handler.tool_name = "web_search"
        handler.execute.return_value = ToolResult(status="success")

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            await execute_tools_parallel(
                [{"id": "call-1", "name": "web_search", "arguments": {"query": "redis stream"}}],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
            )

        handler.log.assert_awaited_once()
        self.assertEqual(handler.log.await_args.kwargs["message_id"], "assistant-1")

    async def test_execute_tools_parallel_with_emitter_keeps_events_log_and_record(self):
        """emitter 路径仍发工具事件、写日志，并返回命名记录。"""
        handler = MagicMock()
        handler.tool_name = "web_search"
        handler.execute = AsyncMock(return_value=ToolResult(status="success"))
        handler.log = AsyncMock()
        handler._build_result_summary.return_value = {"kind": "search", "truncated": False}
        emitter = AsyncMock()
        tool_call = {"id": "call-1", "name": "web_search", "arguments": {"query": "redis stream"}}

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            records = await execute_tools_parallel(
                [tool_call],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                trace_id="trace-1",
                step_number=2,
                message_id="assistant-1",
                emitter=emitter,
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertIsInstance(record, ToolExecutionRecord)
        self.assertIs(record.tool_call, tool_call)
        self.assertIs(record.handler, handler)
        self.assertEqual(record.result.status, "success")
        emitter.tool_call_started.assert_awaited_once_with(
            tool_call_id="call-1",
            tool_name="web_search",
            arguments={"query": "redis stream"},
        )
        emitter.tool_call_completed.assert_awaited_once()
        completed_kwargs = emitter.tool_call_completed.await_args.kwargs
        self.assertEqual(completed_kwargs["tool_call_id"], "call-1")
        self.assertEqual(completed_kwargs["tool_name"], "web_search")
        self.assertEqual(completed_kwargs["status"], "success")
        handler.log.assert_awaited_once()
        self.assertEqual(handler.log.await_args.kwargs["message_id"], "assistant-1")
        self.assertEqual(handler.log.await_args.kwargs["trace_id"], "trace-1")
        self.assertEqual(handler.log.await_args.kwargs["step_number"], 2)

    async def test_execute_tools_parallel_with_emitter_emits_tool_digest_and_evidence(self):
        handler = MagicMock()
        handler.tool_name = "web_search"
        handler.execute = AsyncMock(
            return_value=ToolResult(
                status="success",
                data={
                    "sources": [
                        {
                            "title": "官方发布页",
                            "url": "https://example.com/news",
                            "description": "官方页面确认发布时间。",
                            "content": "官方页面确认发布时间，并给出原始公告。",
                        }
                    ]
                },
            )
        )
        handler.log = AsyncMock()
        handler._build_result_summary.return_value = {
            "kind": "search",
            "title": "找到 1 条搜索结果",
            "count": 1,
            "truncated": False,
        }
        emitter = AsyncMock()
        tool_call = {
            "id": "call-1",
            "name": "web_search",
            "arguments": {"query": "redis stream"},
            "plan_item_id": "search",
        }

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            await execute_tools_parallel(
                [tool_call],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                trace_id="trace-1",
                step_number=2,
                message_id="assistant-1",
                emitter=emitter,
            )

        emitter.tool_result_digest.assert_awaited_once()
        digest_kwargs = emitter.tool_result_digest.await_args.kwargs
        self.assertEqual(digest_kwargs["tool_call_id"], "call-1")
        self.assertEqual(digest_kwargs["tool_name"], "web_search")
        self.assertEqual(digest_kwargs["status"], "success")
        self.assertEqual(digest_kwargs["title"], "搜索完成")
        self.assertEqual(digest_kwargs["plan_item_id"], "search")
        self.assertEqual(emitter.tool_call_started.await_args.kwargs["plan_item_id"], "search")
        self.assertEqual(emitter.tool_call_completed.await_args.kwargs["plan_item_id"], "search")
        expected_evidence_id = stable_web_evidence_id("https://example.com/news", fallback="ev-call-1-0")
        self.assertEqual(digest_kwargs["source_refs"], [expected_evidence_id])
        self.assertLessEqual(len(digest_kwargs["key_findings"]), 5)

        emitter.evidence_item_upserted.assert_awaited_once()
        evidence_kwargs = emitter.evidence_item_upserted.await_args.kwargs
        self.assertEqual(evidence_kwargs["tool_call_id"], "call-1")
        self.assertEqual(evidence_kwargs["evidence"]["id"], expected_evidence_id)
        self.assertEqual(evidence_kwargs["evidence"]["title"], "官方发布页")
        self.assertEqual(evidence_kwargs["evidence"]["domain"], "example.com")

    async def test_execute_tools_parallel_uses_network_budget_normalized_args_for_handler_and_log(self):
        from app.services.stream.network_budget import NetworkToolBudget

        handler = AsyncMock()
        handler.tool_name = "web_search"
        handler.execute.return_value = ToolResult(status="success", data={"sources": []})

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            results = await execute_tools_parallel(
                [
                    {
                        "id": "call-1",
                        "name": "web_search",
                        "arguments": {
                            "query": "redis",
                            "count": 99,
                            "domains": ["https://Redis.io/docs", "bad domain"],
                            "recency_days": 0,
                        },
                    }
                ],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                network_budget=NetworkToolBudget(),
            )

        normalized = {
            "query": "redis",
            "count": 5,
            "context_source_limit": 5,
            "search_budget": "standard",
            "recency_days": 1,
        }
        handler.execute.assert_awaited_once_with(normalized)
        self.assertEqual(handler.log.await_args.kwargs["input_params"], normalized)
        self.assertEqual(results[0].result.data["budget_decision"]["action"], "execute")
        self.assertEqual(results[0].result.data["budget_decision"]["reason_code"], "initial_search")

    async def test_execute_tools_parallel_returns_degraded_when_network_budget_exhausted_without_handler_execute(self):
        from app.services.stream.network_budget import NetworkToolBudget

        handler = AsyncMock()
        handler.tool_name = "web_search"
        handler.execute.return_value = ToolResult(status="success")
        budget = NetworkToolBudget(web_search_calls=4)

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            results = await execute_tools_parallel(
                [{"id": "call-5", "name": "web_search", "arguments": {"query": "q5", "count": 8}}],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                network_budget=budget,
            )

        record = results[0]
        self.assertIsInstance(record, ToolExecutionRecord)
        self.assertEqual(record.tool_name, "web_search")
        self.assertEqual(record.result.status, "degraded")
        self.assertTrue(record.result.data["budget_limited"])
        self.assertEqual(record.result.data["requested_count"], 5)
        self.assertEqual(record.result.data["context_source_limit"], 5)
        self.assertEqual(record.result.data["search_budget"], "standard")
        self.assertIs(record.handler, handler)
        handler.execute.assert_not_awaited()
        handler.log.assert_awaited_once()

    async def test_execute_tools_parallel_budget_exhausted_with_emitter_emits_events_before_log(self):
        from app.services.stream.network_budget import NetworkToolBudget

        order = []

        handler = MagicMock()
        handler.tool_name = "web_search"
        handler.execute = AsyncMock()
        handler.log = AsyncMock(side_effect=lambda **_kwargs: order.append("log"))
        handler._build_result_summary.return_value = {"kind": "search", "truncated": False}

        emitter = AsyncMock()

        async def record_started(**_kwargs):
            order.append("started")

        async def record_completed(**_kwargs):
            order.append("completed")

        emitter.tool_call_started.side_effect = record_started
        emitter.tool_call_completed.side_effect = record_completed

        budget = NetworkToolBudget(web_search_calls=4)

        with patch("app.services.tool_handlers.get_handler", return_value=handler):
            results = await execute_tools_parallel(
                [{"id": "call-5", "name": "web_search", "arguments": {"query": "q5", "count": 8}}],
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
                emitter=emitter,
                network_budget=budget,
            )

        record = results[0]
        self.assertEqual(record.result.status, "degraded")
        handler.execute.assert_not_awaited()
        self.assertEqual(order, ["started", "completed", "log"])
        emitter.tool_call_started.assert_awaited_once_with(
            tool_call_id="call-5",
            tool_name="web_search",
            arguments={
                "query": "q5",
                "count": 5,
                "context_source_limit": 5,
                "search_budget": "standard",
            },
        )
        emitter.tool_call_completed.assert_awaited_once_with(
            tool_call_id="call-5",
            tool_name="web_search",
            status="degraded",
            duration_ms=0,
            result_summary={"kind": "search", "truncated": False},
            error="web_search 已达到本轮联网预算",
        )
        handler.log.assert_awaited_once()
        self.assertEqual(handler.log.await_args.kwargs["message_id"], "assistant-1")


if __name__ == "__main__":
    unittest.main()
