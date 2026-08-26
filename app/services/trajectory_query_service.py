"""普通用户轨迹快照组装；只投影脱敏事件账本。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from app.db.models import AgentEvent, AgentSession
from app.db.trajectory_repository import TrajectoryRepository
from app.schemas.admin_trajectory import AdminTrajectorySnapshot, AdminTrajectoryToolCall
from app.schemas.trajectory import (
    LlmNodeDetail,
    SystemPromptNodeDetail,
    SystemPromptSnapshot,
    ToolNodeDetail,
    TrajectoryCompleteness,
    TrajectoryEventRecord,
    TrajectoryLlmRoundSummary,
    TrajectoryNodeDetailResponse,
    TrajectoryRunListResponse,
    TrajectoryRunSummary,
    TrajectorySnapshot,
)
from app.services.admin_audit_sanitizer import sanitize_admin_value
from app.services.admin_audit_service import AdminAuditService
from app.services.agent.trajectory_projector import project_trajectory
from app.services.agent.trajectory_reconciliation import (
    resolve_ledger_watermark,
    resolve_user_trajectory_status_from_rows,
)
from app.services.mcp.amap_product_tools import AMAP_PRODUCT_TOOL_NAMES
from app.services.mcp.flyai_travel_tools import FLYAI_TRAVEL_TOOL_NAMES
from app.utils.prompt_fingerprint import fingerprint_system_messages
from app.utils.time import as_utc, utc_now

_USER_WEB_SEARCH_ARGUMENT_FIELDS = ("query", "count", "domains", "recency_days", "intent")
_USER_WEB_SEARCH_RESULT_FIELDS = (
    "result_count",
    "requested_count",
    "actual_count",
    "context_source_count",
    "context_source_limit",
    "fallback_used",
    "budget_limited",
)
_USER_WEB_SEARCH_SOURCE_FIELDS = ("title", "url", "favicon", "status")
_USER_URL_READ_ARGUMENT_FIELDS = ("url", "reason")
_USER_URL_READ_RESULT_FIELDS = ("url", "safe_log_url", "title", "status", "content_length", "length", "reason")
_USER_MCP_RESULT_FIELDS = ("status", "payload_bytes", "error_code", "subcall_attempt_count")
_USER_FLYAI_RESULT_FIELDS = ("status", "result_count", "response_bytes", "error_code")


class TrajectoryQueryService:
    """受鉴权 repository 支撑的只读轨迹查询服务。"""

    def __init__(
        self,
        repository: TrajectoryRepository,
        *,
        max_events_per_run: int,
        max_runs_per_conversation: int,
        detail_settle_grace_seconds: float = 5,
        now_provider: Callable[[], datetime] = utc_now,
    ) -> None:
        if max_events_per_run <= 0 or max_runs_per_conversation <= 0:
            raise ValueError("轨迹查询上限必须大于 0")
        if detail_settle_grace_seconds < 0:
            raise ValueError("轨迹详情收敛宽限期不能为负数")
        self._repository = repository
        self._max_events_per_run = max_events_per_run
        self._max_runs_per_conversation = max_runs_per_conversation
        self._detail_settle_grace = timedelta(seconds=detail_settle_grace_seconds)
        self._now_provider = now_provider

    def list_runs(self, conversation_id: str, user_id: str) -> TrajectoryRunListResponse | None:
        rows = self._repository.list_runs(conversation_id, user_id, self._max_runs_per_conversation + 1)
        if rows is None:
            return None
        truncated = len(rows) > self._max_runs_per_conversation
        bounded_rows = rows[: self._max_runs_per_conversation]
        watermark = self._ledger_watermark()
        items = [
            self._run_summary(
                run,
                resolve_user_trajectory_status_from_rows(run.created_at, meta, watermark).trajectory_status,
                llm_detail_schema_version=meta.llm_detail_schema_version if meta is not None else None,
                llm_round_count=meta.llm_round_count if meta is not None else 0,
            )
            for run, meta in bounded_rows
        ]
        return TrajectoryRunListResponse(items=self._grouping_order(items), truncated=truncated)

    def get_user_snapshot(self, conversation_id: str, run_id: str, user_id: str) -> TrajectorySnapshot | None:
        row = self._repository.get_run(conversation_id, run_id, user_id)
        if row is None:
            return None
        return self._snapshot_from_row(conversation_id, run_id, row)

    def get_admin_snapshot(self, conversation_id: str, run_id: str) -> AdminTrajectorySnapshot | None:
        """构造管理员诊断，但不承担权限判断或访问审计。"""
        row = self._repository.get_run_for_admin(conversation_id, run_id)
        if row is None:
            return None
        snapshot = self._snapshot_from_row(conversation_id, run_id, row)
        tool_rows = self._repository.list_tool_diagnostics(run_id, self._max_events_per_run + 1)
        tool_calls_truncated = len(tool_rows) > self._max_events_per_run
        tools = [self._admin_tool_call(tool) for tool in tool_rows[: self._max_events_per_run]]
        return AdminTrajectorySnapshot(
            snapshot=snapshot,
            tool_calls=tools,
            tool_calls_truncated=tool_calls_truncated,
        )

    def get_user_tool_node_detail(
        self,
        conversation_id: str,
        run_id: str,
        tool_call_id: str,
        user_id: str,
    ) -> TrajectoryNodeDetailResponse | None:
        return self._get_tool_node_detail(conversation_id, run_id, tool_call_id, user_id=user_id)

    def get_admin_tool_node_detail(
        self,
        conversation_id: str,
        run_id: str,
        tool_call_id: str,
    ) -> TrajectoryNodeDetailResponse | None:
        return self._get_tool_node_detail(conversation_id, run_id, tool_call_id, user_id=None)

    def get_user_llm_node_detail(
        self,
        conversation_id: str,
        run_id: str,
        llm_round_id: str,
        user_id: str,
    ) -> TrajectoryNodeDetailResponse | None:
        return self._get_llm_node_detail(conversation_id, run_id, llm_round_id, user_id=user_id)

    def get_admin_llm_node_detail(
        self,
        conversation_id: str,
        run_id: str,
        llm_round_id: str,
    ) -> TrajectoryNodeDetailResponse | None:
        return self._get_llm_node_detail(conversation_id, run_id, llm_round_id, user_id=None)

    def get_user_system_prompt_node_detail(
        self,
        conversation_id: str,
        run_id: str,
        user_id: str,
    ) -> TrajectoryNodeDetailResponse | None:
        """正文仅来自同一 Run 的持久快照，不执行当前提示词模板。"""
        run = self._repository.get_detail_run(conversation_id, run_id, user_id)
        if run is None:
            return None
        event = self._repository.get_system_prompt_prepared_event(conversation_id, run_id)
        if event is not None and not isinstance(event.payload, dict):
            return self._unavailable_system_prompt_detail("degraded", "system_prompt_detail_invalid")
        metadata = event.payload if event is not None else None
        if metadata is not None and metadata.get("status") == "failed":
            return self._unavailable_system_prompt_detail("degraded", "system_prompt_assembly_failed")

        snapshot = self._repository.get_system_prompt_snapshot(conversation_id, run_id, user_id)
        if snapshot is None:
            if metadata is None:
                awaiting_ledger = run.status == "running" or (
                    run.terminal_at is not None
                    and as_utc(self._now_provider()) - as_utc(run.terminal_at) <= self._detail_settle_grace
                )
                if awaiting_ledger:
                    return self._unavailable_system_prompt_detail("pending", "system_prompt_detail_settling")
            if metadata is not None and metadata.get("detail_status") in {"available", "degraded"}:
                return self._unavailable_system_prompt_detail("degraded", "system_prompt_detail_missing")
            return self._unavailable_system_prompt_detail("not_recorded", "system_prompt_not_recorded")

        try:
            persisted = SystemPromptSnapshot.model_validate(snapshot)
            messages = [{"role": "system", "content": section.content} for section in persisted.sections]
            valid = (
                len({section.section_id for section in persisted.sections}) == len(persisted.sections)
                and persisted.fingerprint == fingerprint_system_messages(messages)
                and persisted.char_count == sum(len(section.content) for section in persisted.sections)
                and (metadata is None or metadata.get("fingerprint") == persisted.fingerprint)
                and (
                    metadata is None
                    or all(
                        metadata[key] == value
                        for key, value in {
                            "section_ids": [section.section_id for section in persisted.sections],
                            "template_version": persisted.template_version,
                            "char_count": persisted.char_count,
                        }.items()
                        if key in metadata
                    )
                )
            )
        except (ValidationError, UnicodeError):
            valid = False
        if not valid:
            return self._unavailable_system_prompt_detail("degraded", "system_prompt_detail_invalid")

        return TrajectoryNodeDetailResponse(
            status="available",
            node_type="system_prompt",
            available_sections=["summary", "prompt"],
            detail=SystemPromptNodeDetail(
                template_version=persisted.template_version,
                fingerprint=persisted.fingerprint,
                char_count=persisted.char_count,
                sections=persisted.sections,
            ),
            redacted_fields=[],
            truncated_fields=[],
        )

    def _get_llm_node_detail(
        self,
        conversation_id: str,
        run_id: str,
        llm_round_id: str,
        *,
        user_id: str | None,
    ) -> TrajectoryNodeDetailResponse | None:
        run = self._repository.get_detail_run(conversation_id, run_id, user_id)
        if run is None:
            return None
        lifecycle = self._repository.list_llm_round_lifecycle_events(
            conversation_id=conversation_id,
            run_id=run_id,
            llm_round_id=llm_round_id,
        )
        if not lifecycle:
            return None
        if self._repository.get_llm_detail_schema_version(run_id) != 1:
            return None

        detail = self._repository.get_exact_llm_detail(
            conversation_id=conversation_id,
            run_id=run_id,
            llm_round_id=llm_round_id,
        )
        if detail is not None:
            sections = ["summary"]
            if detail.reasoning_text is not None:
                sections.append("thinking")
            if detail.content_text is not None:
                sections.append("output")
            sections.append("timing")
            return TrajectoryNodeDetailResponse(
                status="available",
                node_type="llm",
                available_sections=sections,
                detail=LlmNodeDetail(
                    llm_round_id=llm_round_id,
                    reasoning_text=detail.reasoning_text,
                    output_text=detail.content_text,
                ),
                redacted_fields=list(detail.redacted_fields or []),
                truncated_fields=list(detail.truncated_fields or []),
            )

        terminal = next(
            (
                event
                for event in reversed(lifecycle)
                if event.event_type in {"llm_round_completed", "llm_round_failed", "llm_round_cancelled"}
            ),
            None,
        )
        if terminal is None:
            if run.status == "running":
                return self._unavailable_llm_detail("pending", "llm_round_in_progress")
            terminal_at = run.terminal_at
        else:
            terminal_at = terminal.event_ts
        if terminal_at is not None:
            age = as_utc(self._now_provider()) - as_utc(terminal_at)
            if age <= self._detail_settle_grace:
                return self._unavailable_llm_detail("pending", "llm_detail_settling")
        return self._unavailable_llm_detail("degraded", "llm_detail_missing")

    def _get_tool_node_detail(
        self,
        conversation_id: str,
        run_id: str,
        tool_call_id: str,
        *,
        user_id: str | None,
    ) -> TrajectoryNodeDetailResponse | None:
        run = self._repository.get_detail_run(conversation_id, run_id, user_id)
        if run is None:
            return None
        tool = self._repository.get_exact_tool_detail(
            conversation_id=conversation_id,
            user_id=run.user_id,
            run_id=run_id,
            tool_call_id=tool_call_id,
        )
        if tool is not None:
            return self._available_tool_detail(tool_call_id, tool)

        watermark = self._repository.get_detail_watermark()
        if watermark is None or as_utc(run.created_at) < as_utc(watermark):
            return self._unavailable_tool_detail("not_recorded", "detail_not_recorded")
        if run.status == "running":
            return self._unavailable_tool_detail("pending", "run_in_progress")
        if run.terminal_at is not None:
            age = as_utc(self._now_provider()) - as_utc(run.terminal_at)
            if age <= self._detail_settle_grace:
                return self._unavailable_tool_detail("pending", "detail_settling")
        return self._unavailable_tool_detail("degraded", "tool_detail_missing")

    @staticmethod
    def _available_tool_detail(tool_call_id: str, tool) -> TrajectoryNodeDetailResponse:
        safe_item = TrajectoryQueryService._user_tool_item(tool)
        payload = safe_item["payload"] or None
        result = safe_item["result"] or None
        sections = ["summary"]
        if payload is not None:
            sections.append("payload")
        if result is not None:
            sections.append("result")
        if safe_item["duration_ms"] is not None:
            sections.append("timing")
        return TrajectoryNodeDetailResponse(
            status="available",
            available_sections=sections,
            detail=ToolNodeDetail(
                tool_call_id=tool_call_id,
                tool_name=safe_item["tool_name"],
                status=safe_item["status"],
                duration_ms=safe_item["duration_ms"],
                payload=payload,
                result=result,
                error=safe_item["error"],
            ),
            redacted_fields=safe_item["redacted_fields"],
            reason=None,
        )

    @staticmethod
    def _user_tool_item(tool) -> dict:
        """普通用户 Tool Detail 的独立窄投影，不继承管理员内部诊断字段。"""
        raw_payload = tool.input_params if isinstance(tool.input_params, dict) else {}
        raw_result = tool.output_data if isinstance(tool.output_data, dict) else {}
        payload_projection: dict = {}
        result_projection: dict = {}
        nested_result_cropped = False

        if tool.tool_name == "web_search":
            payload_projection = {
                key: raw_payload[key] for key in _USER_WEB_SEARCH_ARGUMENT_FIELDS if key in raw_payload
            }
            result_projection = {key: raw_result[key] for key in _USER_WEB_SEARCH_RESULT_FIELDS if key in raw_result}
            sources = raw_result.get("sources")
            if isinstance(sources, list):
                projected_sources = []
                for source in sources[:20]:
                    if not isinstance(source, dict):
                        nested_result_cropped = True
                        continue
                    projected_source = {key: source[key] for key in _USER_WEB_SEARCH_SOURCE_FIELDS if key in source}
                    projected_sources.append(projected_source)
                    nested_result_cropped = nested_result_cropped or len(projected_source) != len(source)
                result_projection["sources"] = projected_sources
                nested_result_cropped = nested_result_cropped or len(sources) > 20
        elif tool.tool_name == "url_read":
            payload_projection = {key: raw_payload[key] for key in _USER_URL_READ_ARGUMENT_FIELDS if key in raw_payload}
            result_projection = {key: raw_result[key] for key in _USER_URL_READ_RESULT_FIELDS if key in raw_result}
        elif tool.tool_name.startswith("mcp_") or tool.tool_name in AMAP_PRODUCT_TOOL_NAMES:
            payload_projection = (
                {"argument_count": raw_payload["argument_count"]} if "argument_count" in raw_payload else {}
            )
            result_projection = {key: raw_result[key] for key in _USER_MCP_RESULT_FIELDS if key in raw_result}
        elif tool.tool_name in FLYAI_TRAVEL_TOOL_NAMES:
            payload_projection = (
                {"argument_count": raw_payload["argument_count"]} if "argument_count" in raw_payload else {}
            )
            result_projection = {key: raw_result[key] for key in _USER_FLYAI_RESULT_FIELDS if key in raw_result}

        payload, payload_fields = sanitize_admin_value(
            payload_projection,
            max_string_chars=1000,
            max_list_items=30,
        )
        result, result_fields = sanitize_admin_value(
            result_projection,
            max_string_chars=1000,
            max_list_items=30,
        )
        redacted_fields = {
            *[f"payload.{field}" for field in payload_fields],
            *[f"result.{field}" for field in result_fields],
        }
        if set(raw_payload) != set(payload_projection):
            redacted_fields.add("payload")
        if set(raw_result) != set(result_projection) or nested_result_cropped:
            redacted_fields.add("result")
        if tool.error_message:
            redacted_fields.add("error")
        return {
            "tool_name": tool.tool_name,
            "status": tool.status,
            "duration_ms": tool.duration_ms,
            "payload": payload,
            "result": result,
            "error": AdminAuditService._error_projection(tool.error_message, tool.status),
            "redacted_fields": sorted(redacted_fields),
        }

    @staticmethod
    def _unavailable_tool_detail(status: str, reason: str) -> TrajectoryNodeDetailResponse:
        return TrajectoryNodeDetailResponse(
            status=status,
            available_sections=[],
            detail=None,
            redacted_fields=[],
            reason=reason,
        )

    @staticmethod
    def _unavailable_llm_detail(status: str, reason: str) -> TrajectoryNodeDetailResponse:
        return TrajectoryNodeDetailResponse(
            status=status,
            node_type="llm",
            available_sections=[],
            detail=None,
            redacted_fields=[],
            truncated_fields=[],
            reason=reason,
        )

    @staticmethod
    def _unavailable_system_prompt_detail(status: str, reason: str) -> TrajectoryNodeDetailResponse:
        return TrajectoryNodeDetailResponse(
            status=status,
            node_type="system_prompt",
            available_sections=[],
            detail=None,
            redacted_fields=[],
            truncated_fields=[],
            reason=reason,
        )

    @staticmethod
    def _admin_tool_call(tool) -> AdminTrajectoryToolCall:
        item = {key: value for key, value in AdminAuditService._tool_item(tool).items() if key != "trace_id"}
        created_at = item.get("created_at")
        if isinstance(created_at, datetime):
            item["created_at"] = TrajectoryQueryService._tool_call_log_created_at_as_utc(created_at)
        return AdminTrajectoryToolCall(association="run", **item)

    @staticmethod
    def _tool_call_log_created_at_as_utc(value: datetime | None) -> datetime | None:
        """按 ToolCallLog 的既有北京时间墙钟语义规范化，不能替代通用 as_utc。"""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=ZoneInfo("Asia/Shanghai")).astimezone(UTC)
        return value.astimezone(UTC)

    def _snapshot_from_row(
        self,
        conversation_id: str,
        run_id: str,
        row: tuple[AgentSession, object | None],
    ) -> TrajectorySnapshot:
        run, meta = row
        event_rows = self._repository.list_events(conversation_id, run_id, self._max_events_per_run + 1)
        truncated = len(event_rows) > self._max_events_per_run
        loaded_events = event_rows[: self._max_events_per_run]
        assessment = resolve_user_trajectory_status_from_rows(
            run.created_at,
            meta,
            self._ledger_watermark(),
        )
        records = [self._event_record(event) for event in loaded_events]
        projection = project_trajectory(
            records,
            run_status=run.status,
            run_ended_at=run.terminal_at,
            truncated=truncated,
        )
        llm_summaries = []
        if meta is not None and meta.llm_detail_schema_version == 1:
            llm_summaries = [
                TrajectoryLlmRoundSummary(
                    llm_round_id=detail.llm_round_id,
                    reasoning_preview=detail.reasoning_preview,
                    output_preview=detail.output_preview,
                )
                for detail in self._repository.list_llm_round_details(conversation_id, run_id)
            ]
        return TrajectorySnapshot(
            run=self._run_summary(
                run,
                assessment.trajectory_status,
                llm_detail_schema_version=meta.llm_detail_schema_version if meta is not None else None,
                llm_round_count=meta.llm_round_count if meta is not None else 0,
            ),
            records=projection.records,
            spans=projection.spans,
            completeness=TrajectoryCompleteness(
                status=assessment.trajectory_status,
                degraded_reason=assessment.degraded_reason,
                event_count=meta.event_count if meta is not None else None,
                expected_last_sequence=meta.expected_last_sequence if meta is not None else None,
                loaded_event_count=len(loaded_events),
                first_sequence=loaded_events[0].sequence if loaded_events else None,
                last_sequence=loaded_events[-1].sequence if loaded_events else None,
            ),
            truncated=truncated,
            llm_round_summaries=llm_summaries,
        )

    @staticmethod
    def _run_summary(
        run: AgentSession,
        trajectory_status: str,
        *,
        llm_detail_schema_version: int | None = None,
        llm_round_count: int = 0,
    ) -> TrajectoryRunSummary:
        return TrajectoryRunSummary(
            run_id=run.id,
            message_id=run.message_id,
            turn_message_id=run.turn_message_id,
            attempt_index=run.attempt_index,
            status=run.status,
            trajectory_status=trajectory_status,
            total_steps=run.total_steps or 0,
            total_tool_calls=run.total_tool_calls or 0,
            duration_ms=run.total_duration_ms,
            started_at=run.created_at,
            ended_at=run.terminal_at,
            llm_detail_schema_version=llm_detail_schema_version,
            llm_round_count=llm_round_count,
        )

    @staticmethod
    def _event_record(event: AgentEvent) -> TrajectoryEventRecord:
        return TrajectoryEventRecord(
            run_id=event.run_id,
            sequence=event.sequence,
            event_type=event.event_type,
            schema_version=event.schema_version if event.schema_version is not None else 0,
            timestamp=event.event_ts,
            step_id=event.step_id,
            tool_call_id=event.tool_call_id,
            parent_step_id=event.parent_step_id,
            trace_id=event.trace_id,
            payload=dict(event.payload or {}),
        )

    def _ledger_watermark(self):
        return resolve_ledger_watermark(self._repository.list_ledger_watermark_rows())

    @staticmethod
    def _grouping_order(items: list[TrajectoryRunSummary]) -> list[TrajectoryRunSummary]:
        groups: dict[str, list[TrajectoryRunSummary]] = {}
        for item in items:
            group_key = item.turn_message_id or f"__unlinked__:{item.run_id}"
            groups.setdefault(group_key, []).append(item)
        ordered: list[TrajectoryRunSummary] = []
        for group in sorted(
            groups.values(),
            key=lambda values: max(value.started_at for value in values),
            reverse=True,
        ):
            ordered.extend(
                sorted(
                    group,
                    key=lambda value: (
                        value.attempt_index is None,
                        value.attempt_index if value.attempt_index is not None else 0,
                        value.started_at,
                        value.run_id,
                    ),
                )
            )
        return ordered
