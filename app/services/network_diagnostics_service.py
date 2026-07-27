from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.models import AgentSession, Message, ToolCallLog
from app.schemas.network_diagnostics import (
    NetworkDiagnosticsResponse,
    NetworkDiagnosticsSummary,
    NetworkDiagnosticsToolItem,
    ToolStatus,
)
from app.services.stream.itinerary_observability import (
    ItineraryToolObservation,
    build_itinerary_run_payload,
    build_itinerary_tool_observation_from_values,
)


class NetworkDiagnosticsService:
    def __init__(self, db: Session):
        self.db = db

    def build_for_message(
        self,
        *,
        conversation_id: str,
        message_id: str,
        is_admin: bool,
    ) -> NetworkDiagnosticsResponse:
        session = (
            self.db.query(AgentSession)
            .filter(
                AgentSession.conversation_id == conversation_id,
                AgentSession.message_id == message_id,
            )
            .order_by(AgentSession.created_at.desc())
            .first()
        )
        logs = (
            self.db.query(ToolCallLog)
            .filter(
                ToolCallLog.conversation_id == conversation_id,
                ToolCallLog.message_id == message_id,
            )
            .order_by(ToolCallLog.created_at.asc())
            .all()
        )
        message = (
            self.db.query(Message)
            .filter(
                Message.conversation_id == conversation_id,
                Message.id == message_id,
            )
            .first()
        )

        tools = [self._tool_item_from_log(log, is_admin=is_admin) for log in logs]
        is_empty = session is None and not tools
        return NetworkDiagnosticsResponse(
            conversation_id=conversation_id,
            message_id=message_id,
            run_id=session.id if session else None,
            visibility="admin" if is_admin else "user",
            summary=self._build_summary(session, tools),
            tools=tools,
            is_empty=is_empty,
            admin=(
                self._build_itinerary_admin_summary(
                    session,
                    tools,
                    message.content if message is not None and isinstance(message.content, list) else [],
                )
                if is_admin
                else None
            ),
        )

    def _build_summary(
        self,
        session: AgentSession | None,
        tools: list[NetworkDiagnosticsToolItem],
    ) -> NetworkDiagnosticsSummary:
        return NetworkDiagnosticsSummary(
            total_duration_ms=session.total_duration_ms if session else None,
            total_steps=session.total_steps if session else 0,
            total_tool_calls=len(tools),
            search_calls=sum(1 for item in tools if item.tool_name == "web_search"),
            url_read_calls=sum(1 for item in tools if item.tool_name == "url_read"),
            success_count=sum(1 for item in tools if item.status == "success"),
            failed_count=sum(1 for item in tools if item.status == "failed"),
            degraded_count=sum(1 for item in tools if item.status == "degraded"),
            interrupted_count=sum(1 for item in tools if item.status == "interrupted"),
            limit_reason=session.limit_reason if session else None,
            run_status=session.status if session else None,
        )

    def _tool_item_from_log(
        self,
        log: ToolCallLog,
        *,
        is_admin: bool,
    ) -> NetworkDiagnosticsToolItem:
        input_params = log.input_params or {}
        output_data = log.output_data or {}
        admin: dict[str, Any] | None = None
        if is_admin:
            admin = {
                "trace_id": log.trace_id,
                "step_number": log.step_number,
                "input_params": self._sanitize_input_params(input_params),
                "created_at": log.created_at.isoformat() if log.created_at else None,
            }
            observation = build_itinerary_tool_observation_from_values(
                tool_name=log.tool_name,
                status=log.status,
                duration_ms=log.duration_ms,
                output_data=output_data,
            )
            if observation is not None:
                admin.update(
                    outcome=observation.outcome,
                    error_category=observation.error_category,
                    controlled_repair=observation.controlled_repair,
                    travel_budget_exhausted=observation.travel_budget_exhausted,
                    server_budget_exhausted=observation.server_budget_exhausted,
                    upstream_error=observation.upstream_error,
                )
            else:
                admin["error_message"] = log.error_message

        return NetworkDiagnosticsToolItem(
            tool_call_log_id=log.id,
            tool_name=log.tool_name,
            status=self._normalize_status(log.status),
            duration_ms=log.duration_ms,
            target=self._derive_target(log.tool_name, input_params),
            result_count=self._derive_result_count(log.tool_name, output_data),
            reason=self._derive_reason(log.tool_name, log.status, log.error_message, input_params, output_data),
            requested_count=self._derive_requested_count(log.tool_name, input_params, output_data),
            actual_count=self._derive_actual_count(log.tool_name, output_data),
            context_count=self._derive_context_count(log.tool_name, output_data),
            intent=self._derive_intent(log.tool_name, input_params, output_data),
            domains=self._derive_domains(log.tool_name, input_params, output_data),
            recency_days=self._derive_recency_days(log.tool_name, input_params, output_data),
            budget_limited=bool(output_data.get("budget_limited", False)),
            started_at=log.created_at,
            admin=admin,
        )

    def _build_itinerary_admin_summary(
        self,
        session: AgentSession | None,
        tools: list[NetworkDiagnosticsToolItem],
        content_blocks: list[Any],
    ) -> dict[str, Any] | None:
        observations = [
            ItineraryToolObservation(
                tool_name=item.tool_name,
                outcome=item.admin["outcome"],
                error_category=item.admin.get("error_category"),
                duration_ms=item.duration_ms,
                controlled_repair=item.admin.get("controlled_repair") is True,
                travel_budget_exhausted=item.admin.get("travel_budget_exhausted") is True,
                server_budget_exhausted=item.admin.get("server_budget_exhausted") is True,
                upstream_error=item.admin.get("upstream_error") is True,
            )
            for item in tools
            if item.admin is not None and item.admin.get("outcome") in {"success", "degraded", "failed", "timeout"}
        ]
        if session is None and not observations:
            return None
        payload = build_itinerary_run_payload(
            run_id=session.id if session is not None else "unknown",
            model_id=session.model_id if session is not None else "unknown",
            provider=session.provider if session is not None else "unknown",
            content_blocks=content_blocks,
            tool_observations=observations,
            run_status=session.status if session is not None else "error",
            finish_reason="unknown",
            limit_reason=session.limit_reason if session is not None else None,
            total_duration_ms=session.total_duration_ms if session is not None else 0,
            total_steps=session.total_steps if session is not None else 0,
            total_tool_calls=session.total_tool_calls if session is not None else len(observations),
        )
        if payload is None:
            return None
        return {
            key: value
            for key, value in payload.items()
            if key not in {"event", "schema_version", "run_id", "finish_reason"}
        }

    def _normalize_status(self, status: str) -> ToolStatus:
        if status in ("success", "failed", "degraded", "interrupted"):
            return status
        return "failed"

    def _derive_target(self, tool_name: str, input_params: dict[str, Any]) -> str:
        if tool_name == "web_search":
            return str(input_params.get("query") or "").strip()
        if tool_name == "url_read":
            return str(input_params.get("url") or "").strip()
        return tool_name

    def _derive_result_count(self, tool_name: str, output_data: dict[str, Any]) -> int | None:
        if "result_count" in output_data and isinstance(output_data["result_count"], int):
            return output_data["result_count"]
        sources = output_data.get("sources")
        if tool_name == "web_search" and isinstance(sources, list):
            return len(sources)
        return None

    def _derive_reason(
        self,
        tool_name: str,
        status: str,
        error_message: str | None,
        input_params: dict[str, Any],
        output_data: dict[str, Any],
    ) -> str | None:
        if error_message and error_message.strip():
            return error_message.strip()
        if tool_name == "url_read" and status == "success":
            input_reason = input_params.get("reason")
            if isinstance(input_reason, str) and input_reason.strip():
                return input_reason.strip()
            output_reason = output_data.get("reason")
            if isinstance(output_reason, str) and output_reason.strip():
                return output_reason.strip()
        if status == "degraded":
            return "部分内容不可用，已降级处理"
        if status == "failed":
            return "未取得可用内容"
        if status == "interrupted":
            return "工具调用已中断"
        return None

    def _derive_requested_count(
        self,
        tool_name: str,
        input_params: dict[str, Any],
        output_data: dict[str, Any],
    ) -> int | None:
        if tool_name != "web_search":
            return None
        value = output_data.get("requested_count", input_params.get("count"))
        return value if isinstance(value, int) else None

    def _derive_actual_count(self, tool_name: str, output_data: dict[str, Any]) -> int | None:
        if tool_name != "web_search":
            return None
        value = output_data.get("actual_count", output_data.get("result_count"))
        return value if isinstance(value, int) else None

    def _derive_context_count(self, tool_name: str, output_data: dict[str, Any]) -> int | None:
        if tool_name != "web_search":
            return None
        value = output_data.get("context_source_count", output_data.get("context_count"))
        return value if isinstance(value, int) else None

    def _derive_intent(
        self,
        tool_name: str,
        input_params: dict[str, Any],
        output_data: dict[str, Any],
    ) -> str | None:
        if tool_name != "web_search":
            return None
        value = output_data.get("intent", input_params.get("intent"))
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def _derive_domains(
        self,
        tool_name: str,
        input_params: dict[str, Any],
        output_data: dict[str, Any],
    ) -> list[str]:
        if tool_name != "web_search":
            return []
        value = output_data.get("domains", input_params.get("domains"))
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    def _derive_recency_days(
        self,
        tool_name: str,
        input_params: dict[str, Any],
        output_data: dict[str, Any],
    ) -> int | None:
        if tool_name != "web_search":
            return None
        value = output_data.get("recency_days", input_params.get("recency_days"))
        return value if isinstance(value, int) else None

    def _sanitize_input_params(self, input_params: dict[str, Any]) -> dict[str, Any]:
        allowed_keys = {"query", "url"}
        return {key: value for key, value in input_params.items() if key in allowed_keys}
