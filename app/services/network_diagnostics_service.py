from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.models import AgentSession, ToolCallLog
from app.schemas.network_diagnostics import (
    NetworkDiagnosticsResponse,
    NetworkDiagnosticsSummary,
    NetworkDiagnosticsToolItem,
    ToolStatus,
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
                "error_message": log.error_message,
                "created_at": log.created_at.isoformat() if log.created_at else None,
            }

        return NetworkDiagnosticsToolItem(
            tool_call_log_id=log.id,
            tool_name=log.tool_name,
            status=self._normalize_status(log.status),
            duration_ms=log.duration_ms,
            target=self._derive_target(log.tool_name, input_params),
            result_count=self._derive_result_count(log.tool_name, output_data),
            reason=self._derive_reason(log.status, log.error_message),
            started_at=log.created_at,
            admin=admin,
        )

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

    def _derive_reason(self, status: str, error_message: str | None) -> str | None:
        if error_message and error_message.strip():
            return error_message.strip()
        if status == "degraded":
            return "部分内容不可用，已降级处理"
        if status == "failed":
            return "未取得可用内容"
        if status == "interrupted":
            return "工具调用已中断"
        return None

    def _sanitize_input_params(self, input_params: dict[str, Any]) -> dict[str, Any]:
        allowed_keys = {"query", "url"}
        return {key: value for key, value in input_params.items() if key in allowed_keys}
