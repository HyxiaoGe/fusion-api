"""管理员审计中心独立服务层。"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import ValidationError

from app.db.admin_audit_repository import AdminAuditRepository, page_payload
from app.db.models import AdminAuditEvent, AgentStep, File, Message, PerformanceRun, ToolCallLog, User
from app.schemas.admin_audit import AdminPerformanceRunImport, PerformanceSafeSummary
from app.schemas.response import ApiException, ErrorCode
from app.services.admin_audit_sanitizer import mask_email, sanitize_admin_value


class AdminAuditService:
    def __init__(self, repository: AdminAuditRepository):
        self.repository = repository

    def _record(
        self,
        *,
        admin: User,
        action: str,
        resource_type: str,
        request_id: str,
        resource_id: str | None = None,
        target_user_id: str | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        audit_metadata = {}
        for key, value in (metadata or {}).items():
            if key in {"q", "query"}:
                text = str(value or "")
                audit_metadata[key] = {"present": bool(text), "length": len(text)}
            else:
                audit_metadata[key] = value
        safe_metadata, _ = sanitize_admin_value(audit_metadata, max_string_chars=300, max_list_items=50)
        safe_snapshot, _ = sanitize_admin_value(
            {
                "id": str(admin.id),
                "username": getattr(admin, "username", None),
                "email_masked": mask_email(getattr(admin, "email", None)),
            },
            max_string_chars=300,
        )
        safe_reason, _ = sanitize_admin_value((reason or "").strip()[:300], max_string_chars=300)
        try:
            self.repository.create_audit_event(
                id=str(uuid.uuid4()),
                admin_user_id=str(admin.id),
                admin_snapshot=safe_snapshot,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                target_user_id=target_user_id,
                request_id=request_id,
                reason=safe_reason or None,
                extra_metadata=safe_metadata,
            )
        except Exception as exc:
            self.repository.db.rollback()
            raise ApiException.service_unavailable(
                "管理员访问审计暂时不可用",
                code=ErrorCode.INTERNAL_ERROR,
            ) from exc

    @staticmethod
    def _user_summary(user: User | None) -> dict[str, Any] | None:
        if user is None:
            return None
        return {
            "id": user.id,
            "username": user.username,
            "nickname": user.nickname,
            "email_masked": mask_email(user.email),
        }

    def _user_item(self, row: dict[str, Any], *, include_sensitive: bool = False) -> dict[str, Any]:
        user = row["user"]
        item = {
            "id": user.id,
            "username": user.username,
            "nickname": user.nickname,
            "email_masked": mask_email(user.email),
            "is_superuser": bool(user.is_superuser),
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "last_active_at": row["last_active_at"],
            "conversation_count": row["conversation_count"],
            "message_count": row["message_count"],
            "tool_call_count": row["tool_call_count"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
        }
        if include_sensitive:
            system_prompt, _ = sanitize_admin_value(user.system_prompt or "")
            item.update({"email": user.email, "system_prompt": system_prompt})
        return item

    def list_users(self, *, admin: User, request_id: str, reason: str | None, **filters: Any) -> dict[str, Any]:
        rows, total = self.repository.list_users(**filters)
        items = [self._user_item(row) for row in rows]
        self._record(
            admin=admin,
            action="admin.audit.users.list",
            resource_type="user",
            request_id=request_id,
            reason=reason,
            metadata={key: value for key, value in filters.items() if value is not None},
        )
        return page_payload(
            items,
            total,
            filters["page"],
            filters["page_size"],
        )

    def get_user(self, user_id: str, *, admin: User, request_id: str, reason: str | None) -> dict[str, Any]:
        row = self.repository.get_user(user_id)
        if row is None:
            raise ApiException.not_found("用户不存在")
        item = self._user_item(row, include_sensitive=True)
        self._record(
            admin=admin,
            action="admin.audit.user.view",
            resource_type="user",
            resource_id=user_id,
            target_user_id=user_id,
            request_id=request_id,
            reason=reason,
        )
        return item

    def _conversation_item(self, row: dict[str, Any]) -> dict[str, Any]:
        conversation = row["conversation"]
        return {
            "id": conversation.id,
            "title": conversation.title,
            "model_id": conversation.model_id,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "user": self._user_summary(row["user"]),
            "message_count": row["message_count"],
            "tool_call_count": row["tool_call_count"],
            "file_count": row["file_count"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "latest_agent_status": row["latest_agent_status"],
        }

    def list_conversations(
        self,
        *,
        admin: User,
        request_id: str,
        reason: str | None,
        **filters: Any,
    ) -> dict[str, Any]:
        rows, total = self.repository.list_conversations(**filters)
        items = [self._conversation_item(row) for row in rows]
        self._record(
            admin=admin,
            action="admin.audit.conversations.list",
            resource_type="conversation",
            target_user_id=filters.get("user_id"),
            request_id=request_id,
            reason=reason,
            metadata={key: value for key, value in filters.items() if value is not None},
        )
        return page_payload(
            items,
            total,
            filters["page"],
            filters["page_size"],
        )

    def get_conversation(
        self,
        conversation_id: str,
        *,
        admin: User,
        request_id: str,
        reason: str | None,
    ) -> dict[str, Any]:
        row = self.repository.get_conversation(conversation_id)
        if row is None:
            raise ApiException.not_found("对话不存在")
        item = self._conversation_item(row)
        self._record(
            admin=admin,
            action="admin.audit.conversation.view",
            resource_type="conversation",
            resource_id=conversation_id,
            target_user_id=row["conversation"].user_id,
            request_id=request_id,
            reason=reason,
        )
        return item

    def _assert_conversation(self, conversation_id: str) -> str:
        target_user_id = self.repository.conversation_target_user_id(conversation_id)
        if target_user_id is None:
            raise ApiException.not_found("对话不存在")
        return target_user_id

    @staticmethod
    def _message_item(message: Message) -> dict[str, Any]:
        content, _ = sanitize_admin_value(message.content or [])
        usage, _ = sanitize_admin_value(message.usage) if message.usage else (None, [])
        questions, _ = sanitize_admin_value(message.suggested_questions) if message.suggested_questions else (None, [])
        return {
            "id": message.id,
            "role": message.role,
            "content": content,
            "model_id": message.model_id,
            "usage": usage,
            "suggested_questions": questions,
            "created_at": message.created_at,
        }

    def list_messages(
        self,
        conversation_id: str,
        *,
        page: int,
        page_size: int,
        admin: User,
        request_id: str,
        reason: str | None,
    ) -> dict[str, Any]:
        target_user_id = self._assert_conversation(conversation_id)
        rows, total = self.repository.list_messages(conversation_id, page=page, page_size=page_size)
        items = [self._message_item(row) for row in rows]
        self._record(
            admin=admin,
            action="admin.audit.messages.list",
            resource_type="conversation_messages",
            resource_id=conversation_id,
            target_user_id=target_user_id,
            request_id=request_id,
            reason=reason,
            metadata={"page": page, "page_size": page_size},
        )
        return page_payload(items, total, page, page_size)

    @staticmethod
    def _tool_item(tool: ToolCallLog) -> dict[str, Any]:
        raw_arguments = tool.input_params if isinstance(tool.input_params, dict) else {}
        raw_output = tool.output_data if isinstance(tool.output_data, dict) else {}
        if tool.tool_name == "web_search":
            argument_projection = {
                key: raw_arguments[key]
                for key in ("query", "count", "domains", "recency_days", "intent")
                if key in raw_arguments
            }
            output_projection = {
                key: raw_output[key]
                for key in (
                    "result_count",
                    "requested_count",
                    "actual_count",
                    "context_source_count",
                    "context_source_limit",
                    "requested_provider",
                    "result_provider",
                    "fallback_used",
                    "provider_chain",
                    "budget_limited",
                )
                if key in raw_output
            }
            sources = raw_output.get("sources")
            if isinstance(sources, list):
                output_projection["sources"] = [
                    {
                        key: source[key]
                        for key in ("title", "url", "favicon", "status")
                        if isinstance(source, dict) and key in source
                    }
                    for source in sources[:20]
                    if isinstance(source, dict)
                ]
        elif tool.tool_name == "url_read":
            argument_projection = {key: raw_arguments[key] for key in ("url", "reason") if key in raw_arguments}
            output_projection = {
                key: raw_output[key]
                for key in (
                    "url",
                    "safe_log_url",
                    "title",
                    "status",
                    "content_length",
                    "length",
                    "reason",
                    "requested_provider",
                    "result_provider",
                )
                if key in raw_output
            }
        else:
            argument_projection = {}
            output_projection = {}
        arguments, input_fields = sanitize_admin_value(argument_projection, max_string_chars=1000, max_list_items=30)
        result, output_fields = sanitize_admin_value(output_projection, max_string_chars=1000, max_list_items=30)
        error, error_fields = sanitize_admin_value(tool.error_message or "", max_string_chars=1000)
        redacted = sorted(
            [f"arguments.{field}" for field in input_fields]
            + [f"result_preview.{field}" for field in output_fields]
            + [f"error.{field}" for field in error_fields]
        )
        if raw_arguments and not argument_projection:
            redacted.append("arguments")
        if raw_output and not output_projection:
            redacted.append("result_preview")
        return {
            "id": tool.id,
            "message_id": tool.message_id,
            "trace_id": tool.trace_id,
            "step_number": tool.step_number,
            "tool_name": tool.tool_name,
            "status": tool.status,
            "duration_ms": tool.duration_ms,
            "model_id": tool.model_id,
            "provider": tool.provider,
            "arguments": arguments,
            "result_preview": result,
            "error": error or None,
            "redacted_fields": redacted,
            "created_at": tool.created_at,
        }

    def list_tool_calls(
        self,
        conversation_id: str,
        *,
        page: int,
        page_size: int,
        admin: User,
        request_id: str,
        reason: str | None,
    ) -> dict[str, Any]:
        target_user_id = self._assert_conversation(conversation_id)
        rows, total = self.repository.list_tool_calls(conversation_id, page=page, page_size=page_size)
        items = [self._tool_item(row) for row in rows]
        self._record(
            admin=admin,
            action="admin.audit.tool_calls.list",
            resource_type="conversation_tool_calls",
            resource_id=conversation_id,
            target_user_id=target_user_id,
            request_id=request_id,
            reason=reason,
            metadata={"page": page, "page_size": page_size},
        )
        return page_payload(items, total, page, page_size)

    @staticmethod
    def _step_item(step: AgentStep, tools: list[ToolCallLog]) -> dict[str, Any]:
        names, _ = sanitize_admin_value(step.tool_names or [])
        return {
            "id": step.id,
            "step_number": step.step_number,
            "status": step.status,
            "tool_calls_count": step.tool_calls_count or 0,
            "tool_names": names,
            "duration_ms": step.duration_ms,
            "created_at": step.created_at,
            "tool_calls": [AdminAuditService._tool_item(tool) for tool in tools],
        }

    def _run_item(self, row: dict[str, Any]) -> dict[str, Any]:
        session = row["session"]
        raw_config = session.run_config if isinstance(session.run_config, dict) else {}
        config_projection = {
            key: raw_config[key]
            for key in ("max_steps", "max_tool_calls", "timeout_s", "runtime_config_versions")
            if key in raw_config
        }
        config, _ = sanitize_admin_value(config_projection, max_string_chars=1000, max_list_items=30)
        error, _ = sanitize_admin_value(session.error_message or "", max_string_chars=1000)
        progress = self._progress_projection(row["snapshot"].state) if row["snapshot"] else None
        tools_by_step: dict[int | None, list[ToolCallLog]] = {}
        for tool in row["tool_calls"]:
            tools_by_step.setdefault(tool.step_number, []).append(tool)
        return {
            "id": session.id,
            "message_id": session.message_id,
            "user_id": session.user_id,
            "model_id": session.model_id,
            "provider": session.provider,
            "config": config,
            "total_steps": session.total_steps or 0,
            "total_tool_calls": session.total_tool_calls or 0,
            "total_duration_ms": session.total_duration_ms,
            "status": session.status,
            "limit_reason": session.limit_reason,
            "error": error or None,
            "created_at": session.created_at,
            "progress": progress,
            "steps": [self._step_item(step, tools_by_step.get(step.step_number, [])) for step in row["steps"]],
        }

    @staticmethod
    def _progress_projection(raw_state: Any) -> dict[str, Any] | None:
        if not isinstance(raw_state, dict):
            return None

        def pick(source: Any, keys: tuple[str, ...]) -> dict[str, Any] | None:
            if not isinstance(source, dict):
                return None
            return {key: source[key] for key in keys if key in source}

        progress = pick(
            raw_state.get("progress"),
            ("phase", "label", "completed_steps", "total_steps", "completed_tool_calls", "max_tool_calls"),
        )
        raw_plan = raw_state.get("plan")
        plan = pick(raw_plan, ("plan_id", "revision"))
        if plan is not None and isinstance(raw_plan.get("items"), list):
            plan["items"] = [
                pick(
                    item,
                    ("id", "title", "status", "kind", "summary", "tool_names", "evidence_item_ids"),
                )
                for item in raw_plan["items"][:50]
                if isinstance(item, dict)
            ]
        raw_tool_digests = raw_state.get("tool_digests")
        if not isinstance(raw_tool_digests, list):
            raw_tool_digests = []
        tool_digests = [
            pick(
                item,
                ("tool_call_id", "tool_name", "status", "title", "summary", "key_findings", "source_refs", "truncated"),
            )
            for item in raw_tool_digests[:50]
            if isinstance(item, dict)
        ]
        raw_evidence = raw_state.get("evidence")
        if not isinstance(raw_evidence, list):
            raw_evidence = []
        evidence = [
            pick(
                item,
                ("id", "kind", "status", "title", "url", "domain", "claim", "snippet", "used_by_final_answer"),
            )
            for item in raw_evidence[:50]
            if isinstance(item, dict)
        ]
        projection = {
            key: raw_state[key] for key in ("run_id", "message_id", "status", "updated_at") if key in raw_state
        }
        projection.update(
            {
                "progress": progress,
                "plan": plan,
                "tool_digests": tool_digests,
                "evidence": evidence,
            }
        )
        sanitized, _ = sanitize_admin_value(projection, max_string_chars=1000, max_list_items=50)
        return sanitized

    def list_agent_runs(
        self,
        conversation_id: str,
        *,
        page: int,
        page_size: int,
        admin: User,
        request_id: str,
        reason: str | None,
    ) -> dict[str, Any]:
        target_user_id = self._assert_conversation(conversation_id)
        rows, total = self.repository.list_agent_runs(conversation_id, page=page, page_size=page_size)
        items = [self._run_item(row) for row in rows]
        self._record(
            admin=admin,
            action="admin.audit.agent_runs.list",
            resource_type="conversation_agent_runs",
            resource_id=conversation_id,
            target_user_id=target_user_id,
            request_id=request_id,
            reason=reason,
            metadata={"page": page, "page_size": page_size},
        )
        return page_payload(items, total, page, page_size)

    @staticmethod
    def _file_item(file: File) -> dict[str, Any]:
        name, _ = sanitize_admin_value(file.original_filename, max_string_chars=500)
        return {
            "id": file.id,
            "original_filename": name,
            "mimetype": file.mimetype,
            "size": file.size,
            "status": file.status,
            "width": file.width,
            "height": file.height,
            "created_at": file.created_at,
        }

    def list_files(
        self,
        conversation_id: str,
        *,
        page: int,
        page_size: int,
        admin: User,
        request_id: str,
        reason: str | None,
    ) -> dict[str, Any]:
        target_user_id = self._assert_conversation(conversation_id)
        rows, total = self.repository.list_files(conversation_id, page=page, page_size=page_size)
        items = [self._file_item(row) for row in rows]
        self._record(
            admin=admin,
            action="admin.audit.files.list",
            resource_type="conversation_files",
            resource_id=conversation_id,
            target_user_id=target_user_id,
            request_id=request_id,
            reason=reason,
            metadata={"page": page, "page_size": page_size},
        )
        return page_payload(items, total, page, page_size)

    @staticmethod
    def _audit_event_item(event: AdminAuditEvent) -> dict[str, Any]:
        metadata, _ = sanitize_admin_value(event.extra_metadata or {})
        snapshot, _ = sanitize_admin_value(event.admin_snapshot or {})
        reason, _ = sanitize_admin_value(event.reason or "", max_string_chars=300)
        return {
            "id": event.id,
            "admin_user_id": event.admin_user_id,
            "admin_snapshot": snapshot,
            "action": event.action,
            "resource_type": event.resource_type,
            "resource_id": event.resource_id,
            "target_user_id": event.target_user_id,
            "request_id": event.request_id,
            "reason": reason or None,
            "metadata": metadata,
            "created_at": event.created_at,
        }

    def list_audit_events(
        self,
        *,
        admin: User,
        request_id: str,
        reason: str | None,
        **filters: Any,
    ) -> dict[str, Any]:
        rows, total = self.repository.list_audit_events(**filters)
        items = [self._audit_event_item(row) for row in rows]
        self._record(
            admin=admin,
            action="admin.audit.events.list",
            resource_type="admin_audit_event",
            target_user_id=filters.get("target_user_id"),
            request_id=request_id,
            reason=reason,
            metadata={key: value for key, value in filters.items() if value is not None},
        )
        return page_payload(
            items,
            total,
            filters["page"],
            filters["page_size"],
        )

    @staticmethod
    def _safe_performance_summary(value: Any) -> dict[str, Any]:
        try:
            summary_source = PerformanceSafeSummary.model_validate(value).model_dump(exclude_none=True)
        except ValidationError:
            summary_source = PerformanceSafeSummary(
                stopped=True,
                stop_reasons=["invalid_safe_summary"],
            ).model_dump(exclude_none=True)
        summary, _ = sanitize_admin_value(summary_source, max_string_chars=2000, max_list_items=100)
        return summary

    @classmethod
    def _performance_item(cls, run: PerformanceRun) -> dict[str, Any]:
        return {
            "run_id": run.run_id,
            "environment": run.environment,
            "model_id": run.model_id,
            "status": run.status,
            "schema_version": run.schema_version,
            "safe_summary": cls._safe_performance_summary(run.safe_summary),
            "imported_by_user_id": run.imported_by_user_id,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "created_at": run.created_at,
        }

    @staticmethod
    def _performance_list_item(run: Any) -> dict[str, Any]:
        return {
            "run_id": run.run_id,
            "environment": run.environment,
            "model_id": run.model_id,
            "status": run.status,
            "schema_version": run.schema_version,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "created_at": run.created_at,
        }

    def import_performance_run(
        self,
        payload: AdminPerformanceRunImport,
        *,
        admin: User,
        request_id: str,
        reason: str | None,
    ) -> dict[str, Any]:
        summary_source = payload.safe_summary.model_dump(exclude_none=True)
        safe_summary, _ = sanitize_admin_value(summary_source, max_string_chars=2000, max_list_items=100)
        values = payload.model_dump(exclude={"safe_summary"})
        values.update({"safe_summary": safe_summary, "imported_by_user_id": str(admin.id)})
        run, created = self.repository.import_performance_run(values)
        self._record(
            admin=admin,
            action="admin.audit.performance_run.import",
            resource_type="performance_run",
            resource_id=payload.run_id,
            request_id=request_id,
            reason=reason,
            metadata={"environment": payload.environment, "schema_version": payload.schema_version},
        )
        return {"run_id": run.run_id, "created": created}

    def list_performance_runs(
        self,
        *,
        admin: User,
        request_id: str,
        reason: str | None,
        **filters: Any,
    ) -> dict[str, Any]:
        rows, total = self.repository.list_performance_runs(**filters)
        items = [self._performance_list_item(row) for row in rows]
        self._record(
            admin=admin,
            action="admin.audit.performance_runs.list",
            resource_type="performance_run",
            request_id=request_id,
            reason=reason,
            metadata={key: value for key, value in filters.items() if value is not None},
        )
        return page_payload(
            items,
            total,
            filters["page"],
            filters["page_size"],
        )

    def get_performance_run(
        self,
        run_id: str,
        *,
        admin: User,
        request_id: str,
        reason: str | None,
    ) -> dict[str, Any]:
        run = self.repository.get_performance_run(run_id)
        if run is None:
            raise ApiException.not_found("压测记录不存在")
        item = self._performance_item(run)
        self._record(
            admin=admin,
            action="admin.audit.performance_run.view",
            resource_type="performance_run",
            resource_id=run_id,
            request_id=request_id,
            reason=reason,
        )
        return item
