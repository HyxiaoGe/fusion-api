"""管理员审计中心独立服务层。"""

from __future__ import annotations

import math
import re
import uuid
from typing import Any

from pydantic import ValidationError

from app.ai import litellm_catalog, litellm_health
from app.core.logger import app_logger as logger
from app.db.admin_audit_repository import AdminAuditRepository, page_payload
from app.db.models import AdminAuditEvent, AgentStep, File, Message, PerformanceRun, ToolCallLog, User
from app.schemas.admin_audit import (
    AdminAuditAdminSnapshot,
    AdminAuditEventItem,
    AdminAuditEventMetadata,
    AdminAuditSearchSummary,
    AdminModelOperationsItem,
    AdminModelPerformanceSummary,
    AdminPerformanceRunImport,
    PerformanceSafeSummary,
)
from app.schemas.response import ApiException, ErrorCode
from app.services.admin_audit_sanitizer import mask_email, sanitize_admin_value
from app.services.agent_strategy_config import get_agent_tools_disabled_aliases


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
    def _error_projection(error_message: Any, status: Any = None) -> dict[str, str] | None:
        raw_error = error_message if isinstance(error_message, str) else ""
        normalized = raw_error.lower()
        if re.search(
            r"(?<![\w-])(?:401|403)(?![\w-])|\b(?:unauthorized|forbidden|credentials?)\b",
            normalized,
        ):
            return {"type": "authentication_failed", "message": "上游服务认证失败"}
        if re.search(
            r"(?<![\w-])429(?![\w-])|\brate(?:[ _-]+)limit(?:ed|ing)?\b|\btoo many requests\b|频率限制",
            normalized,
        ):
            return {"type": "rate_limited", "message": "上游服务请求过于频繁"}
        if re.search(r"\b(?:timeout|timed out|deadline exceeded)\b|超时", normalized):
            return {"type": "timeout", "message": "上游服务响应超时"}
        if re.search(r"\b(?:cancelled|canceled|interrupted|aborted)\b|中断|取消", normalized):
            return {"type": "cancelled", "message": "执行已中断"}
        if re.search(
            r"(?<![\w-])(?:502|503|504)(?![\w-])|\b(?:connection|unavailable|network)\b|连接",
            normalized,
        ):
            return {"type": "upstream_unavailable", "message": "上游服务暂时不可用"}
        if raw_error or str(status or "").lower() in {"failed", "error"}:
            return {"type": "execution_failed", "message": "执行失败"}
        return None

    @staticmethod
    def _source_projection(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        allowed = ("kind", "title", "url", "favicon", "status", "tool_call_log_id")
        return [
            {key: source[key] for key in allowed if isinstance(source.get(key), str)}
            for source in value[:20]
            if isinstance(source, dict)
        ]

    @staticmethod
    def _string_fields(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, str]:
        return {key: value[key] for key in keys if isinstance(value.get(key), str)}

    @staticmethod
    def _count_fields(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, int]:
        return {
            key: item
            for key in keys
            if isinstance((item := value.get(key)), int) and not isinstance(item, bool) and 0 <= item <= 1_000_000_000
        }

    @staticmethod
    def _bool_fields(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, bool]:
        return {key: value[key] for key in keys if isinstance(value.get(key), bool)}

    @staticmethod
    def _string_list_fields(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for key in keys:
            raw_items = value.get(key)
            if isinstance(raw_items, list):
                result[key] = [item for item in raw_items[:20] if isinstance(item, str)]
        return result

    @classmethod
    def _content_block_projection(cls, raw_block: Any) -> dict[str, Any]:
        if not isinstance(raw_block, dict):
            return {"type": "unknown", "content_hidden": True}
        block_type = raw_block.get("type") if isinstance(raw_block.get("type"), str) else "unknown"
        common = {"type": block_type, **cls._string_fields(raw_block, ("id",))}
        if block_type == "text":
            projection = {**common, **cls._string_fields(raw_block, ("text",))}
        elif block_type == "thinking":
            projection = {**common, **cls._string_fields(raw_block, ("thinking",))}
        elif block_type == "file":
            projection = {
                **common,
                **cls._string_fields(raw_block, ("file_id", "filename", "mime_type", "status")),
                **cls._count_fields(raw_block, ("width", "height")),
            }
        elif block_type == "search":
            projection = {
                **common,
                **cls._string_fields(
                    raw_block,
                    (
                        "query",
                        "tool_call_log_id",
                        "status",
                        "requested_provider",
                        "result_provider",
                        "search_budget",
                        "intent",
                    ),
                ),
                **cls._count_fields(
                    raw_block,
                    (
                        "source_count",
                        "requested_count",
                        "actual_count",
                        "context_source_count",
                        "context_source_limit",
                        "recency_days",
                    ),
                ),
                **cls._bool_fields(raw_block, ("fallback_used", "budget_limited")),
                **cls._string_list_fields(raw_block, ("provider_chain", "domains")),
                "sources": cls._source_projection(raw_block.get("sources")),
                "source_refs": cls._source_projection(raw_block.get("source_refs")),
            }
            error = cls._error_projection(raw_block.get("error_message"), raw_block.get("status"))
            if error:
                projection.update({"error_type": error["type"], "error_message": error["message"]})
        elif block_type == "url_read":
            projection = {
                **common,
                **cls._string_fields(raw_block, ("url", "title", "favicon", "tool_call_log_id", "status")),
                **cls._count_fields(raw_block, ("source_count",)),
                "source_refs": cls._source_projection(raw_block.get("source_refs")),
            }
            error = cls._error_projection(raw_block.get("error_message"), raw_block.get("status"))
            if error:
                projection.update({"error_type": error["type"], "error_message": error["message"]})
        else:
            projection = {
                "type": block_type,
                **cls._string_fields(raw_block, ("id", "status")),
                "content_hidden": True,
            }
        sanitized, _ = sanitize_admin_value(
            projection,
            max_string_chars=4000,
            max_list_items=20,
            max_dict_items=40,
            max_depth=5,
            max_nodes=300,
        )
        return sanitized

    @classmethod
    def _message_content_projection(cls, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [cls._content_block_projection(block) for block in value[:100]]

    @staticmethod
    def _usage_projection(value: Any) -> dict[str, int] | None:
        if not isinstance(value, dict):
            return None
        usage = {
            key: item
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if isinstance((item := value.get(key)), int) and not isinstance(item, bool) and 0 <= item <= 1_000_000_000
        }
        return usage or None

    @staticmethod
    def _questions_projection(value: Any) -> list[str] | None:
        if not isinstance(value, list):
            return None
        questions = [item for item in value[:10] if isinstance(item, str)]
        sanitized, _ = sanitize_admin_value(questions, max_string_chars=500, max_list_items=10, max_nodes=20)
        return sanitized or None

    @classmethod
    def _message_item(cls, message: Message) -> dict[str, Any]:
        return {
            "id": message.id,
            "role": message.role,
            "content": cls._message_content_projection(message.content),
            "model_id": message.model_id,
            "usage": cls._usage_projection(message.usage),
            "suggested_questions": cls._questions_projection(message.suggested_questions),
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

    @classmethod
    def _tool_item(cls, tool: ToolCallLog) -> dict[str, Any]:
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
        error = cls._error_projection(tool.error_message, tool.status)
        redacted = sorted(
            [f"arguments.{field}" for field in input_fields] + [f"result_preview.{field}" for field in output_fields]
        )
        if raw_arguments and not argument_projection:
            redacted.append("arguments")
        if raw_output and not output_projection:
            redacted.append("result_preview")
        if tool.error_message:
            redacted.append("error")
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
            "error": error,
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
        error = self._error_projection(session.error_message, session.status)
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
            "error": error,
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
    def _positive_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _safe_catalog_text(value: Any, *, max_chars: int) -> str:
        sanitized, _ = sanitize_admin_value(value if isinstance(value, str) else "", max_string_chars=max_chars)
        return sanitized[:max_chars]

    @classmethod
    def _catalog_projection(
        cls,
        model_id: str,
        entry: dict[str, Any],
        agent_tools_disabled_aliases: set[str] | list[str] | tuple[str, ...],
    ) -> dict[str, Any]:
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        provider = metadata.get("provider_key")
        if not isinstance(provider, str) or not provider.strip():
            provider = entry.get("litellm_provider")
        if not isinstance(provider, str) or not provider.strip():
            provider = "litellm"
        provider = cls._safe_catalog_text(provider.strip().lower(), max_chars=100)
        if not re.fullmatch(r"[a-z0-9._:-]+", provider):
            provider = "litellm"
        provider_display = cls._safe_catalog_text(metadata.get("provider_display") or provider, max_chars=200)
        capabilities = litellm_catalog.normalize_capabilities(
            model_id,
            metadata.get("capabilities") if isinstance(metadata.get("capabilities"), dict) else {},
            agent_tools_disabled_aliases=agent_tools_disabled_aliases,
        )
        recommended = metadata.get("recommended_for") if isinstance(metadata.get("recommended_for"), list) else []
        health = litellm_health.get_health(model_id)
        health_error = cls._safe_catalog_text(health.get("error"), max_chars=300) or None

        return {
            "name": cls._safe_catalog_text(metadata.get("display_name") or model_id, max_chars=300),
            "catalog_status": "active",
            "catalog_source": "litellm_model_info",
            "provider": provider,
            "provider_display": provider_display,
            "description": cls._safe_catalog_text(metadata.get("description"), max_chars=1000),
            "knowledge_cutoff": cls._safe_catalog_text(metadata.get("knowledge_cutoff"), max_chars=100) or None,
            "context_window_tokens": cls._positive_int(entry.get("max_input_tokens")),
            "max_output_tokens": cls._positive_int(entry.get("max_output_tokens")),
            "capabilities": {
                key: bool(capabilities.get(key, False))
                for key in (
                    "imageGen",
                    "deepThinking",
                    "fileSupport",
                    "functionCalling",
                    "agentTools",
                    "searchCapable",
                    "vision",
                    "webSearch",
                )
            },
            "health": {
                "status": health.get("status") if health.get("status") in {"healthy", "unhealthy"} else "unknown",
                "error": health_error,
                "checked_at": (
                    health.get("checked_at")
                    if isinstance(health.get("checked_at"), (int, float))
                    and math.isfinite(float(health["checked_at"]))
                    and health["checked_at"] >= 0
                    else None
                ),
            },
            "cost_tier": cls._safe_catalog_text(metadata.get("cost_tier"), max_chars=30) or None,
            "recommended_for": [
                cls._safe_catalog_text(value, max_chars=100) for value in recommended[:50] if isinstance(value, str)
            ],
        }

    @classmethod
    def _model_performance_summary(cls, row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        return AdminModelPerformanceSummary(
            run_id=row.run_id,
            environment=row.environment,
            status=row.status,
            started_at=row.started_at,
            finished_at=row.finished_at,
            created_at=row.created_at,
        ).model_dump()

    @classmethod
    def _model_operations_item(
        cls,
        model_id: str,
        catalog_entry: dict[str, Any] | None,
        stats: dict[str, Any] | None,
        agent_tools_disabled_aliases: set[str] | list[str] | tuple[str, ...],
        missing_catalog_status: str = "historical",
        catalog_availability: str = "available",
    ) -> dict[str, Any]:
        usage = stats or {}
        catalog_fields = (
            cls._catalog_projection(model_id, catalog_entry, agent_tools_disabled_aliases)
            if catalog_entry is not None
            else {"name": model_id, "catalog_status": missing_catalog_status}
        )
        item = AdminModelOperationsItem(
            model_id=model_id,
            **catalog_fields,
            catalog_availability=catalog_availability,
            conversation_count=usage.get("conversation_count", 0),
            user_count=usage.get("user_count", 0),
            assistant_message_count=usage.get("assistant_message_count", 0),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            last_used_at=usage.get("last_used_at"),
            agent_run_count=usage.get("agent_run_count", 0),
            agent_error_count=usage.get("agent_error_count", 0),
            latest_performance_run=cls._model_performance_summary(usage.get("latest_performance_run")),
        )
        return item.model_dump()

    def list_models(
        self,
        *,
        admin: User,
        request_id: str,
        reason: str | None,
        page: int,
        page_size: int,
        query: str | None = None,
        catalog_status: str | None = None,
        provider: str | None = None,
        health_status: str | None = None,
    ) -> dict[str, Any]:
        raw_catalog = litellm_catalog.list_aliases()
        catalog_availability = litellm_catalog.get_cache_status().get("availability", "degraded")
        if catalog_availability not in {"available", "degraded"}:
            catalog_availability = "degraded"
        invalid_model_ids: set[Any] = set()
        catalog: dict[str, dict[str, Any]] = {}
        for raw_model_id, entry in raw_catalog.items():
            if not entry.get("db_model"):
                continue
            model_id = self._safe_model_id(raw_model_id)
            if model_id is None:
                invalid_model_ids.add(raw_model_id)
                continue
            catalog[model_id] = entry
        agent_tools_disabled_aliases = get_agent_tools_disabled_aliases() if catalog else set()
        raw_stats = self.repository.list_model_operation_stats()
        stats: dict[str, dict[str, Any]] = {}
        for raw_model_id, model_stats in raw_stats.items():
            model_id = self._safe_model_id(raw_model_id)
            if model_id is None:
                invalid_model_ids.add(raw_model_id)
                continue
            stats[model_id] = model_stats
        if invalid_model_ids:
            logger.warning(f"admin model operations excluded invalid model ids: count={len(invalid_model_ids)}")
        missing_catalog_status = "unknown" if catalog_availability == "degraded" else "historical"
        model_ids = set(catalog) | set(stats)
        items = [
            self._model_operations_item(
                model_id,
                catalog.get(model_id),
                stats.get(model_id),
                agent_tools_disabled_aliases,
                missing_catalog_status,
                catalog_availability,
            )
            for model_id in model_ids
        ]
        if query and query.strip():
            needle = query.strip().casefold()
            items = [
                item
                for item in items
                if needle
                in " ".join(
                    str(item.get(key) or "") for key in ("model_id", "name", "provider", "provider_display")
                ).casefold()
            ]
        if catalog_status:
            items = [item for item in items if item["catalog_status"] == catalog_status]
        if provider:
            normalized_provider = provider.strip().casefold()
            items = [item for item in items if item["provider"] and item["provider"].casefold() == normalized_provider]
        if health_status:
            items = [item for item in items if item["health"]["status"] == health_status]
        status_order = {"active": 0, "unknown": 1, "historical": 2}
        items.sort(key=lambda item: (status_order[item["catalog_status"]], item["model_id"]))
        total = len(items)
        start = (page - 1) * page_size
        self._record(
            admin=admin,
            action="admin.audit.models.list",
            resource_type="model",
            request_id=request_id,
            reason=reason,
            metadata={"page": page, "page_size": page_size, "query": query, "status": catalog_status},
        )
        result = page_payload(items[start : start + page_size], total, page, page_size)
        result["excluded_invalid_model_count"] = len(invalid_model_ids)
        result["catalog_availability"] = catalog_availability
        return result

    def get_model(
        self,
        model_id: str,
        *,
        admin: User,
        request_id: str,
        reason: str | None,
    ) -> dict[str, Any]:
        normalized_model_id = self._safe_model_id(model_id)
        if normalized_model_id is None:
            raise ApiException.bad_request("模型 ID 无效")
        model_id = normalized_model_id
        entry = litellm_catalog.get_model_entry(model_id)
        catalog_availability = litellm_catalog.get_cache_status().get("availability", "degraded")
        if catalog_availability not in {"available", "degraded"}:
            catalog_availability = "degraded"
        catalog_entry = entry if entry and entry.get("db_model") else None
        stats = self.repository.list_model_operation_stats([model_id]).get(model_id)
        if catalog_entry is None and stats is None:
            raise ApiException.not_found("模型不存在")
        agent_tools_disabled_aliases = get_agent_tools_disabled_aliases() if catalog_entry is not None else set()
        missing_catalog_status = "unknown" if catalog_availability == "degraded" else "historical"
        item = self._model_operations_item(
            model_id,
            catalog_entry,
            stats,
            agent_tools_disabled_aliases,
            missing_catalog_status,
            catalog_availability,
        )
        self._record(
            admin=admin,
            action="admin.audit.model.view",
            resource_type="model",
            resource_id=model_id,
            request_id=request_id,
            reason=reason,
        )
        return item

    @staticmethod
    def _safe_model_id(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        if value != normalized or not normalized or len(normalized) > 200 or re.search(r"[\x00-\x1f\x7f]", normalized):
            return None
        return normalized

    @classmethod
    def _audit_event_item(cls, event: AdminAuditEvent, target_user: User | None) -> dict[str, Any]:
        reason, _ = sanitize_admin_value(event.reason or "", max_string_chars=300)
        item = AdminAuditEventItem(
            id=event.id,
            admin_user_id=event.admin_user_id,
            admin_snapshot=cls._admin_snapshot_projection(event.admin_snapshot),
            action=event.action,
            resource_type=event.resource_type,
            resource_id=event.resource_id,
            target_user_id=event.target_user_id,
            target_user=cls._audit_target_user_summary(target_user),
            request_id=event.request_id,
            reason=reason or None,
            metadata=cls._audit_metadata_projection(event.extra_metadata),
            created_at=event.created_at,
        )
        payload = item.model_dump()
        payload["admin_snapshot"] = item.admin_snapshot.model_dump(exclude_none=True)
        payload["metadata"] = item.metadata.model_dump(exclude_none=True)
        return payload

    @staticmethod
    def _admin_snapshot_projection(value: Any) -> AdminAuditAdminSnapshot:
        raw = value if isinstance(value, dict) else {}
        projected: dict[str, Any] = {}
        for key in ("id", "username"):
            if key not in raw:
                continue
            sanitized, _ = sanitize_admin_value(raw[key], max_string_chars=320)
            try:
                candidate = AdminAuditAdminSnapshot.model_validate({key: sanitized})
            except ValidationError:
                continue
            projected[key] = getattr(candidate, key)
        return AdminAuditAdminSnapshot.model_validate(projected)

    @staticmethod
    def _audit_target_user_summary(user: User | None) -> dict[str, Any] | None:
        if user is None:
            return None
        return {
            "id": user.id,
            "username": user.username,
            "nickname": user.nickname,
        }

    @staticmethod
    def _audit_search_summary(value: Any) -> AdminAuditSearchSummary | None:
        if isinstance(value, str):
            return AdminAuditSearchSummary(present=bool(value), length=len(value))
        if not isinstance(value, dict):
            return None
        present = value.get("present")
        length = value.get("length")
        if not isinstance(present, bool) or isinstance(length, bool) or not isinstance(length, int):
            return None
        try:
            return AdminAuditSearchSummary(present=present, length=length)
        except ValidationError:
            return None

    @classmethod
    def _audit_metadata_projection(cls, value: Any) -> AdminAuditEventMetadata:
        raw = value if isinstance(value, dict) else {}
        projected: dict[str, Any] = {}
        search_keys = {"q", "query"}
        allowed_keys = set(AdminAuditEventMetadata.model_fields) - search_keys
        for key in allowed_keys:
            if key not in raw:
                continue
            sanitized, _ = sanitize_admin_value(raw[key], max_string_chars=300)
            try:
                candidate = AdminAuditEventMetadata.model_validate({key: sanitized})
            except ValidationError:
                continue
            projected[key] = getattr(candidate, key)
        for key in search_keys:
            summary = cls._audit_search_summary(raw.get(key))
            if summary is not None:
                projected[key] = summary
        return AdminAuditEventMetadata.model_validate(projected)

    def list_audit_events(
        self,
        *,
        admin: User,
        request_id: str,
        reason: str | None,
        **filters: Any,
    ) -> dict[str, Any]:
        rows, total = self.repository.list_audit_events(**filters)
        target_user_ids = sorted({row.target_user_id for row in rows if row.target_user_id})
        target_users = self.repository.get_users_by_ids(target_user_ids) if target_user_ids else {}
        items = [self._audit_event_item(row, target_users.get(row.target_user_id)) for row in rows]
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
    def _safe_performance_summary(value: Any, schema_version: int) -> dict[str, Any]:
        if schema_version not in {1, 2}:
            stop_reason = "unsupported_schema_version"
        else:
            try:
                summary_source = PerformanceSafeSummary.model_validate(value).model_dump(exclude_none=True)
            except ValidationError:
                stop_reason = "invalid_safe_summary"
            else:
                summary, _ = sanitize_admin_value(summary_source, max_string_chars=2000, max_list_items=100)
                return summary
        summary_source = PerformanceSafeSummary(
            stopped=True,
            stop_reasons=[stop_reason],
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
            "safe_summary": cls._safe_performance_summary(run.safe_summary, run.schema_version),
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
