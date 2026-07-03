"""Agent run 触顶后的 continuation 上下文构造。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter
from sqlalchemy.orm import Session

from app.ai.prompts.agent_loop import (
    CONTINUATION_SYSTEM_PROMPT as _CONTINUATION_SYSTEM_PROMPT,
)
from app.ai.prompts.agent_loop import get_continuation_system_prompt
from app.db.models import AgentSession
from app.db.models import Message as MessageModel
from app.schemas.chat import ContentBlock
from app.schemas.response import ApiException
from app.services.stream.agent_loop_policy import AgentLoopLimits

_CONTENT_BLOCKS_ADAPTER = TypeAdapter(list[ContentBlock])
CONTINUATION_SYSTEM_PROMPT = _CONTINUATION_SYSTEM_PROMPT


@dataclass(frozen=True)
class AgentContinuationContext:
    assistant_message: MessageModel
    previous_session: AgentSession
    limits: AgentLoopLimits
    initial_content_blocks: list[ContentBlock]


def deserialize_content_blocks(raw_blocks: list[dict[str, Any]] | None) -> list[ContentBlock]:
    return _CONTENT_BLOCKS_ADAPTER.validate_python(raw_blocks or [])


def inject_continuation_prompt(messages: list[dict]) -> list[dict]:
    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    prompt = {"role": "system", "content": get_continuation_system_prompt()}
    return [*messages[:insert_at], prompt, *messages[insert_at:]]


def resolve_continuation_limits(session: AgentSession, *, default_limits: AgentLoopLimits) -> AgentLoopLimits:
    config = session.run_config if isinstance(session.run_config, dict) else {}
    try:
        return AgentLoopLimits(
            max_steps=int(config.get("max_steps", default_limits.max_steps)),
            max_tool_calls=int(config.get("max_tool_calls", default_limits.max_tool_calls)),
            total_timeout_s=float(config.get("timeout_s", default_limits.total_timeout_s)),
        )
    except (TypeError, ValueError):
        return default_limits


def find_latest_limit_reached_session(
    db: Session,
    *,
    conversation_id: str,
    message_id: str,
    previous_run_id: str | None = None,
) -> AgentSession:
    query = db.query(AgentSession).filter(
        AgentSession.conversation_id == conversation_id,
        AgentSession.message_id == message_id,
    )
    session = query.order_by(AgentSession.created_at.desc()).first()
    if session is None or session.status != "limit_reached":
        raise ApiException.bad_request("这条回答当前不能继续执行")
    if previous_run_id and session.id != previous_run_id:
        raise ApiException.bad_request("这条回答当前不能继续执行")
    return session


def build_continuation_context(
    db: Session,
    *,
    conversation_id: str,
    message_id: str,
    previous_run_id: str | None,
    default_limits: AgentLoopLimits,
) -> AgentContinuationContext:
    assistant_message = (
        db.query(MessageModel)
        .filter(
            MessageModel.id == message_id,
            MessageModel.conversation_id == conversation_id,
            MessageModel.role == "assistant",
        )
        .first()
    )
    if assistant_message is None:
        raise ApiException.not_found("会话消息不存在或无权访问")

    previous_session = find_latest_limit_reached_session(
        db,
        conversation_id=conversation_id,
        message_id=message_id,
        previous_run_id=previous_run_id,
    )

    return AgentContinuationContext(
        assistant_message=assistant_message,
        previous_session=previous_session,
        limits=resolve_continuation_limits(previous_session, default_limits=default_limits),
        initial_content_blocks=deserialize_content_blocks(assistant_message.content),
    )
