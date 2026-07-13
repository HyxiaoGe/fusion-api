"""消息落库 + URL 路径 A 预处理。

spec §4.4。两个独立功能放一起的理由：都是 runner 之外的"副作用胶水"，
跟 runner 的"控制流"职责正交。
"""

import asyncio
import copy
import re
import uuid
from typing import Optional

from sqlalchemy import text

from app.ai.tools import build_url_read_tool
from app.core.config import settings
from app.core.logger import app_logger as logger
from app.schemas.chat import UrlBlock, Usage
from app.services.security.url_policy import evaluate_url_policy
from app.services.source_context import UntrustedSourceContext, format_untrusted_source_context

URL_PATTERN = re.compile(r'https?://[^\s<>"\')\]]+')


def acquire_message_persistence_lock(db, assistant_message_id: str) -> None:
    """PostgreSQL 下按 assistant message 获取事务级 advisory lock。"""
    bind = db.get_bind()
    dialect_name = getattr(getattr(bind, "dialect", None), "name", None)
    if dialect_name != "postgresql":
        return
    db.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
        {"key": f"assistant_message:{assistant_message_id}"},
    )


def _serialize_content_block(block) -> dict:
    serialized = block.model_dump() if hasattr(block, "model_dump") else block
    return copy.deepcopy(serialized)


def _merge_same_id_block(existing: dict, incoming: dict) -> dict:
    """合并同 ID block；流式文本按前缀关系保留更完整的一侧。"""
    block_type = incoming.get("type")
    if existing.get("type") != block_type or block_type not in {"text", "thinking"}:
        return incoming

    content_field = "text" if block_type == "text" else "thinking"
    existing_text = existing.get(content_field)
    incoming_text = incoming.get(content_field)
    if not isinstance(existing_text, str) or not isinstance(incoming_text, str):
        return incoming
    if incoming_text.startswith(existing_text):
        return incoming
    if existing_text.startswith(incoming_text):
        return existing
    # 分叉时以客户端当前可见流为准，避免后台旧分支覆盖用户所见。
    return incoming


def merge_partial_content_blocks(existing_content: list, incoming_content: list) -> list[dict]:
    """按 block ID 合并 partial，并保留两侧独有的内容块。"""
    merged = [_serialize_content_block(block) for block in existing_content]
    positions = {
        block.get("id"): index for index, block in enumerate(merged) if isinstance(block, dict) and block.get("id")
    }

    for raw_block in incoming_content:
        incoming = _serialize_content_block(raw_block)
        block_id = incoming.get("id") if isinstance(incoming, dict) else None
        if not block_id or block_id not in positions:
            positions[block_id] = len(merged)
            merged.append(incoming)
            continue
        position = positions[block_id]
        merged[position] = _merge_same_id_block(merged[position], incoming)

    return merged


def persist_message(
    db,
    assistant_message_id: str,
    conversation_id: str,
    model_id: str,
    content_blocks: list,
    usage_data: Optional[Usage] = None,
    partial: bool = False,
) -> None:
    """
    将 assistant 消息写入 PostgreSQL。
    partial=True 时增量更新 content blocks（checkpoint）；若传入 usage，
    同步保存已产生的累计 Token 与最后上下文快照，供失败/中止后恢复。
    partial=False 时写入完整数据（最终落库）。
    """
    try:
        from app.db.models import Message as MessageModel

        acquire_message_persistence_lock(db, assistant_message_id)
        existing = db.query(MessageModel).populate_existing().filter_by(id=assistant_message_id).first()
        serialized_content = [block.model_dump() for block in content_blocks]
        if existing:
            existing.content = (
                merge_partial_content_blocks(existing.content or [], serialized_content)
                if partial
                else serialized_content
            )
            if usage_data:
                existing.usage = usage_data.model_dump()
        else:
            db_message = MessageModel(
                id=assistant_message_id,
                conversation_id=conversation_id,
                role="assistant",
                content=serialized_content,
                model_id=model_id,
                usage=usage_data.model_dump() if usage_data else None,
            )
            db.add(db_message)
        db.commit()
    except Exception as e:
        logger.error(f"写入 assistant 消息失败: {e}")
        db.rollback()


def extract_first_url(message: str) -> str | None:
    urls_in_message = URL_PATTERN.findall(message)
    return urls_in_message[0] if urls_in_message else None


def ensure_url_read_tool(call_kwargs: dict) -> None:
    tools = call_kwargs.setdefault("tools", [])
    if not any(_tool_name(tool) == "url_read" for tool in tools):
        tools.append(build_url_read_tool())


def _tool_name(tool: dict) -> str | None:
    function = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    return None


def resolve_reader_url(policy, detected_url: str) -> str:
    return policy.normalized_url or detected_url


async def read_url_for_context(*, policy, detected_url: str):
    from app.services.external.reader_client import read_url

    timeout = settings.READER_SERVICE_TIMEOUT
    try:
        return await asyncio.wait_for(
            read_url(resolve_reader_url(policy, detected_url), timeout=timeout),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"URL 自动抓取超时: url={policy.safe_log_url}")
        return None


def build_url_context_message(*, read_result, policy, detected_url: str) -> dict:
    from app.services.tool_handlers.url_read import MAX_CONTENT_CHARS

    return {
        "role": "user",
        "content": format_untrusted_source_context(
            UntrustedSourceContext(
                source_id="U1",
                source_type="url_read",
                title=read_result.title or "未知",
                url=read_result.url or resolve_reader_url(policy, detected_url),
                content=read_result.content,
                provider="web",
            ),
            max_chars=MAX_CONTENT_CHARS,
        ),
    }


def build_url_read_block(*, read_result, policy, detected_url: str, block_id: str) -> UrlBlock:
    return UrlBlock(
        type="url_read",
        id=block_id,
        url=read_result.url or resolve_reader_url(policy, detected_url),
        title=read_result.title,
        favicon=read_result.favicon,
    )


def remove_disabled_thinking(call_kwargs: dict) -> None:
    if "extra_body" in call_kwargs and call_kwargs["extra_body"].get("thinking", {}).get("type") == "disabled":
        del call_kwargs["extra_body"]


def fallback_to_url_read_tool(call_kwargs: dict, detected_url: str | None = None):
    ensure_url_read_tool(call_kwargs)
    return None, None, detected_url


def build_successful_url_preprocess_result(
    *,
    read_result,
    policy,
    detected_url: str,
    block_id: str,
    call_kwargs: dict,
):
    remove_disabled_thinking(call_kwargs)
    return (
        build_url_read_block(
            read_result=read_result,
            policy=policy,
            detected_url=detected_url,
            block_id=block_id,
        ),
        build_url_context_message(
            read_result=read_result,
            policy=policy,
            detected_url=detected_url,
        ),
        detected_url,
    )


async def preprocess_url_in_message(
    original_message: str,
    supports_fc: bool,
    call_kwargs: dict,
) -> tuple[Optional[UrlBlock], Optional[dict], Optional[str]]:
    """URL 路径 A：自动读取首个 URL，成功时注入不可信上下文，失败时交给 url_read 工具。"""
    if not supports_fc:
        return None, None, None

    # 消息中无 URL：仍然把 URL_READ_TOOL 加入 tools，让 LLM 自决
    auto_detected_url = extract_first_url(original_message)
    if not auto_detected_url:
        return fallback_to_url_read_tool(call_kwargs)

    url_read_block_id = f"blk_{uuid.uuid4().hex[:12]}"
    policy = evaluate_url_policy(auto_detected_url)
    if not policy.allowed:
        logger.info(f"URL 自动抓取被策略拒绝: reason={policy.reason}, url={policy.safe_log_url}")
        return fallback_to_url_read_tool(call_kwargs, auto_detected_url)

    read_result = await read_url_for_context(policy=policy, detected_url=auto_detected_url)
    if not read_result:
        # 抓取失败 → 追加 URL_READ_TOOL，让 LLM 自决
        return fallback_to_url_read_tool(call_kwargs, auto_detected_url)

    # 抓取成功 → 注入 user role 不可信上下文 + 关闭 volcengine thinking + 返回 UrlBlock
    return build_successful_url_preprocess_result(
        read_result=read_result,
        policy=policy,
        detected_url=auto_detected_url,
        block_id=url_read_block_id,
        call_kwargs=call_kwargs,
    )
