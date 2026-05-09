"""消息落库 + URL 路径 A 预处理。

spec §4.4。两个独立功能放一起的理由：都是 runner 之外的"副作用胶水"，
跟 runner 的"控制流"职责正交。
"""

import asyncio
import re
import uuid
from typing import Optional

from app.core.logger import app_logger as logger
from app.schemas.chat import UrlBlock, Usage


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
    partial=True 时增量更新 content blocks（checkpoint），不写 usage。
    partial=False 时写入完整数据（最终落库）。
    """
    try:
        from app.db.models import Message as MessageModel

        existing = db.query(MessageModel).filter_by(id=assistant_message_id).first()
        if existing:
            existing.content = [block.model_dump() for block in content_blocks]
            if usage_data and not partial:
                existing.usage = usage_data.model_dump()
        else:
            db_message = MessageModel(
                id=assistant_message_id,
                conversation_id=conversation_id,
                role="assistant",
                content=[block.model_dump() for block in content_blocks],
                model_id=model_id,
                usage=usage_data.model_dump() if usage_data and not partial else None,
            )
            db.add(db_message)
        db.commit()
    except Exception as e:
        logger.error(f"写入 assistant 消息失败: {e}")
        db.rollback()


async def preprocess_url_in_message(
    original_message: str,
    supports_fc: bool,
    call_kwargs: dict,
) -> tuple[Optional[UrlBlock], Optional[dict], Optional[str]]:
    """URL 路径 A：在 agent loop 之前自动抓取消息中的第一个 URL，注入 system 消息。

    spec §4.4 的 inout 语义：
    - 抓取成功：修改 call_kwargs 删 extra_body（对 volcengine 关闭 thinking）；
      返回 (UrlBlock, system_msg, url) 让调用方 append/insert
    - 抓取失败 / 消息无 URL：往 call_kwargs["tools"] 追加 URL_READ_TOOL，让 LLM 自决
    - supports_fc=False：完全跳过，返回三个 None

    返回 (url_read_block, url_context_msg, auto_detected_url):
      url_read_block: 成功时返回的 UrlBlock，调用方 append 到 content_blocks
      url_context_msg: 成功时返回的 system 消息，调用方插入 messages 倒数第二位
      auto_detected_url: 检测到的第一个 URL（仅日志/调试，None 表示未检测到）
    """
    if not supports_fc:
        return None, None, None

    url_pattern = re.compile(r'https?://[^\s<>"\')\]]+')
    urls_in_message = url_pattern.findall(original_message)

    # 消息中无 URL：仍然把 URL_READ_TOOL 加入 tools，让 LLM 自决
    if not urls_in_message:
        from app.ai.tools import URL_READ_TOOL

        if URL_READ_TOOL not in call_kwargs.get("tools", []):
            call_kwargs.setdefault("tools", []).append(URL_READ_TOOL)
        return None, None, None

    auto_detected_url = urls_in_message[0]
    url_read_block_id = f"blk_{uuid.uuid4().hex[:12]}"

    try:
        from app.services.reader_client import read_url

        read_result = await asyncio.wait_for(read_url(auto_detected_url, timeout=8.0), timeout=8.0)
    except asyncio.TimeoutError:
        logger.warning(f"URL 自动抓取超时: {auto_detected_url}")
        read_result = None

    if not read_result:
        # 抓取失败 → 追加 URL_READ_TOOL，让 LLM 自决
        from app.ai.tools import URL_READ_TOOL

        if URL_READ_TOOL not in call_kwargs.get("tools", []):
            call_kwargs.setdefault("tools", []).append(URL_READ_TOOL)
        return None, None, auto_detected_url

    # 抓取成功 → 注入 system 消息 + 关闭 volcengine thinking + 返回 UrlBlock
    from app.services.tool_handlers.url_read import MAX_CONTENT_CHARS

    content_text = read_result.content
    truncation_note = ""
    if len(content_text) > MAX_CONTENT_CHARS:
        content_text = content_text[:MAX_CONTENT_CHARS]
        truncation_note = "\n（内容已截断，仅展示前部分）"
    url_context_msg = {
        "role": "system",
        "content": (
            f"以下是用户消息中提到的网页 {auto_detected_url} 的内容：\n"
            f"标题：{read_result.title or '未知'}\n\n"
            f"{content_text}{truncation_note}\n\n"
            "请基于以上网页内容回答用户的问题。"
        ),
    }

    if (
        "extra_body" in call_kwargs
        and call_kwargs["extra_body"].get("thinking", {}).get("type") == "disabled"
    ):
        del call_kwargs["extra_body"]

    url_read_block = UrlBlock(
        type="url_read",
        id=url_read_block_id,
        url=auto_detected_url,
        title=read_result.title,
        favicon=read_result.favicon,
    )

    return url_read_block, url_context_msg, auto_detected_url
