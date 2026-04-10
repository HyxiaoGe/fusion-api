"""
LLM 消息构建模块

将 Message content blocks 转换为 LiteLLM 所需的 dict 格式，
支持多模态图片 base64 注入。

独立为模块，避免 chat_service ↔ stream_handler 循环导入。
"""

import base64
from typing import Dict, List, Optional

from app.core.logger import app_logger as logger
from app.db.repositories import FileRepository
from app.services.file_service import is_image_mime
from app.services.storage import get_storage

# 历史消息中保留图片的最大轮数（避免 token 爆炸）
MAX_VISION_HISTORY_TURNS = 3


async def file_block_to_image_part(block, file_repo: FileRepository) -> Optional[dict]:
    """将图片 FileBlock 转为 LiteLLM image_url content part"""
    try:
        file_record = file_repo.get_file_by_id(block.file_id)
        if not file_record or not file_record.storage_key:
            return None

        storage = get_storage()
        image_data = await storage.download(file_record.storage_key)
        b64 = base64.b64encode(image_data).decode()
        mime = file_record.mimetype or "image/jpeg"

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{b64}",
            },
        }
    except Exception as e:
        logger.warning(f"图片 base64 注入失败 (file_id={block.file_id}): {e}")
        return None


async def build_llm_messages(
    messages,
    has_vision: bool = False,
    file_repo: FileRepository = None,
) -> List[dict]:
    """
    将 content blocks 消息列表转为 LLM 可消费的 dict 格式。

    - thinking / search block 不传给 LLM（避免污染上下文）
    - 当 has_vision=True 时，图片 FileBlock 转为 base64 image_url 内容块
    - 历史消息中的图片仅保留最近 MAX_VISION_HISTORY_TURNS 轮
    """
    result = []

    # 计算最近 N 轮用户消息的起始索引，用于控制图片注入范围
    vision_cutoff_idx = 0
    if has_vision:
        user_msg_count = 0
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                user_msg_count += 1
                if user_msg_count >= MAX_VISION_HISTORY_TURNS:
                    vision_cutoff_idx = i
                    break

    for idx, msg in enumerate(messages):
        content_parts = []
        has_image = False
        # 仅在最近几轮中注入图片 base64
        inject_images = has_vision and idx >= vision_cutoff_idx

        for block in msg.content:
            if block.type == "text":
                if block.text:
                    content_parts.append(
                        {
                            "type": "text",
                            "text": block.text,
                        }
                    )

            elif block.type == "file" and inject_images:
                # 图片 FileBlock → base64 image_url
                mime = getattr(block, "mime_type", "")
                if is_image_mime(mime):
                    image_part = await file_block_to_image_part(block, file_repo)
                    if image_part:
                        content_parts.append(image_part)
                        has_image = True

            # thinking / search block 不传给 LLM

        if not content_parts:
            continue

        # 无图片时退化为纯文本（节省 token 开销）
        if not has_image and len(content_parts) == 1 and content_parts[0]["type"] == "text":
            result.append(
                {
                    "role": msg.role,
                    "content": content_parts[0]["text"],
                }
            )
        else:
            result.append(
                {
                    "role": msg.role,
                    "content": content_parts,
                }
            )

    return result


def inject_file_content(
    messages: List[dict],
    original_message: str,
    file_contents: Dict[str, str],
) -> List[dict]:
    """将非图片文件的解析内容注入到最后一条用户消息的文本中"""
    if not messages:
        return messages

    combined = "\n\n".join(f"文件内容 ({i + 1}):\n{content}" for i, content in enumerate(file_contents.values()))
    enhanced = f"{original_message}\n\n以下是相关文件内容，请结合这些内容回答：\n{combined}"

    result = messages.copy()
    result[-1] = {"role": "user", "content": enhanced}
    return result


def is_image_file(file_id: str, file_repo: FileRepository) -> bool:
    """判断 file_id 对应的文件是否为图片"""
    file_record = file_repo.get_file_by_id(file_id)
    if not file_record:
        return False
    return is_image_mime(file_record.mimetype or "")
