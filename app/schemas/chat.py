# app/schemas/chat.py
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

# ============================================================
# Content Blocks（消息内容块）
# ============================================================


class TextBlock(BaseModel):
    """纯文本内容块"""

    type: Literal["text"]
    id: str = Field(default_factory=lambda: f"blk_{uuid4().hex[:12]}")
    text: str


class ThinkingBlock(BaseModel):
    """模型推理过程内容块，仅出现在 assistant 消息中"""

    type: Literal["thinking"]
    id: str = Field(default_factory=lambda: f"blk_{uuid4().hex[:12]}")
    thinking: str


class FileBlock(BaseModel):
    """文件引用内容块，仅出现在 user 消息中"""

    type: Literal["file"]
    id: str = Field(default_factory=lambda: f"blk_{uuid4().hex[:12]}")
    file_id: str
    filename: str
    mime_type: str
    thumbnail_url: Optional[str] = None  # 缩略图 URL（presigned 或 API 代理）
    width: Optional[int] = None  # 图片宽度
    height: Optional[int] = None  # 图片高度


class SearchSource(BaseModel):
    """单条搜索来源"""

    title: str
    url: str
    description: str
    content: Optional[str] = None  # 网页正文摘要（Tavily 等 provider 支持）
    favicon: Optional[str] = None  # 网站 favicon URL


class SearchSourceSummary(BaseModel):
    """轻量搜索来源摘要，用于 Message.content 中的 SearchBlock"""

    title: str
    url: str
    favicon: Optional[str] = None


class SearchBlock(BaseModel):
    """搜索结果内容块，出现在 assistant 消息中"""

    type: Literal["search"]
    id: str = Field(default_factory=lambda: f"blk_{uuid4().hex[:12]}")
    query: str
    tool_call_log_id: str = ""  # 关联 tool_call_logs 表
    sources: List[SearchSourceSummary]  # 轻量版，前端展示用


class UrlBlock(BaseModel):
    """网页读取内容块，出现在 assistant 消息中"""

    type: Literal["url_read"]
    id: str = Field(default_factory=lambda: f"blk_{uuid4().hex[:12]}")
    url: str
    title: Optional[str] = None
    favicon: Optional[str] = None
    tool_call_log_id: str = ""


# content block 的联合类型，后续扩展直接在此添加
ContentBlock = Union[TextBlock, ThinkingBlock, FileBlock, SearchBlock, UrlBlock]


# ============================================================
# Usage（Token 消耗）
# ============================================================


class Usage(BaseModel):
    """Token 消耗统计，仅 assistant 消息携带"""

    input_tokens: int = 0
    output_tokens: int = 0


# ============================================================
# Message（消息）
# ============================================================


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Literal["user", "assistant"]
    # content 为 content blocks 数组
    # user 消息示例：[TextBlock, FileBlock]
    # assistant 消息示例：[ThinkingBlock, TextBlock]
    content: List[ContentBlock]
    # 仅 assistant 消息填充，记录实际生成该消息时使用的模型
    model_id: Optional[str] = None
    # 仅 assistant 消息填充
    usage: Optional[Usage] = None
    # 仅 assistant 消息填充，持久化推荐问题
    suggested_questions: Optional[List[str]] = None
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True


# ============================================================
# Conversation（会话）
# ============================================================


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    model_id: str
    title: str
    messages: List[Message] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True


# ============================================================
# Chat API 请求 / 响应
# ============================================================


class ChatRequest(BaseModel):
    model_id: str  # 替换原来的 provider + model 两个字段
    message: str  # 用户输入的文本
    conversation_id: Optional[str] = None
    stream: bool = True  # 默认开启流式
    options: Optional[Dict[str, Any]] = None  # 扩展选项，如 use_reasoning
    file_ids: Optional[List[str]] = None  # 附带的文件 ID 列表


class ChatResponse(BaseModel):
    """非流式响应结构（流式走 SSE，不走这个）"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    message: Message
    created_at: datetime = Field(default_factory=datetime.now)


# ============================================================
# SSE 流式结构（对齐 OpenAI Chat Completions streaming 格式）
# ============================================================


class StreamDelta(BaseModel):
    """
    SSE 每个 chunk 携带的增量内容。
    注意：content blocks 里的 text/thinking 字段携带的是增量内容（delta），
    不是完整内容。同一个 block 的多个 chunk 共享同一个 block id，
    前端按 block id 拼接还原完整内容。
    例：
      chunk1: ThinkingBlock(id="blk_001", thinking="让我")
      chunk2: ThinkingBlock(id="blk_001", thinking="分析一下")
      chunk3: TextBlock(id="blk_002", text="根据")
      chunk4: TextBlock(id="blk_002", text="你的需求")
    """

    content: Optional[List[ContentBlock]] = None


class StreamChoice(BaseModel):
    delta: StreamDelta
    finish_reason: Optional[Literal["stop", "error"]] = None


class StreamChunk(BaseModel):
    """SSE 单个 chunk 的完整结构"""

    id: str  # 与最终落库的 message.id 一致
    conversation_id: str  # 会话 ID，前端据此关联消息
    choices: List[StreamChoice]
    # 仅在 finish_reason=stop 的最后一个 chunk 携带
    usage: Optional[Usage] = None


# ============================================================
# 标题生成 / 推荐问题
# ============================================================


class TitleGenerationRequest(BaseModel):
    conversation_id: str
    options: Optional[Dict[str, Any]] = None


class TitleGenerationResponse(BaseModel):
    title: str
    conversation_id: str


class SuggestedQuestionsRequest(BaseModel):
    conversation_id: str
    options: Optional[Dict[str, Any]] = None


class SuggestedQuestionsResponse(BaseModel):
    questions: List[str]
    conversation_id: str


# ============================================================
# 会话列表 API
# ============================================================


class ConversationSummary(BaseModel):
    """会话列表接口返回的轻量结构，不携带 messages"""

    id: str
    model_id: str
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MessageUpdateRequest(BaseModel):
    """消息更新请求（内部使用）"""

    content: Optional[List[ContentBlock]] = None
    usage: Optional[Usage] = None
