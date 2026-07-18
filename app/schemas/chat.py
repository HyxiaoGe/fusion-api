# app/schemas/chat.py
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import parse_qs, urlsplit
from uuid import RFC_4122, UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.utils.time import utc_now

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
    requested_provider: Optional[str] = None
    result_provider: Optional[str] = None
    fallback_used: bool = False
    provider_chain: List[str] = Field(default_factory=list)


class SearchSourceSummary(BaseModel):
    """轻量搜索来源摘要，用于 Message.content 中的 SearchBlock"""

    title: str
    url: str
    favicon: Optional[str] = None


class SourceReference(BaseModel):
    """统一来源引用摘要，用于搜索/URL 读取等联网 content block"""

    kind: Literal["search", "url_read"]
    title: str = ""
    url: str = ""
    favicon: Optional[str] = None
    status: Literal["success", "failed", "degraded"] = "success"
    tool_call_log_id: str = ""
    error_message: Optional[str] = None


class SearchBlock(BaseModel):
    """搜索结果内容块，出现在 assistant 消息中"""

    type: Literal["search"]
    id: str = Field(default_factory=lambda: f"blk_{uuid4().hex[:12]}")
    query: str
    tool_call_log_id: str = ""  # 关联 tool_call_logs 表
    sources: List[SearchSourceSummary]  # 轻量版，前端展示用
    status: Literal["success", "failed", "degraded"] = "success"
    error_message: Optional[str] = None
    source_count: int = 0
    source_refs: List[SourceReference] = Field(default_factory=list)
    requested_provider: Optional[str] = None
    result_provider: Optional[str] = None
    fallback_used: bool = False
    provider_chain: List[str] = Field(default_factory=list)
    requested_count: Optional[int] = None
    actual_count: Optional[int] = None
    context_source_count: Optional[int] = None
    context_source_limit: Optional[int] = None
    search_budget: Optional[str] = None
    intent: Optional[str] = None
    domains: List[str] = Field(default_factory=list)
    recency_days: Optional[int] = None
    budget_limited: bool = False


class UrlBlock(BaseModel):
    """网页读取内容块，出现在 assistant 消息中"""

    type: Literal["url_read"]
    id: str = Field(default_factory=lambda: f"blk_{uuid4().hex[:12]}")
    url: str
    title: Optional[str] = None
    favicon: Optional[str] = None
    tool_call_log_id: str = ""
    status: Literal["success", "failed", "degraded"] = "success"
    error_message: Optional[str] = None
    source_count: int = 0
    source_refs: List[SourceReference] = Field(default_factory=list)
    reason: Optional[str] = None


class PlacePhoto(BaseModel):
    """地点图片；只接受高德官方 HTTPS 域名，避免把上游任意 URL 带给前端。"""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(max_length=2048)
    title: Optional[str] = Field(default=None, max_length=120)

    @field_validator("url")
    @classmethod
    def validate_official_photo_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        hostname = (parsed.hostname or "").lower()
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in {None, 443}
            or parsed.fragment
            or not _is_amap_official_hostname(hostname)
        ):
            raise ValueError("地点图片必须使用高德官方 HTTPS 地址")
        return value


class PlaceResult(BaseModel):
    """与供应商无关的单个地点结果。"""

    model_config = ConfigDict(extra="forbid")

    provider_place_id: Optional[str] = Field(default=None, max_length=160)
    name: str = Field(min_length=1, max_length=120)
    address: Optional[str] = Field(default=None, max_length=240)
    district: Optional[str] = Field(default=None, max_length=120)
    category: Optional[str] = Field(default=None, max_length=160)
    distance_m: Optional[int] = Field(default=None, ge=0, le=10_000_000)
    photos: List[PlacePhoto] = Field(default_factory=list, max_length=1)
    rating: Optional[float] = Field(default=None, ge=0, le=5)
    reference_cost_yuan: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    platform_url: Optional[str] = Field(default=None, max_length=2048)
    business_area: Optional[str] = Field(default=None, max_length=120)
    open_hours: Optional[str] = Field(default=None, max_length=240)
    detail_status: Literal["enriched", "unavailable", "budget_limited", "not_requested"] = "not_requested"

    @field_validator("platform_url")
    @classmethod
    def validate_platform_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        parsed = urlsplit(value)
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
        if (
            parsed.scheme != "https"
            or (parsed.hostname or "").lower() != "uri.amap.com"
            or parsed.path != "/marker"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in {None, 443}
            or parsed.fragment
            or set(query) != {"poiid", "src", "callnative"}
            or any(len(items) != 1 for items in query.values())
            or not query["poiid"][0]
            or query["src"] != ["fusion"]
            or query["callnative"] != ["0"]
        ):
            raise ValueError("地点跳转地址不符合高德官方 URI 契约")
        return value


class PlaceResultsBlock(BaseModel):
    """地点搜索产品结果块。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["place_results"]
    id: str = Field(default_factory=lambda: f"blk_{uuid4().hex[:12]}", max_length=160)
    schema_version: Literal[1]
    provider: str = Field(min_length=1, max_length=40)
    query: str = Field(min_length=1, max_length=80)
    near: Optional[str] = Field(default=None, max_length=120)
    status: Literal["success", "degraded"]
    result_count: int = Field(ge=0, le=5)
    places: List[PlaceResult] = Field(default_factory=list, max_length=5)
    limitations: List[str] = Field(default_factory=list, max_length=8)
    tool_call_log_id: str = Field(default="", max_length=160)

    @model_validator(mode="after")
    def validate_result_count(self):
        if self.result_count != len(self.places):
            raise ValueError("result_count 必须等于 places 数量")
        return self

    @field_validator("limitations")
    @classmethod
    def validate_limitations(cls, value: List[str]) -> List[str]:
        if any(not item.strip() or len(item) > 240 for item in value):
            raise ValueError("limitations 单项不能为空且不能超过 240 字符")
        return value


class RouteEndpoint(BaseModel):
    """路线端点的安全展示字段，不包含坐标。"""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=120)
    city: Optional[str] = Field(default=None, max_length=40)


class TransitLeg(BaseModel):
    """公共交通单段安全展示字段；所有字段可选以兼容不完整上游结果。"""

    model_config = ConfigDict(extra="forbid")

    kind: Optional[Literal["walking", "subway", "bus", "other"]] = None
    line_name: Optional[str] = Field(default=None, max_length=120)
    departure_stop: Optional[str] = Field(default=None, max_length=120)
    arrival_stop: Optional[str] = Field(default=None, max_length=120)
    via_stop_count: Optional[int] = Field(default=None, ge=0, le=100)
    distance_m: Optional[int] = Field(default=None, ge=0, le=100_000_000)
    duration_s: Optional[int] = Field(default=None, ge=0, le=10_000_000)
    entrance: Optional[str] = Field(default=None, max_length=80)
    exit: Optional[str] = Field(default=None, max_length=80)


class TransitAlternative(BaseModel):
    """公共交通备选方案；不递归嵌套 alternatives。"""

    model_config = ConfigDict(extra="forbid")

    transit_type: Optional[Literal["subway", "bus", "mixed", "public_transit"]] = None
    duration_s: Optional[int] = Field(default=None, ge=0, le=10_000_000)
    walking_distance_m: Optional[int] = Field(default=None, ge=0, le=100_000_000)
    transfers: Optional[int] = Field(default=None, ge=0, le=100)
    summary: Optional[str] = Field(default=None, max_length=160)
    legs: List[TransitLeg] = Field(default_factory=list, max_length=8)


class RouteOption(BaseModel):
    """单种出行方式的路线摘要。"""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["driving", "transit", "walking", "bicycling"]
    distance_m: Optional[int] = Field(default=None, ge=0, le=100_000_000)
    duration_s: Optional[int] = Field(default=None, ge=0, le=10_000_000)
    summary: Optional[str] = Field(default=None, max_length=160)
    toll_yuan: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    transfers: Optional[int] = Field(default=None, ge=0, le=100)
    transit_type: Optional[Literal["subway", "bus", "mixed", "public_transit"]] = None
    walking_distance_m: Optional[int] = Field(default=None, ge=0, le=100_000_000)
    legs: List[TransitLeg] = Field(default_factory=list, max_length=8)
    alternatives: List[TransitAlternative] = Field(default_factory=list, max_length=2)


class RouteResultsBlock(BaseModel):
    """路线对比产品结果块。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["route_results"]
    id: str = Field(default_factory=lambda: f"blk_{uuid4().hex[:12]}", max_length=160)
    schema_version: Literal[1]
    provider: str = Field(min_length=1, max_length=40)
    status: Literal["success", "degraded"]
    origin: RouteEndpoint
    destination: RouteEndpoint
    routes: List[RouteOption] = Field(default_factory=list, min_length=1, max_length=3)
    unavailable_modes: List[Literal["driving", "transit", "walking", "bicycling"]] = Field(
        default_factory=list,
        max_length=3,
    )
    limitations: List[str] = Field(default_factory=list, max_length=8)
    tool_call_log_id: str = Field(default="", max_length=160)

    @field_validator("limitations")
    @classmethod
    def validate_limitations(cls, value: List[str]) -> List[str]:
        if any(not item.strip() or len(item) > 240 for item in value):
            raise ValueError("limitations 单项不能为空且不能超过 240 字符")
        return value


def _is_amap_official_hostname(hostname: str) -> bool:
    return hostname in {"amap.com", "autonavi.com"} or hostname.endswith((".amap.com", ".autonavi.com"))


ProductResultBlock = Union[PlaceResultsBlock, RouteResultsBlock]


# content block 的联合类型，后续扩展直接在此添加
ContentBlock = Union[
    TextBlock,
    ThinkingBlock,
    FileBlock,
    SearchBlock,
    UrlBlock,
    PlaceResultsBlock,
    RouteResultsBlock,
]


# ============================================================
# Usage（Token 消耗）
# ============================================================


ContextStatus = Literal[
    "bypass_unknown_window",
    "no_op_fast_path",
    "no_op",
    "trimmed",
    "trimmed_required_above_target",
    "required_context_over_budget",
    "estimator_unavailable",
]


class ContextUsage(BaseModel):
    """最后一次 LLM 调用的安全上下文状态，不包含消息正文或内部来源。"""

    status: ContextStatus
    round_index: Optional[int] = Field(default=None, ge=1)
    window_tokens: Optional[int] = Field(default=None, ge=0)
    estimated_tokens_before: Optional[int] = Field(default=None, ge=0)
    estimated_tokens_after: Optional[int] = Field(default=None, ge=0)
    actual_prompt_tokens: Optional[int] = Field(default=None, ge=0)
    removed_turns: int = Field(default=0, ge=0)
    removed_messages: int = Field(default=0, ge=0)
    removed_tool_transactions: int = Field(default=0, ge=0)


class Usage(BaseModel):
    """Token 消耗统计，仅 assistant 消息携带"""

    input_tokens: int = 0
    output_tokens: int = 0
    context: Optional[ContextUsage] = None

    @field_validator("context", mode="before")
    @classmethod
    def discard_invalid_context(cls, value):
        """坏的新增字段不能拖垮旧会话详情；保留原有 token usage。"""
        if value is None or isinstance(value, ContextUsage):
            return value
        try:
            return ContextUsage.model_validate(value)
        except (ValidationError, TypeError, ValueError):
            return None


# ============================================================
# Agent Run（消息级最新运行摘要）
# ============================================================


class AgentRunSummary(BaseModel):
    """assistant 消息最近一次 agent run 摘要，用于前端恢复终态和继续执行入口"""

    run_id: str
    status: Literal["running", "completed", "limit_reached", "incomplete", "interrupted", "error"]
    config: Dict[str, Any] = Field(default_factory=dict)
    total_steps: int = 0
    total_tool_calls: int = 0
    limit_reason: Optional[Literal["max_steps", "max_tool_calls", "timeout"]] = None
    progress: Optional[Dict[str, Any]] = None


# ============================================================
# Message（消息）
# ============================================================


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    sequence: Optional[int] = None
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
    # 仅 assistant 消息填充，最近一次 agent run 摘要
    agent_run: Optional[AgentRunSummary] = None
    created_at: datetime = Field(default_factory=utc_now)

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
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Config:
        from_attributes = True


# ============================================================
# Chat API 请求 / 响应
# ============================================================


class ChatRequest(BaseModel):
    model_id: str  # 替换原来的 provider + model 两个字段
    message: str  # 用户输入的文本
    conversation_id: Optional[str] = None
    user_message_id: Optional[str] = None
    assistant_message_id: Optional[str] = None
    stream: bool = True  # 默认开启流式
    options: Optional[Dict[str, Any]] = None  # 扩展选项，如 use_reasoning
    file_ids: Optional[List[str]] = None  # 附带的文件 ID 列表

    @field_validator("user_message_id", "assistant_message_id")
    @classmethod
    def validate_message_uuid4(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        try:
            parsed = UUID(value)
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("消息 ID 必须是合法 UUIDv4") from error
        if parsed.version != 4 or parsed.variant != RFC_4122 or str(parsed) != value.lower():
            raise ValueError("消息 ID 必须是合法 UUIDv4")
        return value

    @model_validator(mode="after")
    def validate_distinct_message_ids(self) -> "ChatRequest":
        if (
            self.user_message_id is not None
            and self.assistant_message_id is not None
            and self.user_message_id.lower() == self.assistant_message_id.lower()
        ):
            raise ValueError("user_message_id 与 assistant_message_id 必须不同")
        return self


class ContinueAgentRunRequest(BaseModel):
    previous_run_id: Optional[str] = None
    stream: bool = True


class GeolocationContextPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accuracy_m: float = Field(ge=0, le=50_000)
    acquired_at: float = Field(ge=0)


class AgentContextResultRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_type: Literal["geolocation"]
    status: Literal["provided", "denied", "timeout", "unavailable"]
    location: Optional[GeolocationContextPayload] = None
    reason: Optional[str] = Field(default=None, min_length=1, max_length=120)

    @model_validator(mode="after")
    def validate_status_payload(self) -> "AgentContextResultRequest":
        if self.status == "provided":
            if self.location is None or self.reason is not None:
                raise ValueError("provided 必须且只能携带 location")
            return self
        if self.location is not None or self.reason is None:
            raise ValueError("非 provided 必须且只能携带 reason")
        return self


class StopStreamRequest(BaseModel):
    """停止流前由客户端提交的当前可见助手内容；空数组兼容旧客户端。"""

    partial_content: List[ContentBlock] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """非流式响应结构（流式走 SSE，不走这个）"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    message: Message
    created_at: datetime = Field(default_factory=utc_now)


# ============================================================
# SSE 流式结构（对齐 OpenAI Chat Completions streaming 格式）
# ============================================================


class StreamDelta(BaseModel):
    """[DEPRECATED] SSE 输出已切到 {chunk_type, data} envelope（Task 8）。
    本类待 Task 11/12 FE cut-over 后删除，期间禁止新代码引用。
    """

    content: Optional[List[ContentBlock]] = None


class StreamChoice(BaseModel):
    """[DEPRECATED] 同 StreamDelta，待 Task 11/12 后删除。"""

    delta: StreamDelta
    finish_reason: Optional[Literal["stop", "error"]] = None


class StreamChunk(BaseModel):
    """[DEPRECATED] 同 StreamDelta，待 Task 11/12 后删除。"""

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
