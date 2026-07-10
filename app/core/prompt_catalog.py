"""Fusion 与 PromptHub 之间的固定 Prompt 映射契约。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    key: str
    slug: str
    name: str
    variables: tuple[str, ...]
    marker: str


PROMPT_SPECS = (
    PromptSpec("app_identity", "app-identity", "Fusion 应用身份", (), "【Fusion 身份一致性规则】"),
    PromptSpec("tool_usage_contract", "tool-usage-contract", "工具调用一致性规则", (), "【工具调用一致性规则】"),
    PromptSpec("no_tool_network_boundary", "no-tool-network-boundary", "无联网工具边界", (), "【无联网工具边界规则】"),
    PromptSpec(
        "no_vision_file_boundary", "no-vision-file-boundary", "无图片理解能力边界", (), "【无图片理解能力边界规则】"
    ),
    PromptSpec("url_read_tool_description", "url-read-tool-description", "URL 读取工具说明", (), "读取指定 URL"),
    PromptSpec("limit_summary", "limit-summary", "工具上限总结", (), "工具调用上限"),
    PromptSpec("continuation_system", "continuation-system", "续写系统规则", (), "继续上一轮"),
    PromptSpec("generate_title", "generate-title", "生成会话标题", ("content",), "对话内容："),
    PromptSpec(
        "generate_suggested_questions",
        "generate-suggested-questions",
        "生成推荐问题",
        ("content",),
        "三个推荐问题",
    ),
    PromptSpec("file_analysis", "file-analysis", "文件分析", ("query", "file_content"), "问题:"),
    PromptSpec(
        "file_content_enhancement",
        "file-content-enhancement",
        "文件内容增强",
        ("query", "file_content"),
        "参考以下文件内容:",
    ),
)

PROMPT_SPEC_BY_KEY = {spec.key: spec for spec in PROMPT_SPECS}
PROMPT_SPEC_BY_SLUG = {spec.slug: spec for spec in PROMPT_SPECS}
