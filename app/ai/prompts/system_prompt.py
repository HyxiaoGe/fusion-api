"""按可信段落身份组装系统提示词；仅处理内存数据，不访问服务或外部 IO。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.ai.prompts.agent_loop import build_current_date_system_prompt, get_app_identity_prompt
from app.utils.prompt_fingerprint import fingerprint_system_messages

TEMPLATE_VERSION = "2026-08-27.2"


@dataclass(frozen=True)
class SystemPromptSection:
    section_id: str
    content: str


@dataclass(frozen=True)
class SystemPromptAssembly:
    messages: list[dict]
    metadata: dict[str, Any]


class SystemPromptAssemblyError(Exception):
    """仅向调用方提供固定安全错误信息和组装元数据。"""

    def __init__(self, metadata: dict[str, Any]):
        super().__init__("系统提示词组装失败")
        self.metadata = metadata


def build_stable_base_sections() -> list[SystemPromptSection]:
    return [SystemPromptSection("app_identity", get_app_identity_prompt())]


def build_dynamic_sections(
    user_system_prompt: str | None = None,
    *,
    include_current_date: bool = True,
) -> list[SystemPromptSection]:
    sections = [SystemPromptSection("current_date", build_current_date_system_prompt())] if include_current_date else []
    if user_system_prompt and user_system_prompt.strip():
        sections.append(
            SystemPromptSection(
                "user_preferences",
                "以下是用户的个性化偏好设置，请在回答中自然遵守，但不要主动提及这些设置本身：\n\n"
                + user_system_prompt.strip(),
            )
        )
    return sections


def build_base_sections(
    user_system_prompt: str | None = None,
    *,
    include_current_date: bool = True,
) -> list[SystemPromptSection]:
    return [
        *build_stable_base_sections(),
        *build_dynamic_sections(
            user_system_prompt,
            include_current_date=include_current_date,
        ),
    ]


def assemble_system_prompt(
    *,
    user_system_prompt: str | None = None,
    include_current_date: bool = True,
    sections: Callable[[], Iterable[SystemPromptSection]] | None = None,
) -> SystemPromptAssembly:
    """回调只允许读取已准备的内存状态，段落内容不参与身份判定。"""
    started = perf_counter()
    section_ids: list[str] = []
    metadata: dict[str, Any] = {
        "status": "failed",
        "source": "code",
        "template_version": TEMPLATE_VERSION,
        "section_ids": section_ids,
        "fingerprint": None,
        "char_count": None,
    }
    try:
        selected = build_stable_base_sections()
        if sections is not None:
            selected.extend(sections())
        selected.extend(
            build_dynamic_sections(
                user_system_prompt,
                include_current_date=include_current_date,
            )
        )
        messages = []
        for section in selected:
            if section.section_id in section_ids:
                continue
            section_ids.append(section.section_id)
            messages.append({"role": "system", "content": section.content})
        metadata.update(
            status="ready",
            fingerprint=fingerprint_system_messages(messages),
            char_count=sum(len(message["content"]) for message in messages),
            duration_ms=max(0, int((perf_counter() - started) * 1000)),
        )
        return SystemPromptAssembly(messages=messages, metadata=metadata)
    except Exception:
        metadata.update(
            duration_ms=max(0, int((perf_counter() - started) * 1000)),
            error_code="assembly_failed",
            message="系统提示词组装失败",
        )
        raise SystemPromptAssemblyError(metadata) from None
