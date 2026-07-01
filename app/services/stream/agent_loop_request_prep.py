"""Agent loop 请求进入 driver 前的输入准备。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.ai.litellm_utils import merge_extra_body
from app.ai.prompts.agent_loop import TOOL_USAGE_CONTRACT_PROMPT
from app.ai.tools import build_web_search_tool
from app.db.repositories import FileRepository
from app.services.chat.message_builder import (
    build_llm_messages,
    inject_file_content,
    is_image_file,
)
from app.services.stream.persistence import preprocess_url_in_message

VOLCENGINE_PROVIDERS = {"volcengine"}


@dataclass(frozen=True)
class AgentLoopCallConfig:
    should_use_reasoning: bool
    supports_function_calling: bool
    call_kwargs: dict
    announced_tools: list[str]


@dataclass(frozen=True)
class AgentLoopPreparedMessages:
    messages: list[dict]
    initial_content_blocks: list[Any] = field(default_factory=list)
    final_tool_names: list[str] = field(default_factory=list)


def announced_tool_names_from_call_kwargs(call_kwargs: dict) -> list[str]:
    ordered_names: list[str] = []
    for tool in call_kwargs.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            ordered_names.append(str(fn["name"]))
    return ordered_names


def build_agent_loop_call_config(
    *,
    provider: str,
    options: dict | None,
    capabilities: dict | None,
    volcengine_providers: set[str] | frozenset[str] = frozenset(VOLCENGINE_PROVIDERS),
    build_web_search_tool_fn: Callable[[], dict] = build_web_search_tool,
) -> AgentLoopCallConfig:
    options = options or {}
    capabilities = capabilities or {}

    use_reasoning = options.get("use_reasoning")
    supports_thinking = bool(capabilities.get("deepThinking", False))
    should_use_reasoning = use_reasoning is True or (use_reasoning is None and supports_thinking)

    supports_function_calling = bool(capabilities.get("functionCalling", False)) and bool(
        capabilities.get("agentTools", capabilities.get("functionCalling", False))
    )
    call_kwargs: dict = {}
    if supports_function_calling:
        call_kwargs["tools"] = [build_web_search_tool_fn()]
        call_kwargs["tool_choice"] = "auto"
        if should_use_reasoning and provider in volcengine_providers:
            merge_extra_body(call_kwargs, {"thinking": {"type": "disabled"}})

    return AgentLoopCallConfig(
        should_use_reasoning=should_use_reasoning,
        supports_function_calling=supports_function_calling,
        call_kwargs=call_kwargs,
        announced_tools=announced_tool_names_from_call_kwargs(call_kwargs),
    )


def load_user_system_prompt(db, user_id: str) -> str | None:
    from app.db.models import User as UserModel

    user_record = db.query(UserModel).filter(UserModel.id == user_id).first()
    return user_record.system_prompt if user_record else None


async def prepare_agent_loop_messages(
    *,
    db,
    user_id: str,
    raw_messages: list,
    has_vision: bool,
    file_ids: list | None,
    original_message: str,
    call_config: AgentLoopCallConfig,
    file_repo_factory: Callable[[Any], Any] | None = None,
    load_user_system_prompt_fn: Callable[[Any, str], str | None] | None = None,
    build_llm_messages_fn: Callable[..., Awaitable[list[dict]]] | None = None,
    is_image_file_fn: Callable[[str, Any], bool] | None = None,
    inject_file_content_fn: Callable[[list[dict], str, dict[str, str]], list[dict]] | None = None,
    preprocess_url_in_message_fn: Callable[..., Awaitable[tuple[Any | None, dict | None, str | None]]] | None = None,
    preprocess_user_input: bool = True,
    extra_system_prompts: list[str] | None = None,
) -> AgentLoopPreparedMessages:
    file_repo_factory = file_repo_factory or FileRepository
    load_user_system_prompt_fn = load_user_system_prompt_fn or load_user_system_prompt
    build_llm_messages_fn = build_llm_messages_fn or build_llm_messages
    is_image_file_fn = is_image_file_fn or is_image_file
    inject_file_content_fn = inject_file_content_fn or inject_file_content
    preprocess_url_in_message_fn = preprocess_url_in_message_fn or preprocess_url_in_message

    file_repo = file_repo_factory(db)
    user_system_prompt = load_user_system_prompt_fn(db, user_id)
    messages = await build_llm_messages_fn(raw_messages, has_vision, file_repo, user_system_prompt)
    messages = inject_extra_system_prompts(messages, extra_system_prompts or [])

    if preprocess_user_input:
        messages = _inject_non_image_file_contents(
            messages=messages,
            file_ids=file_ids,
            original_message=original_message,
            file_repo=file_repo,
            is_image_file_fn=is_image_file_fn,
            inject_file_content_fn=inject_file_content_fn,
        )

        messages, initial_content_blocks = await _prepare_url_context(
            messages=messages,
            original_message=original_message,
            call_config=call_config,
            preprocess_url_in_message_fn=preprocess_url_in_message_fn,
        )
    else:
        initial_content_blocks = []

    messages = inject_tool_usage_contract(messages, call_config.call_kwargs)
    return AgentLoopPreparedMessages(
        messages=messages,
        initial_content_blocks=initial_content_blocks,
        final_tool_names=announced_tool_names_from_call_kwargs(call_config.call_kwargs),
    )


def inject_extra_system_prompts(messages: list[dict], prompts: list[str]) -> list[dict]:
    if not prompts:
        return messages

    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    prompt_messages = [{"role": "system", "content": prompt} for prompt in prompts]
    return [*messages[:insert_at], *prompt_messages, *messages[insert_at:]]


def _inject_non_image_file_contents(
    *,
    messages: list[dict],
    file_ids: list | None,
    original_message: str,
    file_repo: Any,
    is_image_file_fn: Callable[[str, Any], bool],
    inject_file_content_fn: Callable[[list[dict], str, dict[str, str]], list[dict]],
) -> list[dict]:
    if not file_ids:
        return messages

    non_image_ids = [fid for fid in file_ids if not is_image_file_fn(fid, file_repo)]
    if not non_image_ids:
        return messages

    file_contents = file_repo.get_parsed_file_content(non_image_ids)
    if not file_contents:
        return messages
    return inject_file_content_fn(messages, original_message, file_contents)


async def _prepare_url_context(
    *,
    messages: list[dict],
    original_message: str,
    call_config: AgentLoopCallConfig,
    preprocess_url_in_message_fn: Callable[..., Awaitable[tuple[Any | None, dict | None, str | None]]],
) -> tuple[list[dict], list[Any]]:
    initial_content_blocks = []
    url_read_block, url_context_msg, _auto_detected_url = await preprocess_url_in_message_fn(
        original_message,
        call_config.supports_function_calling,
        call_config.call_kwargs,
    )
    if url_context_msg:
        messages.insert(-1, url_context_msg)
    if url_read_block:
        initial_content_blocks.append(url_read_block)
    return messages, initial_content_blocks


def inject_tool_usage_contract(messages: list[dict], call_kwargs: dict) -> list[dict]:
    """工具模式下补一条 system 约束，避免 reasoning 口头承诺搜索但不发 tool_call。"""
    if "web_search" not in set(announced_tool_names_from_call_kwargs(call_kwargs)):
        return messages
    if any(msg.get("role") == "system" and "【工具调用一致性规则】" in str(msg.get("content", "")) for msg in messages):
        return messages

    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    contract_msg = {"role": "system", "content": TOOL_USAGE_CONTRACT_PROMPT}
    return [*messages[:insert_at], contract_msg, *messages[insert_at:]]
