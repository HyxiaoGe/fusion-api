"""Agent loop 请求进入 driver 前的输入准备。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from app.ai.prompts.agent_loop import (
    DEEP_RESEARCH_CONTRACT_PROMPT,
    get_agent_plan_control_prompt,
    get_no_tool_network_boundary_prompt,
    get_no_vision_file_boundary_prompt,
    get_tool_usage_contract_prompt,
)
from app.ai.prompts.system_prompt import SystemPromptSection, assemble_system_prompt
from app.ai.tools import build_url_read_tool, build_web_search_tool
from app.db.repositories import FileRepository
from app.services.agent.plan_coordinator import PlanMode
from app.services.chat.message_builder import (
    build_llm_messages,
    inject_file_content,
    is_image_file,
)
from app.services.mcp.amap_product_tools import AMAP_FACT_BOUNDARY_SYSTEM_PROMPT, AMAP_PRODUCT_TOOL_NAMES
from app.services.mcp.flyai_travel_tools import (
    FLYAI_TRAVEL_FACT_BOUNDARY_SYSTEM_PROMPT,
    FLYAI_TRAVEL_TOOL_NAMES,
)
from app.services.stream.agent_plan_tool_policy import AgentPlanToolPolicy, resolve_agent_plan_tool_policy
from app.services.stream.agent_task_policy import resolve_agent_task_policy
from app.services.stream.persistence import preprocess_url_in_message
from app.services.stream.reasoning_policy import configure_reasoning_call_kwargs

VOLCENGINE_PROVIDERS = {"volcengine"}
MAX_CONTROLLED_OUTPUT_TOKENS = 4096
PLAN_ITEM_ARGUMENT_NAME = "_plan_item_id"
PLAN_ITEM_ID_PATTERN = "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
VERIFIED_RESEARCH_PLAN_CONTRACT_PROMPT = """【可核验证据计划规则】
当前请求需要建立可核验的证据链。首个 update_plan 必须包含至少 1 个 web_search 步骤和至少 2 个独立的 url_read 来源核验步骤。每个 url_read 步骤必须通过 depends_on 直接或间接依赖 web_search，不能作为首批可执行步骤。最终 answer 或 synthesis 步骤必须依赖全部读取步骤。先搜索候选来源，再读取原文，最后综合结论。"""


@dataclass(frozen=True)
class AgentLoopCallConfig:
    should_use_reasoning: bool
    supports_function_calling: bool
    call_kwargs: dict
    announced_tools: list[str]
    supports_dynamic_tools: bool = False
    dynamic_tool_handlers: dict[str, Any] = field(default_factory=dict)
    tool_bindings: list[dict[str, Any]] = field(default_factory=list)
    plan_mode: PlanMode = "auto"
    control_tool_names: frozenset[str] = frozenset()
    task_mode: str = "standard"
    network_profile: str = "standard"
    evidence_policy: str = "standard"
    required_initial_tool_counts: dict[str, int] = field(default_factory=dict)
    plan_tool_policy_reason: str | None = None


def build_update_plan_tool(allowed_tool_names: list[str] | None = None) -> dict[str, Any]:
    """仅供 Agent Loop 控制面消费，不映射到任何外部 handler。"""

    planned_tool_schema: dict[str, Any] = {"type": "string"}
    normalized_allowed_tool_names = list(dict.fromkeys(allowed_tool_names or []))
    if normalized_allowed_tool_names:
        planned_tool_schema["enum"] = normalized_allowed_tool_names

    return {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": (
                "创建或更新本次任务的执行计划。复杂、多步骤或需要调用外部工具的任务应先调用；"
                "这是内部控制工具，不查询外部数据。"
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "explanation": {"type": "string", "description": "本次创建或修订计划的原因。"},
                    "plan": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 6,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "pattern": PLAN_ITEM_ID_PATTERN,
                                    "description": "稳定步骤 ID；可使用数字、字母、下划线和连字符。",
                                },
                                "step": {"type": "string", "description": "用户可理解的结果导向步骤。"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress"],
                                    "description": (
                                        "为兼容提供商接受 pending/in_progress；请始终提交 pending，"
                                        "真实运行态和终态由服务端推进。"
                                    ),
                                },
                                "kind": {
                                    "type": "string",
                                    "enum": ["reasoning", "search", "read", "synthesis", "answer", "other"],
                                },
                                "depends_on": {"type": "array", "items": {"type": "string"}},
                                "planned_tools": {
                                    "type": "array",
                                    "items": planned_tool_schema,
                                    "maxItems": 1,
                                    "description": ("执行步骤最多声明一个当前已提供的真实工具；多工具拆成独立步骤。"),
                                },
                            },
                            "required": ["id", "step", "status", "planned_tools"],
                        },
                    },
                },
                "required": ["plan"],
            },
        },
    }


@dataclass(frozen=True)
class AgentLoopPreparedMessages:
    messages: list[dict]
    initial_content_blocks: list[Any] = field(default_factory=list)
    final_tool_names: list[str] = field(default_factory=list)
    prompt_assembly: dict[str, Any] | None = None
    prompt_snapshot: dict[str, Any] | None = None


def announced_tool_names_from_call_kwargs(call_kwargs: dict) -> list[str]:
    ordered_names: list[str] = []
    for tool in call_kwargs.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            ordered_names.append(str(fn["name"]))
    return ordered_names


def _tool_definition_name(tool: dict) -> str:
    function = tool.get("function") if isinstance(tool, dict) else None
    return str(function.get("name", "")) if isinstance(function, dict) else ""


def _with_plan_item_binding(tool: dict, *, required: bool) -> dict:
    """给外部工具增加仅供 Agent Loop 消费的计划项关联字段。"""

    if _tool_definition_name(tool) == "update_plan":
        return tool
    prepared = deepcopy(tool)
    function = prepared.get("function")
    if not isinstance(function, dict):
        return prepared
    parameters = function.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        return prepared
    properties = parameters.setdefault("properties", {})
    if not isinstance(properties, dict):
        return prepared
    properties[PLAN_ITEM_ARGUMENT_NAME] = {
        "type": "string",
        "pattern": PLAN_ITEM_ID_PATTERN,
        "description": (
            "内部计划步骤 ID。存在执行计划时，必须填写本次调用所属步骤的精确 id；该字段由 Fusion 在调用真实工具前移除。"
        ),
    }
    if required:
        required_fields = parameters.setdefault("required", [])
        if isinstance(required_fields, list) and PLAN_ITEM_ARGUMENT_NAME not in required_fields:
            required_fields.append(PLAN_ITEM_ARGUMENT_NAME)
    return prepared


def supports_search_tools(capabilities: dict) -> bool:
    function_calling = bool(capabilities.get("functionCalling", False))
    if "searchCapable" in capabilities:
        return function_calling and bool(capabilities.get("searchCapable"))
    return function_calling and bool(capabilities.get("agentTools", capabilities.get("functionCalling", False)))


def supports_dynamic_agent_tools(capabilities: dict) -> bool:
    """显式拒绝 agentTools 的模型不得绕过兼容性门禁加载动态工具。"""
    return bool(capabilities.get("functionCalling", False)) and bool(
        capabilities.get("agentTools", capabilities.get("functionCalling", False))
    )


def normalize_controlled_max_tokens(value: Any) -> int | None:
    """只接受受控的正整数输出上限，拒绝 bool 与隐式类型转换。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return min(value, MAX_CONTROLLED_OUTPUT_TOKENS)


def build_agent_loop_call_config(
    *,
    provider: str,
    options: dict | None,
    capabilities: dict | None,
    volcengine_providers: set[str] | frozenset[str] = frozenset(VOLCENGINE_PROVIDERS),
    build_web_search_tool_fn: Callable[[], dict] = build_web_search_tool,
    build_url_read_tool_fn: Callable[[], dict] = build_url_read_tool,
    additional_tools: list[dict] | None = None,
    dynamic_tool_handlers: dict[str, Any] | None = None,
    tool_bindings: list[dict[str, Any]] | None = None,
    original_message: str | None = None,
    task_context_messages: list[object] | None = None,
) -> AgentLoopCallConfig:
    options = options or {}
    capabilities = capabilities or {}
    knowledge_grounded = options.get("knowledge_grounded") is True

    use_reasoning = options.get("use_reasoning")
    supports_thinking = bool(capabilities.get("deepThinking", False))
    should_use_reasoning = use_reasoning is True or (use_reasoning is None and supports_thinking)

    tools_disabled = options.get("disable_tools") is True or knowledge_grounded
    task_policy = resolve_agent_task_policy(options=options, capabilities=capabilities)
    requested_plan_mode = "off" if knowledge_grounded else task_policy.plan_mode
    supports_function_calling = supports_search_tools(capabilities) and not tools_disabled
    supports_control_tools = bool(capabilities.get("functionCalling", False)) and not tools_disabled
    plan_mode: PlanMode = requested_plan_mode if supports_control_tools else "off"
    supports_dynamic_tools = supports_dynamic_agent_tools(capabilities) and not tools_disabled
    call_kwargs: dict = {}
    max_tokens = normalize_controlled_max_tokens(options.get("max_tokens"))
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens
    tools: list[dict] = []
    if supports_function_calling:
        tools.append(build_web_search_tool_fn())
        if plan_mode != "off":
            tools.append(build_url_read_tool_fn())
    provided_handlers = dynamic_tool_handlers or {}
    if supports_dynamic_tools:
        tools.extend(tool for tool in (additional_tools or []) if _tool_definition_name(tool) in provided_handlers)
    external_tool_names = [_tool_definition_name(tool) for tool in tools if _tool_definition_name(tool)]
    if task_policy.task_mode == "deep_research":
        schedulable_names = frozenset({"web_search", "url_read"}).intersection(external_tool_names)
        plan_tool_policy = AgentPlanToolPolicy(
            allowed_tool_names=frozenset(schedulable_names),
            reason="deep_research_schedulable_tools",
        )
    else:
        plan_tool_policy = resolve_agent_plan_tool_policy(
            original_message=original_message,
            announced_tool_names=external_tool_names,
            task_context_messages=task_context_messages,
        )
    if requested_plan_mode != "off" and plan_tool_policy.allowed_tool_names is not None:
        tools = [tool for tool in tools if _tool_definition_name(tool) in plan_tool_policy.allowed_tool_names]
        external_tool_names = [_tool_definition_name(tool) for tool in tools if _tool_definition_name(tool)]
    control_tool_names: frozenset[str] = frozenset()
    if supports_control_tools and plan_mode != "off":
        tools.append(build_update_plan_tool(external_tool_names))
        control_tool_names = frozenset({"update_plan"})
        tools = [
            _with_plan_item_binding(tool, required=plan_mode == "on")
            if _tool_definition_name(tool) not in control_tool_names
            else tool
            for tool in tools
        ]
    if tools:
        call_kwargs["tools"] = tools
        call_kwargs["tool_choice"] = "auto"
    effective_provider = "volcengine" if provider in volcengine_providers else provider
    call_kwargs = configure_reasoning_call_kwargs(
        call_kwargs,
        provider=effective_provider,
        should_use_reasoning=should_use_reasoning,
    )

    announced_tools = [
        name for name in announced_tool_names_from_call_kwargs(call_kwargs) if name not in control_tool_names
    ]
    announced_tool_set = set(announced_tools)
    active_handlers = {name: handler for name, handler in provided_handlers.items() if name in announced_tool_set}
    active_bindings = [
        binding for binding in (tool_bindings or []) if str(binding.get("alias", "")) in announced_tool_set
    ]
    return AgentLoopCallConfig(
        should_use_reasoning=should_use_reasoning,
        supports_function_calling=supports_function_calling,
        call_kwargs=call_kwargs,
        announced_tools=announced_tools,
        supports_dynamic_tools=bool(active_handlers),
        dynamic_tool_handlers=active_handlers,
        tool_bindings=active_bindings,
        plan_mode=plan_mode,
        control_tool_names=control_tool_names,
        task_mode=task_policy.task_mode,
        network_profile=task_policy.network_profile,
        evidence_policy="knowledge_grounded_v1" if knowledge_grounded else task_policy.evidence_policy,
        required_initial_tool_counts=dict(plan_tool_policy.required_initial_tool_counts),
        plan_tool_policy_reason=plan_tool_policy.reason,
    )


def load_user_system_prompt(db, user_id: str) -> str | None:
    from app.db.models import User as UserModel

    user_record = db.query(UserModel).filter(UserModel.id == user_id).first()
    return user_record.system_prompt if user_record else None


async def prepare_agent_loop_messages(
    *,
    db,
    user_id: str,
    conversation_id: str | None = None,
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
    messages = await build_llm_messages_fn(
        raw_messages,
        has_vision,
        file_repo,
        None,
        include_base_system=False,
        user_id=user_id,
        conversation_id=conversation_id,
    )

    if preprocess_user_input:
        has_image_attachment = _has_image_file(
            file_ids=file_ids,
            file_repo=file_repo,
            is_image_file_fn=is_image_file_fn,
        )
        messages = _inject_non_image_file_contents(
            messages=messages,
            file_ids=file_ids,
            original_message=original_message,
            file_repo=file_repo,
            is_image_file_fn=is_image_file_fn,
            inject_file_content_fn=inject_file_content_fn,
        )

        if call_config.evidence_policy == "knowledge_grounded_v1":
            initial_content_blocks = []
        else:
            messages, initial_content_blocks = await _prepare_url_context(
                messages=messages,
                original_message=original_message,
                call_config=call_config,
                preprocess_url_in_message_fn=preprocess_url_in_message_fn,
            )
    else:
        initial_content_blocks = []
        has_image_attachment = False

    def selected_sections():
        # 只在空消息集上选择可信模板，用户文本不会影响段落是否存在。
        if has_image_attachment and not has_vision:
            yield SystemPromptSection("no_vision_file_boundary", get_no_vision_file_boundary_prompt())
        selectors = [
            ("tool_usage_contract", inject_tool_usage_contract, call_config.call_kwargs),
            ("amap_fact_boundary", inject_amap_fact_boundary, call_config.call_kwargs),
            ("flyai_travel_fact_boundary", inject_flyai_travel_fact_boundary, call_config.call_kwargs),
            ("agent_plan_control", inject_plan_control_contract, call_config),
            ("verified_research_plan", inject_verified_research_plan_contract, call_config),
            ("deep_research_contract", inject_deep_research_contract, call_config),
            ("no_tool_network_boundary", inject_no_tool_network_boundary, call_config.call_kwargs),
        ]
        for index, prompt in enumerate(extra_system_prompts or []):
            yield SystemPromptSection(f"extra_system_{index}", prompt)
        for section_id, selector, config in selectors:
            for message in selector([], config):
                yield SystemPromptSection(section_id, message["content"])

    assembly = assemble_system_prompt(user_system_prompt=user_system_prompt, sections=selected_sections)
    messages = [*assembly.messages, *messages]
    return AgentLoopPreparedMessages(
        messages=messages,
        initial_content_blocks=initial_content_blocks,
        prompt_assembly=assembly.metadata,
        prompt_snapshot={
            "schema_version": 1,
            "template_version": assembly.metadata["template_version"],
            "fingerprint": assembly.metadata["fingerprint"],
            "char_count": assembly.metadata["char_count"],
            "sections": [
                {"section_id": section_id, "content": message["content"]}
                for section_id, message in zip(assembly.metadata["section_ids"], assembly.messages, strict=True)
            ],
        },
        final_tool_names=[
            name
            for name in announced_tool_names_from_call_kwargs(call_config.call_kwargs)
            if name not in getattr(call_config, "control_tool_names", frozenset())
        ],
    )


def inject_extra_system_prompts(messages: list[dict], prompts: list[str]) -> list[dict]:
    if not prompts:
        return messages

    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    prompt_messages = [{"role": "system", "content": prompt} for prompt in prompts]
    return [*messages[:insert_at], *prompt_messages, *messages[insert_at:]]


def _has_image_file(
    *,
    file_ids: list | None,
    file_repo: Any,
    is_image_file_fn: Callable[[str, Any], bool],
) -> bool:
    if not file_ids:
        return False
    return any(is_image_file_fn(fid, file_repo) for fid in file_ids)


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
    if any(msg.get("role") == "system" and msg.get("content") == get_tool_usage_contract_prompt() for msg in messages):
        return messages

    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    contract_msg = {"role": "system", "content": get_tool_usage_contract_prompt()}
    return [*messages[:insert_at], contract_msg, *messages[insert_at:]]


def inject_amap_fact_boundary(messages: list[dict], call_kwargs: dict) -> list[dict]:
    """地点与路线产品工具启用时前置通用事实边界，不提升任何外部结果为 system 内容。"""
    announced_tools = set(announced_tool_names_from_call_kwargs(call_kwargs))
    if not AMAP_PRODUCT_TOOL_NAMES.intersection(announced_tools):
        return messages
    if any(msg.get("role") == "system" and msg.get("content") == AMAP_FACT_BOUNDARY_SYSTEM_PROMPT for msg in messages):
        return messages

    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    boundary_msg = {"role": "system", "content": AMAP_FACT_BOUNDARY_SYSTEM_PROMPT}
    return [*messages[:insert_at], boundary_msg, *messages[insert_at:]]


def inject_flyai_travel_fact_boundary(messages: list[dict], call_kwargs: dict) -> list[dict]:
    """航班或高铁产品工具启用时注入供应商中性的事实边界。"""

    announced_tools = set(announced_tool_names_from_call_kwargs(call_kwargs))
    if not FLYAI_TRAVEL_TOOL_NAMES.intersection(announced_tools):
        return messages
    if any(
        msg.get("role") == "system" and msg.get("content") == FLYAI_TRAVEL_FACT_BOUNDARY_SYSTEM_PROMPT
        for msg in messages
    ):
        return messages

    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    boundary_msg = {"role": "system", "content": FLYAI_TRAVEL_FACT_BOUNDARY_SYSTEM_PROMPT}
    return [*messages[:insert_at], boundary_msg, *messages[insert_at:]]


def inject_plan_control_contract(
    messages: list[dict],
    call_config: AgentLoopCallConfig,
) -> list[dict]:
    """向支持计划控制的模型注入行为契约，避免复杂任务只展示观察型占位计划。"""

    if "update_plan" not in getattr(call_config, "control_tool_names", frozenset()):
        return messages
    if any(
        msg.get("role") == "system" and msg.get("content") == get_agent_plan_control_prompt(call_config.plan_mode)
        for msg in messages
    ):
        return messages

    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    contract_msg = {
        "role": "system",
        "content": get_agent_plan_control_prompt(call_config.plan_mode),
    }
    return [*messages[:insert_at], contract_msg, *messages[insert_at:]]


def inject_verified_research_plan_contract(
    messages: list[dict],
    call_config: AgentLoopCallConfig,
) -> list[dict]:
    """仅为高置信可核验研究请求前置首计划 DAG 约束。"""

    policy_reasons = set((getattr(call_config, "plan_tool_policy_reason", "") or "").split("+"))
    if "verified_research_request" not in policy_reasons:
        return messages
    if any(
        message.get("role") == "system" and message.get("content") == VERIFIED_RESEARCH_PLAN_CONTRACT_PROMPT
        for message in messages
    ):
        return messages
    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    return [
        *messages[:insert_at],
        {"role": "system", "content": VERIFIED_RESEARCH_PLAN_CONTRACT_PROMPT},
        *messages[insert_at:],
    ]


def inject_deep_research_contract(
    messages: list[dict],
    call_config: AgentLoopCallConfig,
) -> list[dict]:
    if getattr(call_config, "task_mode", "standard") != "deep_research":
        return messages
    if any(
        message.get("role") == "system" and message.get("content") == DEEP_RESEARCH_CONTRACT_PROMPT
        for message in messages
    ):
        return messages
    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    return [
        *messages[:insert_at],
        {"role": "system", "content": DEEP_RESEARCH_CONTRACT_PROMPT},
        *messages[insert_at:],
    ]


def inject_no_tool_network_boundary(messages: list[dict], call_kwargs: dict) -> list[dict]:
    """无联网工具模式下补一条 system 边界，避免模型把内部知识包装成实时搜索。"""
    announced_tools = set(announced_tool_names_from_call_kwargs(call_kwargs))
    network_tool_names = {"web_search", "url_read"}
    if (
        network_tool_names.intersection(announced_tools)
        or AMAP_PRODUCT_TOOL_NAMES.intersection(announced_tools)
        or FLYAI_TRAVEL_TOOL_NAMES.intersection(announced_tools)
        or any(name.startswith("mcp_") for name in announced_tools)
    ):
        return messages
    if any(
        msg.get("role") == "system" and msg.get("content") == get_no_tool_network_boundary_prompt() for msg in messages
    ):
        return messages

    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    boundary_msg = {"role": "system", "content": get_no_tool_network_boundary_prompt()}
    return [*messages[:insert_at], boundary_msg, *messages[insert_at:]]


def inject_no_vision_file_boundary(messages: list[dict]) -> list[dict]:
    """图片已附加但当前模型无 vision 时，给 LLM 明确能力边界，避免臆测图片内容。"""
    if any(
        msg.get("role") == "system" and msg.get("content") == get_no_vision_file_boundary_prompt() for msg in messages
    ):
        return messages

    insert_at = 0
    while insert_at < len(messages) and messages[insert_at].get("role") == "system":
        insert_at += 1
    boundary_msg = {"role": "system", "content": get_no_vision_file_boundary_prompt()}
    return [*messages[:insert_at], boundary_msg, *messages[insert_at:]]
