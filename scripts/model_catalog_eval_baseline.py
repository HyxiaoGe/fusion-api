"""对 Fusion 可选模型执行统一测验，生成 JSONL 基线和 summary。

脚本默认 dry-run，只列出将被测的模型和场景；显式 `--apply` 才会请求
`/api/chat/send`。v1.1 默认使用 stream transport，覆盖真实产品链路；
nonstream transport 仅用于快速 liveness smoke。
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4

import httpx

DEFAULT_FUSION_BASE_URL = "https://fusion.seanfield.org"
DEFAULT_TRANSPORT = "stream"
LONG_CONTEXT_THRESHOLD_TOKENS = 128_000

SLOW_RESPONSE_THRESHOLDS_MS = {
    "basic_chat": 15_000,
    "cn_factual": 20_000,
    "coding_reasoning": 30_000,
    "autonomous_search": 90_000,
    "long_context_contract": 60_000,
    "no_search_simple": 15_000,
    "no_vision_image_boundary": 30_000,
    "vision_image_understanding": 30_000,
    "long_answer": 60_000,
}

VISION_TEST_IMAGE_FILENAME = "fusion-vision-acceptance.png"
VISION_TEST_IMAGE_MIME_TYPE = "image/png"


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _build_vision_test_png() -> bytes:
    """生成一张带 FUSION 字样的小 PNG，避免脚本依赖 Pillow 或外部文件。"""
    glyphs = {
        "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
        "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
        "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
        "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    }
    width, height = 320, 140
    pixels = bytearray([255] * width * height * 3)

    def paint_rect(x0: int, y0: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        for y in range(max(0, y0), min(height, y0 + h)):
            row_start = y * width * 3
            for x in range(max(0, x0), min(width, x0 + w)):
                idx = row_start + x * 3
                pixels[idx : idx + 3] = bytes(color)

    paint_rect(8, 8, width - 16, 4, (0, 90, 180))
    paint_rect(8, height - 12, width - 16, 4, (0, 90, 180))
    paint_rect(8, 8, 4, height - 16, (0, 90, 180))
    paint_rect(width - 12, 8, 4, height - 16, (0, 90, 180))

    scale = 8
    x = 24
    y = 42
    for char in "FUSION":
        for row_idx, row in enumerate(glyphs[char]):
            for col_idx, bit in enumerate(row):
                if bit == "1":
                    paint_rect(x + col_idx * scale, y + row_idx * scale, scale - 1, scale - 1, (0, 0, 0))
        x += 6 * scale

    raw = b"".join(b"\x00" + pixels[row * width * 3 : (row + 1) * width * 3] for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )


VISION_TEST_IMAGE_BYTES = _build_vision_test_png()

QUALITY_FLAG_POLICIES: dict[str, dict[str, str]] = {
    "expected_answer_missing": {
        "severity": "medium",
        "recommendation": "回答未命中该场景的最小验收关键词，建议人工复核模型真实能力或调整提示词。",
    },
    "missing_no_vision_boundary": {
        "severity": "high",
        "recommendation": "非读图模型收到图片后未明确能力边界，可能造成用户误以为模型已经看图。",
    },
    "reasoning_tag_leak": {
        "severity": "high",
        "recommendation": "回答暴露内部思考标签，建议暂不作为默认模型或在渲染层兜底过滤。",
    },
    "expected_search_without_agent_tools": {
        "severity": "medium",
        "recommendation": "模型不支持 agent 工具，建议从实时搜索任务候选集中剔除或明确标注不可联网。",
    },
    "expected_search_without_read": {
        "severity": "medium",
        "recommendation": "搜索场景已触发联网但没有深读网页，建议降低搜索任务权重或强制读取关键来源。",
    },
    "slow_response": {
        "severity": "medium",
        "recommendation": "响应耗时超过场景阈值，建议在自动路由中降权或设置更短超时兜底。",
    },
}

QUALITY_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class EvalScenario:
    scenario_id: str
    category: str
    question: str
    expected_tool_use: str
    requires_source_read: bool = False
    required_capabilities: tuple[str, ...] = ()
    excluded_capabilities: tuple[str, ...] = ()
    attachment_kind: str | None = None
    expected_answer_keywords: tuple[str, ...] = ()
    expects_no_vision_boundary: bool = False


DEFAULT_SCENARIOS: tuple[EvalScenario, ...] = (
    EvalScenario(
        scenario_id="basic_chat",
        category="basic",
        question="请用一句话介绍你能做什么。",
        expected_tool_use="forbidden",
    ),
    EvalScenario(
        scenario_id="cn_factual",
        category="factual",
        question="为什么苹果 iPhone 从 Lightning 接口换成 USB-C？请简洁说明核心原因。",
        expected_tool_use="forbidden",
    ),
    EvalScenario(
        scenario_id="coding_reasoning",
        category="reasoning",
        question="用 Python 写一个函数，判断字符串是否是回文，并说明时间复杂度。",
        expected_tool_use="forbidden",
    ),
    EvalScenario(
        scenario_id="autonomous_search",
        category="search",
        question="OpenAI 最近一次公开发布的新模型或模型更新是什么？请给出时间和依据。",
        expected_tool_use="expected",
        requires_source_read=True,
    ),
    EvalScenario(
        scenario_id="no_search_simple",
        category="search_guard",
        question="你好，今天可以帮我做什么？",
        expected_tool_use="forbidden",
    ),
    EvalScenario(
        scenario_id="vision_image_understanding",
        category="vision",
        question="请识别图片里的英文单词，只回答这个单词。",
        expected_tool_use="forbidden",
        required_capabilities=("vision",),
        attachment_kind="vision_test_image",
        expected_answer_keywords=("FUSION",),
    ),
    EvalScenario(
        scenario_id="no_vision_image_boundary",
        category="vision_guard",
        question="请识别图片里的英文单词，只回答这个单词。",
        expected_tool_use="forbidden",
        excluded_capabilities=("vision",),
        attachment_kind="vision_test_image",
        expects_no_vision_boundary=True,
    ),
    EvalScenario(
        scenario_id="long_context_contract",
        category="long_context",
        question=("请记住这个验收标记：FUSION-LONG-CONTEXT-V1。随后用一句话复述这个标记，并说明你适合处理较长上下文。"),
        expected_tool_use="forbidden",
        required_capabilities=("longContext",),
        expected_answer_keywords=("FUSION-LONG-CONTEXT-V1",),
    ),
    EvalScenario(
        scenario_id="long_answer",
        category="long_form",
        question="请用三段话说明 AI 编程助手在真实工程团队里的主要价值、风险和落地建议。",
        expected_tool_use="forbidden",
    ),
)


@dataclass(frozen=True)
class EvalResult:
    model_id: str
    provider: str
    model_name: str
    model_health: str
    scenario_id: str
    scenario_category: str
    question: str
    expected_tool_use: str
    requires_source_read: bool
    transport: str
    success: bool
    elapsed_ms: int
    answer_preview: str
    conversation_id: str
    message_id: str
    observed_tool_calls: int
    observed_tool_names: list[str]
    agent_tools_supported: bool
    capability_contract: dict[str, Any]
    tool_expectation_met: bool
    quality_flags: list[str]
    skipped: bool
    skip_reason: str
    required_capabilities: list[str]
    excluded_capabilities: list[str]
    attachment_kind: str | None
    attached_file_count: int
    attachment_kinds: list[str]
    error: dict[str, Any] | None


class StreamEvalError(RuntimeError):
    """流式响应中返回 error envelope。"""


def select_scenarios(scenario_ids: Sequence[str] | None = None) -> list[EvalScenario]:
    if not scenario_ids:
        return list(DEFAULT_SCENARIOS)
    scenarios_by_id = {scenario.scenario_id: scenario for scenario in DEFAULT_SCENARIOS}
    selected: list[EvalScenario] = []
    for scenario_id in scenario_ids:
        scenario = scenarios_by_id.get(scenario_id)
        if scenario is None:
            raise ValueError(f"未知测验场景: {scenario_id}")
        selected.append(scenario)
    return selected


def select_models(
    models: Sequence[Mapping[str, Any]],
    *,
    include_unhealthy: bool = False,
    model_ids: Sequence[str] | None = None,
) -> list[Mapping[str, Any]]:
    allowed_ids = set(model_ids or [])
    selected: list[Mapping[str, Any]] = []
    for model in models:
        model_id = str(model.get("modelId") or "")
        if allowed_ids and model_id not in allowed_ids:
            continue
        health = model.get("health") or {}
        if not include_unhealthy and health.get("status") == "unhealthy":
            continue
        selected.append(model)
    return selected


def _model_health_status(model: Mapping[str, Any]) -> str:
    health = model.get("health") or {}
    return str(health.get("status") or "unknown")


def _extract_answer_preview(response_payload: Mapping[str, Any], limit: int = 240) -> str:
    data = response_payload.get("data") or {}
    message = data.get("message") or {}
    content = message.get("content") or data.get("content") or ""
    if isinstance(content, list):
        content = " ".join(
            str(item.get("text") or item.get("content") or "") for item in content if isinstance(item, dict)
        )
    text = str(content).strip().replace("\n", " ")
    return text[:limit]


def _detect_quality_flags(answer_text: str) -> list[str]:
    normalized = answer_text.lower()
    if "<think" in normalized or "</think>" in normalized:
        return ["reasoning_tag_leak"]
    return []


def _detect_eval_quality_flags(
    *,
    scenario: EvalScenario,
    answer_text: str,
    observed_tool_calls: int,
    observed_tool_names: Sequence[str],
    agent_tools_supported: bool,
    elapsed_ms: int,
) -> list[str]:
    flags: list[str] = []
    normalized_answer = answer_text.lower()
    if scenario.expected_answer_keywords and not any(
        keyword.lower() in normalized_answer for keyword in scenario.expected_answer_keywords
    ):
        flags.append("expected_answer_missing")
    if scenario.expects_no_vision_boundary and not _answer_mentions_no_vision_boundary(answer_text):
        flags.append("missing_no_vision_boundary")
    if scenario.expected_tool_use == "expected" and not agent_tools_supported and observed_tool_calls == 0:
        flags.append("expected_search_without_agent_tools")
    if (
        scenario.expected_tool_use == "expected"
        and scenario.requires_source_read
        and agent_tools_supported
        and "web_search" in observed_tool_names
        and "url_read" not in observed_tool_names
    ):
        flags.append("expected_search_without_read")
    slow_threshold_ms = SLOW_RESPONSE_THRESHOLDS_MS.get(scenario.scenario_id)
    if slow_threshold_ms is not None and elapsed_ms > slow_threshold_ms:
        flags.append("slow_response")
    return flags


def _answer_mentions_no_vision_boundary(answer_text: str) -> bool:
    normalized = answer_text.lower()
    limitation_terms = ("无法", "不能", "不支持", "看不到", "不可", "cannot", "can't", "unable")
    has_image_term = any(term in normalized for term in ("图片", "image", "图像"))
    has_limitation_term = any(term in normalized for term in limitation_terms)
    return has_image_term and has_limitation_term


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _capability_contract(model: Mapping[str, Any]) -> dict[str, Any]:
    capabilities = model.get("capabilities") or {}
    if not isinstance(capabilities, Mapping):
        capabilities = {}

    function_calling = _coerce_bool(capabilities.get("functionCalling", False))
    agent_tools = _coerce_bool(capabilities.get("agentTools", function_calling))
    search_capable = _coerce_bool(capabilities.get("searchCapable", agent_tools))
    web_search = _coerce_bool(capabilities.get("webSearch", search_capable))
    context_window_tokens = _positive_int_or_none(model.get("contextWindowTokens"))
    max_output_tokens = _positive_int_or_none(model.get("maxOutputTokens"))

    return {
        "agentTools": agent_tools,
        "contextWindowTokens": context_window_tokens,
        "functionCalling": function_calling,
        "longContext": bool(
            context_window_tokens is not None and context_window_tokens >= LONG_CONTEXT_THRESHOLD_TOKENS
        ),
        "maxOutputTokens": max_output_tokens,
        "searchCapable": search_capable,
        "vision": _coerce_bool(capabilities.get("vision", False)),
        "webSearch": web_search,
    }


def _model_supports_agent_tools(model: Mapping[str, Any]) -> bool:
    return bool(_capability_contract(model)["agentTools"])


def _scenario_eligibility(model: Mapping[str, Any], scenario: EvalScenario) -> tuple[bool, str]:
    contract = _capability_contract(model)
    missing_required = [capability for capability in scenario.required_capabilities if not contract.get(capability)]
    if missing_required:
        return False, f"missing required capabilities: {', '.join(missing_required)}"
    present_excluded = [capability for capability in scenario.excluded_capabilities if contract.get(capability)]
    if present_excluded:
        return False, f"excluded capabilities present: {', '.join(present_excluded)}"
    return True, ""


def _scenario_attachment_kinds(scenario: EvalScenario, file_ids: Sequence[str] | None = None) -> list[str]:
    if not scenario.attachment_kind or not file_ids:
        return []
    return [scenario.attachment_kind for _ in file_ids]


def _tool_expectation_met(expected_tool_use: str, observed_tool_calls: int, agent_tools_supported: bool) -> bool:
    if expected_tool_use == "expected":
        if not agent_tools_supported:
            return observed_tool_calls == 0
        return observed_tool_calls > 0
    if expected_tool_use == "forbidden":
        return observed_tool_calls == 0
    return True


def _base_result_fields(
    *,
    model: Mapping[str, Any],
    scenario: EvalScenario,
    transport: str,
    elapsed_ms: int,
    answer_preview: str,
    answer_text: str = "",
    conversation_id: str = "",
    message_id: str = "",
    observed_tool_names: Sequence[str] | None = None,
    observed_tool_calls: int | None = None,
    quality_flags: Sequence[str] | None = None,
    include_eval_quality_flags: bool = True,
    skipped: bool = False,
    skip_reason: str = "",
    attachment_kinds: Sequence[str] | None = None,
) -> dict[str, Any]:
    tool_names = list(observed_tool_names or [])
    tool_call_count = len(tool_names) if observed_tool_calls is None else observed_tool_calls
    agent_tools_supported = _model_supports_agent_tools(model)
    eval_quality_flags = (
        _detect_eval_quality_flags(
            scenario=scenario,
            answer_text=answer_text or answer_preview,
            observed_tool_calls=tool_call_count,
            observed_tool_names=tool_names,
            agent_tools_supported=agent_tools_supported,
            elapsed_ms=elapsed_ms,
        )
        if include_eval_quality_flags and not skipped
        else []
    )
    result_quality_flags = _unique_in_order(
        [
            *(quality_flags or []),
            *eval_quality_flags,
        ]
    )
    attached_kinds = list(attachment_kinds or [])
    return {
        "model_id": str(model.get("modelId") or ""),
        "provider": str(model.get("provider") or ""),
        "model_name": str(model.get("name") or ""),
        "model_health": _model_health_status(model),
        "scenario_id": scenario.scenario_id,
        "scenario_category": scenario.category,
        "question": scenario.question,
        "expected_tool_use": scenario.expected_tool_use,
        "requires_source_read": scenario.requires_source_read,
        "transport": transport,
        "elapsed_ms": elapsed_ms,
        "answer_preview": answer_preview,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "observed_tool_calls": tool_call_count,
        "observed_tool_names": tool_names,
        "agent_tools_supported": agent_tools_supported,
        "capability_contract": _capability_contract(model),
        "tool_expectation_met": _tool_expectation_met(
            scenario.expected_tool_use,
            tool_call_count,
            agent_tools_supported,
        ),
        "quality_flags": result_quality_flags,
        "skipped": skipped,
        "skip_reason": skip_reason,
        "required_capabilities": list(scenario.required_capabilities),
        "excluded_capabilities": list(scenario.excluded_capabilities),
        "attachment_kind": scenario.attachment_kind,
        "attached_file_count": len(attached_kinds),
        "attachment_kinds": attached_kinds,
    }


def build_success_result(
    *,
    model: Mapping[str, Any],
    scenario: EvalScenario,
    transport: str,
    elapsed_ms: int,
    response_payload: Mapping[str, Any],
    attachment_kinds: Sequence[str] | None = None,
) -> EvalResult:
    data = response_payload.get("data") or {}
    message = data.get("message") or {}
    answer_preview = _extract_answer_preview(response_payload)
    return EvalResult(
        **_base_result_fields(
            model=model,
            scenario=scenario,
            transport=transport,
            elapsed_ms=elapsed_ms,
            answer_preview=answer_preview,
            answer_text=answer_preview,
            conversation_id=str(data.get("conversation_id") or response_payload.get("conversation_id") or ""),
            message_id=str(message.get("id") or response_payload.get("message_id") or ""),
            quality_flags=_detect_quality_flags(answer_preview),
            attachment_kinds=attachment_kinds,
        ),
        success=True,
        error=None,
    )


def _classify_error(error: Exception) -> str:
    if isinstance(error, StreamEvalError):
        return "stream_error"
    if isinstance(error, RuntimeError) and str(error) == "empty_answer":
        return "empty_answer"
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return "timeout"
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        if status_code in (401, 403):
            return "auth_error"
        return "http_error"
    return "unknown_error"


def build_failure_result(
    *,
    model: Mapping[str, Any],
    scenario: EvalScenario,
    transport: str,
    elapsed_ms: int,
    error: Exception,
    conversation_id: str = "",
    message_id: str = "",
    observed_tool_names: Sequence[str] | None = None,
    observed_tool_calls: int | None = None,
    attachment_kinds: Sequence[str] | None = None,
) -> EvalResult:
    return EvalResult(
        **_base_result_fields(
            model=model,
            scenario=scenario,
            transport=transport,
            elapsed_ms=elapsed_ms,
            answer_preview="",
            conversation_id=conversation_id,
            message_id=message_id,
            observed_tool_names=observed_tool_names,
            observed_tool_calls=observed_tool_calls,
            include_eval_quality_flags=False,
            attachment_kinds=attachment_kinds,
        ),
        success=False,
        error={"category": _classify_error(error), "type": type(error).__name__, "message": str(error)},
    )


def build_skipped_result(
    *,
    model: Mapping[str, Any],
    scenario: EvalScenario,
    transport: str,
    skip_reason: str,
) -> EvalResult:
    return EvalResult(
        **_base_result_fields(
            model=model,
            scenario=scenario,
            transport=transport,
            elapsed_ms=0,
            answer_preview="",
            include_eval_quality_flags=False,
            skipped=True,
            skip_reason=skip_reason,
        ),
        success=False,
        error=None,
    )


def _unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def parse_sse_events(lines: Iterable[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def _stream_error_from_event(event: Mapping[str, Any]) -> StreamEvalError:
    data = event.get("data") or {}
    if isinstance(data, Mapping):
        code = data.get("code")
        message = data.get("message") or data.get("error") or "流式响应返回错误"
        if code:
            return StreamEvalError(f"{code}: {message}")
        return StreamEvalError(str(message))
    return StreamEvalError(str(data or "流式响应返回错误"))


def build_stream_result(
    *,
    model: Mapping[str, Any],
    scenario: EvalScenario,
    elapsed_ms: int,
    events: Sequence[Mapping[str, Any]],
    response_payload: Mapping[str, Any] | None = None,
    attachment_kinds: Sequence[str] | None = None,
) -> EvalResult:
    response_payload = response_payload or {}
    answer_parts: list[str] = []
    tool_names: list[str] = []
    message_id = str(response_payload.get("message_id") or "")
    conversation_id = str(response_payload.get("conversation_id") or "")

    for event in events:
        chunk_type = event.get("chunk_type")
        data = event.get("data") or {}
        if not isinstance(data, Mapping):
            data = {}
        if chunk_type == "error":
            return build_failure_result(
                model=model,
                scenario=scenario,
                transport="stream",
                elapsed_ms=elapsed_ms,
                error=_stream_error_from_event(event),
                conversation_id=conversation_id,
                message_id=message_id,
                observed_tool_names=_unique_in_order(tool_names),
                observed_tool_calls=len(tool_names),
                attachment_kinds=attachment_kinds,
            )
        if chunk_type == "answering":
            answer_parts.append(str(data.get("delta") or ""))
        if chunk_type == "agent_event":
            if data.get("message_id") and not message_id:
                message_id = str(data.get("message_id"))
            event_type = data.get("type")
            if event_type == "tool_call_started":
                tool_names.append(str(data.get("tool_name") or ""))

    answer_text = "".join(answer_parts)
    answer_preview = answer_text.strip().replace("\n", " ")[:240]
    observed_tool_names = _unique_in_order(tool_names)
    if not answer_preview:
        return build_failure_result(
            model=model,
            scenario=scenario,
            transport="stream",
            elapsed_ms=elapsed_ms,
            error=RuntimeError("empty_answer"),
            conversation_id=conversation_id,
            message_id=message_id,
            observed_tool_names=observed_tool_names,
            observed_tool_calls=len(tool_names),
            attachment_kinds=attachment_kinds,
        )
    return EvalResult(
        **_base_result_fields(
            model=model,
            scenario=scenario,
            transport="stream",
            elapsed_ms=elapsed_ms,
            answer_preview=answer_preview,
            answer_text=answer_text,
            conversation_id=conversation_id,
            message_id=message_id,
            observed_tool_names=observed_tool_names,
            observed_tool_calls=len(tool_names),
            quality_flags=_detect_quality_flags(answer_text),
            attachment_kinds=attachment_kinds,
        ),
        success=True,
        error=None,
    )


def to_jsonl(result: EvalResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False, sort_keys=True) + "\n"


def load_results_from_jsonl(path: str | Path) -> list[EvalResult]:
    """从 JSONL 回放 EvalResult，便于不重跑 LLM 直接生成报告。"""
    results: list[EvalResult] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} 不是合法 JSONL 行") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} 必须是 JSON object")
        if "capability_contract" not in payload:
            agent_tools_supported = bool(payload.get("agent_tools_supported", False))
            payload["capability_contract"] = _capability_contract(
                {
                    "modelId": payload.get("model_id"),
                    "provider": payload.get("provider"),
                    "name": payload.get("model_name"),
                    "capabilities": {
                        "agentTools": agent_tools_supported,
                        "functionCalling": agent_tools_supported,
                        "searchCapable": agent_tools_supported,
                        "webSearch": agent_tools_supported,
                    },
                }
            )
        payload.setdefault("skipped", False)
        payload.setdefault("skip_reason", "")
        payload.setdefault("required_capabilities", [])
        payload.setdefault("excluded_capabilities", [])
        payload.setdefault("attachment_kind", None)
        payload.setdefault("attached_file_count", 0)
        payload.setdefault("attachment_kinds", [])
        results.append(EvalResult(**payload))
    return results


def fetch_models(base_url: str, auth_token: str | None = None) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    response = httpx.get(f"{base_url.rstrip('/')}/api/models/", headers=headers, timeout=20.0)
    response.raise_for_status()
    payload = response.json()
    return list((payload.get("data") or {}).get("models") or [])


def call_chat_send(
    *,
    base_url: str,
    auth_token: str,
    model_id: str,
    question: str,
    conversation_id: str | None = None,
    file_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"model_id": model_id, "message": question, "stream": False}
    if conversation_id:
        body["conversation_id"] = conversation_id
    if file_ids:
        body["file_ids"] = list(file_ids)
    response = httpx.post(
        f"{base_url.rstrip('/')}/api/chat/send",
        headers={"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"},
        json=body,
        timeout=90.0,
    )
    response.raise_for_status()
    return dict(response.json())


def call_chat_send_stream(
    *,
    base_url: str,
    auth_token: str,
    model_id: str,
    question: str,
    conversation_id: str | None = None,
    file_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    conversation_id = conversation_id or str(uuid4())
    body: dict[str, Any] = {
        "model_id": model_id,
        "message": question,
        "conversation_id": conversation_id,
        "stream": True,
    }
    if file_ids:
        body["file_ids"] = list(file_ids)
    with httpx.stream(
        "POST",
        f"{base_url.rstrip('/')}/api/chat/send",
        headers={"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"},
        json=body,
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        events = parse_sse_events(response.iter_lines())
    return events, {"conversation_id": conversation_id}


def upload_scenario_files(
    *,
    base_url: str,
    auth_token: str,
    model: Mapping[str, Any],
    scenario: EvalScenario,
    conversation_id: str,
) -> list[str]:
    if scenario.attachment_kind != "vision_test_image":
        return []
    response = httpx.post(
        f"{base_url.rstrip('/')}/api/files/upload",
        headers={"Authorization": f"Bearer {auth_token}"},
        data={
            "provider": str(model.get("provider") or ""),
            "model": str(model.get("modelId") or ""),
            "conversation_id": conversation_id,
        },
        files={
            "files": (
                VISION_TEST_IMAGE_FILENAME,
                VISION_TEST_IMAGE_BYTES,
                VISION_TEST_IMAGE_MIME_TYPE,
            )
        },
        timeout=60.0,
    )
    response.raise_for_status()
    payload = response.json()
    items = (payload.get("data") or {}).get("files") or []
    file_ids = [str(item.get("file_id") or "") for item in items if isinstance(item, Mapping) and item.get("file_id")]
    if not file_ids:
        raise RuntimeError("upload_returned_no_file_ids")
    return file_ids


def run_eval(
    *,
    base_url: str,
    auth_token: str,
    models: Iterable[Mapping[str, Any]],
    scenarios: Iterable[EvalScenario],
    transport: str = DEFAULT_TRANSPORT,
    on_result: Callable[[EvalResult], None] | None = None,
) -> list[EvalResult]:
    results: list[EvalResult] = []

    def record_result(result: EvalResult) -> None:
        results.append(result)
        if on_result:
            on_result(result)

    for model in models:
        for scenario in scenarios:
            eligible, skip_reason = _scenario_eligibility(model, scenario)
            if not eligible:
                record_result(
                    build_skipped_result(
                        model=model,
                        scenario=scenario,
                        transport=transport,
                        skip_reason=skip_reason,
                    )
                )
                continue
            started = time.perf_counter()
            conversation_id = str(uuid4())
            file_ids: list[str] = []
            attachment_kinds: list[str] = []
            try:
                file_ids = upload_scenario_files(
                    base_url=base_url,
                    auth_token=auth_token,
                    model=model,
                    scenario=scenario,
                    conversation_id=conversation_id,
                )
                attachment_kinds = _scenario_attachment_kinds(scenario, file_ids)
                if transport == "stream":
                    events, response_payload = call_chat_send_stream(
                        base_url=base_url,
                        auth_token=auth_token,
                        model_id=str(model.get("modelId") or ""),
                        question=scenario.question,
                        conversation_id=conversation_id,
                        file_ids=file_ids,
                    )
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    record_result(
                        build_stream_result(
                            model=model,
                            scenario=scenario,
                            elapsed_ms=elapsed_ms,
                            events=events,
                            response_payload=response_payload,
                            attachment_kinds=attachment_kinds,
                        )
                    )
                else:
                    payload = call_chat_send(
                        base_url=base_url,
                        auth_token=auth_token,
                        model_id=str(model.get("modelId") or ""),
                        question=scenario.question,
                        conversation_id=conversation_id,
                        file_ids=file_ids,
                    )
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    record_result(
                        build_success_result(
                            model=model,
                            scenario=scenario,
                            transport=transport,
                            elapsed_ms=elapsed_ms,
                            response_payload=payload,
                            attachment_kinds=attachment_kinds,
                        )
                    )
            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                record_result(
                    build_failure_result(
                        model=model,
                        scenario=scenario,
                        transport=transport,
                        elapsed_ms=elapsed_ms,
                        error=exc,
                        conversation_id=conversation_id,
                        attachment_kinds=attachment_kinds,
                    )
                )
    return results


def _format_eval_progress(result: EvalResult, completed: int, total: int) -> str:
    if result.skipped:
        status = f"skipped:{result.skip_reason}"
    else:
        status = "success" if result.success else f"failure:{(result.error or {}).get('category', 'unknown_error')}"
    flags = ",".join(result.quality_flags) if result.quality_flags else "-"
    tools = ",".join(result.observed_tool_names) if result.observed_tool_names else "-"
    return (
        f"[{completed}/{total}] {result.model_id}/{result.scenario_id} "
        f"{status} {result.elapsed_ms}ms tools={result.observed_tool_calls}({tools}) flags={flags}"
    )


def _empty_group() -> dict[str, Any]:
    return {"total": 0, "success_count": 0, "failure_count": 0, "skipped_count": 0, "elapsed_ms_total": 0}


def _record_group(group: dict[str, Any], result: EvalResult) -> None:
    group["total"] += 1
    group["elapsed_ms_total"] += result.elapsed_ms
    if result.skipped:
        group["skipped_count"] += 1
    elif result.success:
        group["success_count"] += 1
    else:
        group["failure_count"] += 1


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    total = group["total"]
    return {
        "total": total,
        "success_count": group["success_count"],
        "failure_count": group["failure_count"],
        "skipped_count": group["skipped_count"],
        "success_rate": round(group["success_count"] / total, 4) if total else 0,
        "avg_elapsed_ms": round(group["elapsed_ms_total"] / total) if total else 0,
    }


def _quality_flag_severity(flag: str) -> str:
    policy = QUALITY_FLAG_POLICIES.get(flag) or {}
    return policy.get("severity") or "low"


def _quality_flag_recommendation(flag: str) -> str:
    policy = QUALITY_FLAG_POLICIES.get(flag) or {}
    return policy.get("recommendation") or "质量标记未配置处理建议，需要人工复核。"


def _highest_quality_severity(flags: Sequence[str]) -> str:
    severity = "low"
    for flag in flags:
        candidate = _quality_flag_severity(flag)
        if QUALITY_SEVERITY_RANK.get(candidate, 0) > QUALITY_SEVERITY_RANK.get(severity, 0):
            severity = candidate
    return severity


def _build_quality_issue(result: EvalResult) -> dict[str, Any]:
    return {
        "model_id": result.model_id,
        "provider": result.provider,
        "scenario_id": result.scenario_id,
        "severity": _highest_quality_severity(result.quality_flags),
        "flags": list(result.quality_flags),
        "recommendations": _unique_in_order(_quality_flag_recommendation(flag) for flag in result.quality_flags),
    }


def _record_quality_risk_by_model(
    quality_risk_by_model: dict[str, dict[str, Any]],
    result: EvalResult,
    issue: Mapping[str, Any],
) -> None:
    model_risk = quality_risk_by_model.setdefault(
        result.model_id,
        {
            "provider": result.provider,
            "issue_count": 0,
            "flag_counts": {},
            "severity_counts": {},
        },
    )
    model_risk["issue_count"] += 1
    severity = str(issue.get("severity") or "low")
    model_risk["severity_counts"][severity] = model_risk["severity_counts"].get(severity, 0) + 1
    for flag in result.quality_flags:
        model_risk["flag_counts"][flag] = model_risk["flag_counts"].get(flag, 0) + 1


def _build_capability_contract_summary(results: Sequence[EvalResult]) -> dict[str, Any]:
    contracts_by_model: dict[str, Mapping[str, Any]] = {}
    for result in results:
        contracts_by_model.setdefault(result.model_id, result.capability_contract)

    models_by_capability: dict[str, list[str]] = {
        "agent_tools": [],
        "function_calling": [],
        "long_context": [],
        "search_capable": [],
        "vision": [],
        "web_search": [],
    }
    missing_context_window_count = 0

    for model_id, contract in contracts_by_model.items():
        if contract.get("agentTools"):
            models_by_capability["agent_tools"].append(model_id)
        if contract.get("functionCalling"):
            models_by_capability["function_calling"].append(model_id)
        if contract.get("longContext"):
            models_by_capability["long_context"].append(model_id)
        if contract.get("searchCapable"):
            models_by_capability["search_capable"].append(model_id)
        if contract.get("vision"):
            models_by_capability["vision"].append(model_id)
        if contract.get("webSearch"):
            models_by_capability["web_search"].append(model_id)
        if contract.get("contextWindowTokens") is None:
            missing_context_window_count += 1

    sorted_models_by_capability = {key: sorted(value) for key, value in models_by_capability.items()}
    return {
        "model_count": len(contracts_by_model),
        "agent_tools_count": len(sorted_models_by_capability["agent_tools"]),
        "function_calling_count": len(sorted_models_by_capability["function_calling"]),
        "long_context_count": len(sorted_models_by_capability["long_context"]),
        "search_capable_count": len(sorted_models_by_capability["search_capable"]),
        "vision_count": len(sorted_models_by_capability["vision"]),
        "web_search_count": len(sorted_models_by_capability["web_search"]),
        "missing_context_window_count": missing_context_window_count,
        "models_by_capability": sorted_models_by_capability,
    }


def _build_scenario_matrix_summary(results: Sequence[EvalResult]) -> dict[str, Any]:
    matrix: dict[str, dict[str, Any]] = {}
    for result in results:
        item = matrix.setdefault(
            result.scenario_id,
            {
                "category": result.scenario_category,
                "expected_tool_use": result.expected_tool_use,
                "requires_source_read": result.requires_source_read,
                "required_capabilities": list(result.required_capabilities),
                "excluded_capabilities": list(result.excluded_capabilities),
                "attachment_kind": result.attachment_kind,
                "total": 0,
                "success_count": 0,
                "failure_count": 0,
                "skipped_count": 0,
            },
        )
        item["total"] += 1
        if result.skipped:
            item["skipped_count"] += 1
        elif result.success:
            item["success_count"] += 1
        else:
            item["failure_count"] += 1
    return {key: matrix[key] for key in sorted(matrix)}


def _build_capability_behavior_summary(results: Sequence[EvalResult]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {
        "text": {"scenario_ids": ["basic_chat", "cn_factual", "coding_reasoning"], "tested": 0, "success": 0},
        "web_search": {"scenario_ids": ["autonomous_search"], "tested": 0, "success": 0},
        "vision": {"scenario_ids": ["vision_image_understanding"], "tested": 0, "success": 0},
        "no_vision_boundary": {"scenario_ids": ["no_vision_image_boundary"], "tested": 0, "success": 0},
        "long_context": {"scenario_ids": ["long_context_contract"], "tested": 0, "success": 0},
    }
    scenario_to_bucket = {
        scenario_id: bucket_name for bucket_name, bucket in buckets.items() for scenario_id in bucket["scenario_ids"]
    }
    for result in results:
        if result.skipped:
            continue
        bucket_name = scenario_to_bucket.get(result.scenario_id)
        if not bucket_name:
            continue
        bucket = buckets[bucket_name]
        bucket["tested"] += 1
        if result.success and not result.quality_flags:
            bucket["success"] += 1
    for bucket in buckets.values():
        tested = bucket["tested"]
        bucket["success_rate"] = round(bucket["success"] / tested, 4) if tested else 0
    return buckets


def build_summary(results: Sequence[EvalResult]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    by_scenario: dict[str, dict[str, Any]] = {}
    failure_types: dict[str, int] = {}
    quality_flags: dict[str, int] = {}
    quality_issues: list[dict[str, Any]] = []
    quality_risk_by_model: dict[str, dict[str, Any]] = {}
    mismatch_count = 0
    total_group = _empty_group()

    for result in results:
        _record_group(total_group, result)
        model_group = by_model.setdefault(result.model_id, _empty_group())
        scenario_group = by_scenario.setdefault(result.scenario_id, _empty_group())
        _record_group(model_group, result)
        _record_group(scenario_group, result)
        if result.skipped:
            continue
        if result.error:
            category = str(result.error.get("category") or "unknown_error")
            failure_types[category] = failure_types.get(category, 0) + 1
        for flag in result.quality_flags:
            quality_flags[flag] = quality_flags.get(flag, 0) + 1
        if result.quality_flags:
            issue = _build_quality_issue(result)
            quality_issues.append(issue)
            _record_quality_risk_by_model(quality_risk_by_model, result, issue)
        if not result.tool_expectation_met:
            mismatch_count += 1

    summary = _finalize_group(total_group)
    summary.update(
        {
            "by_model": {key: _finalize_group(value) for key, value in by_model.items()},
            "by_scenario": {key: _finalize_group(value) for key, value in by_scenario.items()},
            "failure_types": failure_types,
            "quality_flags": quality_flags,
            "quality_issue_count": len(quality_issues),
            "quality_issues": quality_issues,
            "quality_risk_by_model": quality_risk_by_model,
            "tool_expectation_mismatch_count": mismatch_count,
            "capability_contract": _build_capability_contract_summary(results),
            "capability_behavior_matrix": _build_capability_behavior_summary(results),
            "scenario_matrix": _build_scenario_matrix_summary(results),
        }
    )
    return summary


def _markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def _result_conversation_url(base_url: str, result: EvalResult) -> str:
    if not result.conversation_id:
        return ""
    if result.conversation_id.startswith("http://") or result.conversation_id.startswith("https://"):
        return result.conversation_id
    return f"{base_url.rstrip('/')}/chat/{result.conversation_id}"


def _format_quality_flags(flags: Sequence[str]) -> str:
    return ", ".join(flags) if flags else "-"


def _format_tool_names(names: Sequence[str]) -> str:
    return ", ".join(names) if names else "-"


def build_markdown_report(
    results: Sequence[EvalResult],
    summary: Mapping[str, Any],
    *,
    base_url: str,
    generated_at: str | None = None,
    source_label: str = "",
) -> str:
    """把全模型验收 JSONL/Summary 转成可交付 Markdown 报告。"""
    generated = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    success_count = int(summary.get("success_count") or 0)
    total = int(summary.get("total") or 0)
    failure_count = int(summary.get("failure_count") or 0)
    skipped_count = int(summary.get("skipped_count") or 0)
    mismatch_count = int(summary.get("tool_expectation_mismatch_count") or 0)
    quality_issue_count = int(summary.get("quality_issue_count") or 0)

    lines: list[str] = [
        "# Fusion 全模型验收报告",
        "",
        "## 元信息",
        "",
        f"- 生成时间：`{_markdown_cell(generated)}`",
        f"- 目标环境：`{_markdown_cell(base_url.rstrip('/'))}`",
        f"- 数据来源：`{_markdown_cell(source_label or 'current run')}`",
        f"- 总体结果：`{success_count}/{total}` 通过，失败 `{failure_count}`，跳过 `{skipped_count}`，工具契约不匹配 `{mismatch_count}`，质量风险 `{quality_issue_count}`",
        "",
        "## 自动验收总览",
        "",
        "| 维度 | 结果 |",
        "|---|---|",
        f"| 总用例 | `{total}` |",
        f"| 成功 | `{success_count}` |",
        f"| 失败 | `{failure_count}` |",
        f"| 跳过 | `{skipped_count}` |",
        f"| 平均耗时 | `{summary.get('avg_elapsed_ms', 0)}ms` |",
        f"| 工具契约不匹配 | `{mismatch_count}` |",
        f"| 质量风险 | `{quality_issue_count}` |",
        "",
        "## 按场景统计",
        "",
        "| 场景 | 成功/总数 | 平均耗时 |",
        "|---|---:|---:|",
    ]

    for scenario_id, item in sorted((summary.get("by_scenario") or {}).items()):
        skip_part = f"，跳过 {item.get('skipped_count', 0)}" if item.get("skipped_count") else ""
        lines.append(
            f"| `{_markdown_cell(scenario_id)}` | `{item.get('success_count', 0)}/{item.get('total', 0)}`{skip_part} | `{item.get('avg_elapsed_ms', 0)}ms` |"
        )

    lines.extend(
        [
            "",
            "## 按模型统计",
            "",
            "| 模型 | 成功/总数 | 平均耗时 |",
            "|---|---:|---:|",
        ]
    )
    for model_id, item in sorted((summary.get("by_model") or {}).items()):
        skip_part = f"，跳过 {item.get('skipped_count', 0)}" if item.get("skipped_count") else ""
        lines.append(
            f"| `{_markdown_cell(model_id)}` | `{item.get('success_count', 0)}/{item.get('total', 0)}`{skip_part} | `{item.get('avg_elapsed_ms', 0)}ms` |"
        )

    capability_contract = summary.get("capability_contract") or {}
    lines.extend(
        [
            "",
            "## 能力契约快照",
            "",
            "| 能力 | 模型数 |",
            "|---|---:|",
            f"| 可联网模型 | `{capability_contract.get('search_capable_count', 0)}` |",
            f"| 可调用工具模型 | `{capability_contract.get('agent_tools_count', 0)}` |",
            f"| Function Calling 模型 | `{capability_contract.get('function_calling_count', 0)}` |",
            f"| 视觉模型 | `{capability_contract.get('vision_count', 0)}` |",
            f"| 长上下文模型 | `{capability_contract.get('long_context_count', 0)}` |",
            f"| 缺少上下文窗口标注 | `{capability_contract.get('missing_context_window_count', 0)}` |",
        ]
    )

    lines.extend(
        [
            "",
            "## 场景矩阵",
            "",
            "| 场景 | 类别 | 需要能力 | 排除能力 | 附件 | 成功/失败/跳过 |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for scenario_id, item in (summary.get("scenario_matrix") or {}).items():
        lines.append(
            "| "
            f"`{_markdown_cell(scenario_id)}` | "
            f"`{_markdown_cell(item.get('category'))}` | "
            f"`{_markdown_cell(', '.join(item.get('required_capabilities') or []) or '-')}` | "
            f"`{_markdown_cell(', '.join(item.get('excluded_capabilities') or []) or '-')}` | "
            f"`{_markdown_cell(item.get('attachment_kind') or '-')}` | "
            f"`{item.get('success_count', 0)}/{item.get('failure_count', 0)}/{item.get('skipped_count', 0)}` |"
        )

    lines.extend(["", "## 质量风险", ""])
    quality_issues = list(summary.get("quality_issues") or [])
    if quality_issues:
        lines.extend(["| 模型 | 场景 | 严重度 | flags | 建议 |", "|---|---|---|---|---|"])
        for issue in quality_issues:
            lines.append(
                "| "
                f"`{_markdown_cell(issue.get('model_id'))}` | "
                f"`{_markdown_cell(issue.get('scenario_id'))}` | "
                f"`{_markdown_cell(issue.get('severity'))}` | "
                f"`{_markdown_cell(', '.join(issue.get('flags') or []))}` | "
                f"{_markdown_cell('；'.join(issue.get('recommendations') or []))} |"
            )
    else:
        lines.append("- 无自动质量风险。")

    lines.extend(
        [
            "",
            "## 明细结果",
            "",
            "| 模型 | 场景 | 结果 | 耗时 | 工具 | flags | 对话 |",
            "|---|---|---|---:|---|---|---|",
        ]
    )
    for result in results:
        if result.skipped:
            status = "跳过"
        else:
            status = "通过" if result.success else "失败"
        conv_url = _result_conversation_url(base_url, result)
        conv_cell = conv_url or result.conversation_id or "-"
        flags = _format_quality_flags(result.quality_flags)
        if result.skipped and result.skip_reason:
            flags = result.skip_reason
        lines.append(
            "| "
            f"`{_markdown_cell(result.model_id)}` | "
            f"`{_markdown_cell(result.scenario_id)}` | "
            f"{status} | "
            f"`{result.elapsed_ms}ms` | "
            f"`{_markdown_cell(_format_tool_names(result.observed_tool_names))}` | "
            f"`{_markdown_cell(flags)}` | "
            f"{_markdown_cell(conv_cell)} |"
        )

    lines.extend(
        [
            "",
            "## 真实 Chrome 回归补充记录",
            "",
            "- 只复用用户已打开且已登录的 Fusion Chrome 标签，禁止新开 Chrome/标签。",
            "- 没有可复用标签时记录阻塞，不用本地服务或旧历史页替代结论。",
            "",
            "| 用例 | 输入/页面 | 预期 | 实际 | console error | 刷新后结果 | 结论 |",
            "|---|---|---|---|---|---|---|",
            "| 模型选择器 | `/chat/new` | 模型目录、能力标签、上传入口与 `/api/models/` 一致 |  |  |  |  |",
            "| 实时搜索代表用例 | 新建真实对话 | 可联网模型展示搜索、读取、回答依据 |  |  |  |  |",
            "| 非联网代表用例 | 新建真实对话 | 不展示工具过程，并说明实时能力边界 |  |  |  |  |",
            "| 刷新恢复 | 已完成对话 URL | 正文、执行过程、回答依据按场景恢复 |  |  |  |  |",
            "",
            "## 推荐判定",
            "",
            "- 自动验收失败、工具契约不匹配或高严重度质量风险：不得作为推荐模型上线。",
            "- 慢响应属于质量风险，优先进入模型标注/路由权重评估，不自动替换用户显式选择的模型。",
            "- 新增、下线或能力标注调整后，应重新生成 JSONL、summary 和本报告。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_dry_run_rows(
    *,
    models: Sequence[Mapping[str, Any]],
    scenarios: Sequence[EvalScenario],
    transport: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for scenario in scenarios:
            eligible, skip_reason = _scenario_eligibility(model, scenario)
            rows.append(
                {
                    "modelId": model.get("modelId"),
                    "provider": model.get("provider"),
                    "health": _model_health_status(model),
                    "scenario_id": scenario.scenario_id,
                    "scenario_category": scenario.category,
                    "expected_tool_use": scenario.expected_tool_use,
                    "requires_source_read": scenario.requires_source_read,
                    "transport": transport,
                    "capability_contract": _capability_contract(model),
                    "required_capabilities": list(scenario.required_capabilities),
                    "excluded_capabilities": list(scenario.excluded_capabilities),
                    "attachment_kind": scenario.attachment_kind,
                    "eligible": eligible,
                    "skip_reason": skip_reason,
                }
            )
    return rows


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Fusion 多模型 smoke 基线")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只列出将被测模型（默认）")
    mode.add_argument("--apply", action="store_true", help="实际调用 /api/chat/send")
    parser.add_argument("--base-url", default=DEFAULT_FUSION_BASE_URL)
    parser.add_argument("--auth-token", default="")
    parser.add_argument("--transport", choices=("stream", "nonstream"), default=DEFAULT_TRANSPORT)
    parser.add_argument("--scenarios", default="", help="逗号分隔的 scenario id 白名单")
    parser.add_argument("--models", default="", help="逗号分隔的 modelId 白名单")
    parser.add_argument("--include-unhealthy", action="store_true")
    parser.add_argument("--output", default="", help="JSONL 输出文件；为空则输出到 stdout")
    parser.add_argument("--summary-output", default="", help="summary JSON 输出文件；为空则输出到 stderr")
    parser.add_argument("--report-output", default="", help="Markdown 验收报告输出文件")
    parser.add_argument("--from-jsonl", default="", help="从已有 JSONL 结果生成 summary/report，不重新请求 Fusion")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if args.from_jsonl:
        results = load_results_from_jsonl(args.from_jsonl)
        summary = build_summary(results)
        summary_content = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        if args.summary_output:
            Path(args.summary_output).write_text(summary_content + "\n", encoding="utf-8")
        else:
            print(summary_content, file=sys.stderr)
        report = build_markdown_report(
            results,
            summary,
            base_url=args.base_url,
            source_label=args.from_jsonl,
        )
        if args.report_output:
            Path(args.report_output).write_text(report, encoding="utf-8")
        else:
            print(report)
        return 0

    models = fetch_models(args.base_url, args.auth_token or None)
    selected = select_models(
        models,
        include_unhealthy=args.include_unhealthy,
        model_ids=_split_csv(args.models),
    )
    scenarios = select_scenarios(_split_csv(args.scenarios))

    if not args.apply:
        rows = build_dry_run_rows(models=selected, scenarios=scenarios, transport=args.transport)
        print(
            json.dumps(
                {"total": len(rows), "items": rows},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not args.auth_token:
        raise RuntimeError("实际测验需要 --auth-token")

    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.write_text("", encoding="utf-8")
    total_items = len(selected) * len(scenarios)
    completed_items = 0

    def on_result(result: EvalResult) -> None:
        nonlocal completed_items
        completed_items += 1
        line = to_jsonl(result)
        if output_path:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        else:
            print(line, end="", flush=True)
        print(_format_eval_progress(result, completed_items, total_items), file=sys.stderr, flush=True)

    results = run_eval(
        base_url=args.base_url,
        auth_token=args.auth_token,
        models=selected,
        scenarios=scenarios,
        transport=args.transport,
        on_result=on_result,
    )

    summary = build_summary(results)
    summary_content = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.summary_output:
        Path(args.summary_output).write_text(summary_content + "\n", encoding="utf-8")
    else:
        print(summary_content, file=sys.stderr)
    if args.report_output:
        report = build_markdown_report(
            results,
            summary,
            base_url=args.base_url,
            source_label=str(output_path) if output_path else "stdout",
        )
        Path(args.report_output).write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
