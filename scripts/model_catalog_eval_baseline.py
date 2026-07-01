"""对 Fusion 可选模型执行统一测验，生成 JSONL 基线和 summary。

脚本默认 dry-run，只列出将被测的模型和场景；显式 `--apply` 才会请求
`/api/chat/send`。v1.1 默认使用 stream transport，覆盖真实产品链路；
nonstream transport 仅用于快速 liveness smoke。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4

import httpx

DEFAULT_FUSION_BASE_URL = "https://fusion.seanfield.org"
DEFAULT_TRANSPORT = "stream"

SLOW_RESPONSE_THRESHOLDS_MS = {
    "basic_chat": 15_000,
    "cn_factual": 20_000,
    "coding_reasoning": 30_000,
    "autonomous_search": 90_000,
    "no_search_simple": 15_000,
    "long_answer": 60_000,
}


@dataclass(frozen=True)
class EvalScenario:
    scenario_id: str
    category: str
    question: str
    expected_tool_use: str


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
    ),
    EvalScenario(
        scenario_id="no_search_simple",
        category="search_guard",
        question="你好，今天可以帮我做什么？",
        expected_tool_use="forbidden",
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
    transport: str
    success: bool
    elapsed_ms: int
    answer_preview: str
    conversation_id: str
    message_id: str
    observed_tool_calls: int
    observed_tool_names: list[str]
    agent_tools_supported: bool
    tool_expectation_met: bool
    quality_flags: list[str]
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
    observed_tool_calls: int,
    observed_tool_names: Sequence[str],
    agent_tools_supported: bool,
    elapsed_ms: int,
) -> list[str]:
    flags: list[str] = []
    if scenario.expected_tool_use == "expected" and not agent_tools_supported and observed_tool_calls == 0:
        flags.append("expected_search_without_agent_tools")
    if (
        scenario.expected_tool_use == "expected"
        and agent_tools_supported
        and "web_search" in observed_tool_names
        and "url_read" not in observed_tool_names
    ):
        flags.append("expected_search_without_read")
    slow_threshold_ms = SLOW_RESPONSE_THRESHOLDS_MS.get(scenario.scenario_id)
    if slow_threshold_ms is not None and elapsed_ms > slow_threshold_ms:
        flags.append("slow_response")
    return flags


def _model_supports_agent_tools(model: Mapping[str, Any]) -> bool:
    capabilities = model.get("capabilities") or {}
    if not isinstance(capabilities, Mapping):
        return False
    return bool(capabilities.get("agentTools", capabilities.get("functionCalling", False)))


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
    conversation_id: str = "",
    message_id: str = "",
    observed_tool_names: Sequence[str] | None = None,
    observed_tool_calls: int | None = None,
    quality_flags: Sequence[str] | None = None,
    include_eval_quality_flags: bool = True,
) -> dict[str, Any]:
    tool_names = list(observed_tool_names or [])
    tool_call_count = len(tool_names) if observed_tool_calls is None else observed_tool_calls
    agent_tools_supported = _model_supports_agent_tools(model)
    eval_quality_flags = (
        _detect_eval_quality_flags(
            scenario=scenario,
            observed_tool_calls=tool_call_count,
            observed_tool_names=tool_names,
            agent_tools_supported=agent_tools_supported,
            elapsed_ms=elapsed_ms,
        )
        if include_eval_quality_flags
        else []
    )
    result_quality_flags = _unique_in_order(
        [
            *(quality_flags or []),
            *eval_quality_flags,
        ]
    )
    return {
        "model_id": str(model.get("modelId") or ""),
        "provider": str(model.get("provider") or ""),
        "model_name": str(model.get("name") or ""),
        "model_health": _model_health_status(model),
        "scenario_id": scenario.scenario_id,
        "scenario_category": scenario.category,
        "question": scenario.question,
        "expected_tool_use": scenario.expected_tool_use,
        "transport": transport,
        "elapsed_ms": elapsed_ms,
        "answer_preview": answer_preview,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "observed_tool_calls": tool_call_count,
        "observed_tool_names": tool_names,
        "agent_tools_supported": agent_tools_supported,
        "tool_expectation_met": _tool_expectation_met(
            scenario.expected_tool_use,
            tool_call_count,
            agent_tools_supported,
        ),
        "quality_flags": result_quality_flags,
    }


def build_success_result(
    *,
    model: Mapping[str, Any],
    scenario: EvalScenario,
    transport: str,
    elapsed_ms: int,
    response_payload: Mapping[str, Any],
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
            conversation_id=str(data.get("conversation_id") or response_payload.get("conversation_id") or ""),
            message_id=str(message.get("id") or response_payload.get("message_id") or ""),
            quality_flags=_detect_quality_flags(answer_preview),
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
        ),
        success=False,
        error={"category": _classify_error(error), "type": type(error).__name__, "message": str(error)},
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
        )
    return EvalResult(
        **_base_result_fields(
            model=model,
            scenario=scenario,
            transport="stream",
            elapsed_ms=elapsed_ms,
            answer_preview=answer_preview,
            conversation_id=conversation_id,
            message_id=message_id,
            observed_tool_names=observed_tool_names,
            observed_tool_calls=len(tool_names),
            quality_flags=_detect_quality_flags(answer_text),
        ),
        success=True,
        error=None,
    )


def to_jsonl(result: EvalResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False, sort_keys=True) + "\n"


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
) -> dict[str, Any]:
    response = httpx.post(
        f"{base_url.rstrip('/')}/api/chat/send",
        headers={"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"},
        json={"model_id": model_id, "message": question, "stream": False},
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    conversation_id = conversation_id or str(uuid4())
    with httpx.stream(
        "POST",
        f"{base_url.rstrip('/')}/api/chat/send",
        headers={"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"},
        json={"model_id": model_id, "message": question, "conversation_id": conversation_id, "stream": True},
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        events = parse_sse_events(response.iter_lines())
    return events, {"conversation_id": conversation_id}


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
            started = time.perf_counter()
            try:
                if transport == "stream":
                    events, response_payload = call_chat_send_stream(
                        base_url=base_url,
                        auth_token=auth_token,
                        model_id=str(model.get("modelId") or ""),
                        question=scenario.question,
                    )
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    record_result(
                        build_stream_result(
                            model=model,
                            scenario=scenario,
                            elapsed_ms=elapsed_ms,
                            events=events,
                            response_payload=response_payload,
                        )
                    )
                else:
                    payload = call_chat_send(
                        base_url=base_url,
                        auth_token=auth_token,
                        model_id=str(model.get("modelId") or ""),
                        question=scenario.question,
                    )
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    record_result(
                        build_success_result(
                            model=model,
                            scenario=scenario,
                            transport=transport,
                            elapsed_ms=elapsed_ms,
                            response_payload=payload,
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
                    )
                )
    return results


def _format_eval_progress(result: EvalResult, completed: int, total: int) -> str:
    status = "success" if result.success else f"failure:{(result.error or {}).get('category', 'unknown_error')}"
    flags = ",".join(result.quality_flags) if result.quality_flags else "-"
    tools = ",".join(result.observed_tool_names) if result.observed_tool_names else "-"
    return (
        f"[{completed}/{total}] {result.model_id}/{result.scenario_id} "
        f"{status} {result.elapsed_ms}ms tools={result.observed_tool_calls}({tools}) flags={flags}"
    )


def _empty_group() -> dict[str, Any]:
    return {"total": 0, "success_count": 0, "failure_count": 0, "elapsed_ms_total": 0}


def _record_group(group: dict[str, Any], result: EvalResult) -> None:
    group["total"] += 1
    group["elapsed_ms_total"] += result.elapsed_ms
    if result.success:
        group["success_count"] += 1
    else:
        group["failure_count"] += 1


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    total = group["total"]
    return {
        "total": total,
        "success_count": group["success_count"],
        "failure_count": group["failure_count"],
        "success_rate": round(group["success_count"] / total, 4) if total else 0,
        "avg_elapsed_ms": round(group["elapsed_ms_total"] / total) if total else 0,
    }


def build_summary(results: Sequence[EvalResult]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    by_scenario: dict[str, dict[str, Any]] = {}
    failure_types: dict[str, int] = {}
    quality_flags: dict[str, int] = {}
    mismatch_count = 0
    total_group = _empty_group()

    for result in results:
        _record_group(total_group, result)
        model_group = by_model.setdefault(result.model_id, _empty_group())
        scenario_group = by_scenario.setdefault(result.scenario_id, _empty_group())
        _record_group(model_group, result)
        _record_group(scenario_group, result)
        if result.error:
            category = str(result.error.get("category") or "unknown_error")
            failure_types[category] = failure_types.get(category, 0) + 1
        for flag in result.quality_flags:
            quality_flags[flag] = quality_flags.get(flag, 0) + 1
        if not result.tool_expectation_met:
            mismatch_count += 1

    summary = _finalize_group(total_group)
    summary.update(
        {
            "by_model": {key: _finalize_group(value) for key, value in by_model.items()},
            "by_scenario": {key: _finalize_group(value) for key, value in by_scenario.items()},
            "failure_types": failure_types,
            "quality_flags": quality_flags,
            "tool_expectation_mismatch_count": mismatch_count,
        }
    )
    return summary


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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    models = fetch_models(args.base_url, args.auth_token or None)
    selected = select_models(
        models,
        include_unhealthy=args.include_unhealthy,
        model_ids=_split_csv(args.models),
    )
    scenarios = select_scenarios(_split_csv(args.scenarios))

    if not args.apply:
        rows = [
            {
                "modelId": model.get("modelId"),
                "provider": model.get("provider"),
                "health": _model_health_status(model),
                "scenario_id": scenario.scenario_id,
                "scenario_category": scenario.category,
                "expected_tool_use": scenario.expected_tool_use,
                "transport": args.transport,
            }
            for model in selected
            for scenario in scenarios
        ]
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

    summary_content = json.dumps(build_summary(results), ensure_ascii=False, indent=2, sort_keys=True)
    if args.summary_output:
        Path(args.summary_output).write_text(summary_content + "\n", encoding="utf-8")
    else:
        print(summary_content, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
