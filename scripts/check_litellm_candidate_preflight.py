"""对 LiteLLM 候选模型执行注册前最小契约验收。

默认 dry-run 只输出计划矩阵；只有显式传入 ``--apply`` 才发送收费请求。
本脚本不会注册模型、修改 allowlist 或调用 Fusion。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Protocol

import httpx

LITELLM_BASE_URL_ENV = "LITELLM_BASE_URL"
LITELLM_PROXY_URL_ENV = "LITELLM_PROXY_URL"
LITELLM_CANDIDATE_KEY_ENV = "LITELLM_CANDIDATE_KEY"
DEFAULT_LITELLM_BASE_URL = "http://localhost:4000"
DEFAULT_ACCEPTANCE_TTL_SECONDS = 7 * 24 * 60 * 60
RED_PIXEL_PNG_DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


class HttpClient(Protocol):
    def post(self, url: str, **kwargs: Any) -> Any: ...

    def stream(self, method: str, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class Candidate:
    model_id: str
    litellm_model: str
    preflight_model: str
    preflight_route: dict[str, Any]
    capabilities: dict[str, Any]
    pricing: dict[str, Any]
    provider_key: str = ""
    api_base: str = ""
    api_key_env: str = ""
    metadata_evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def supports_tool_calling(self) -> bool:
        return any(
            self.capabilities.get(key) is True
            for key in ("functionCalling", "toolCalling", "tool_calling", "supports_function_calling")
        )

    @property
    def supports_vision(self) -> bool:
        return self.capabilities.get("vision") is True

    @property
    def supports_reasoning(self) -> bool:
        return self.capabilities.get("deepThinking") is True

    @property
    def requires_preserved_tool_round(self) -> bool:
        return self.supports_tool_calling and self.supports_reasoning


@dataclass(frozen=True)
class CaseResult:
    name: str
    status: str
    usage_present: bool | None = None
    cost_present: bool | None = None
    output_present: bool | None = None
    stream_done: bool | None = None
    reasoning_present: bool | None = None
    issues: tuple[str, ...] = ()
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreflightReport:
    healthy: bool
    dry_run: bool
    candidate: Candidate
    cases: list[CaseResult]


def parse_candidate(payload: Mapping[str, Any]) -> Candidate:
    """解析并校验候选模型描述。"""
    model_id = payload.get("model_id")
    litellm_model = payload.get("litellm_model")
    preflight_model = payload.get("preflight_model")
    preflight_route = payload.get("preflight_route")
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    registration = payload.get("registration")
    registration = registration if isinstance(registration, Mapping) else {}
    evidence = payload.get("metadata_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    capabilities = payload.get("capabilities", metadata.get("capabilities"))
    pricing = payload.get("pricing", metadata.get("pricing"))
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("candidate.model_id 必须是非空字符串")
    if not isinstance(litellm_model, str) or not litellm_model.strip():
        raise ValueError("candidate.litellm_model 必须是非空字符串")
    if not isinstance(preflight_model, str) or not preflight_model.strip():
        raise ValueError("candidate.preflight_model 必须是非空字符串")
    if not isinstance(preflight_route, Mapping):
        raise ValueError("candidate.preflight_route 必须是 JSON 对象")
    if preflight_route.get("status") != "ready":
        raise ValueError("candidate.preflight_route 尚未就绪")
    if not isinstance(capabilities, Mapping):
        raise ValueError("candidate.capabilities 必须是 JSON 对象")
    if not isinstance(pricing, Mapping):
        raise ValueError("candidate.pricing 必须是 JSON 对象")
    return Candidate(
        model_id=model_id.strip(),
        litellm_model=litellm_model.strip(),
        preflight_model=preflight_model.strip(),
        preflight_route=dict(preflight_route),
        capabilities=dict(capabilities),
        pricing=dict(pricing),
        provider_key=str(payload.get("provider_key") or metadata.get("provider_key") or "").strip(),
        api_base=str(registration.get("api_base") or "").strip(),
        api_key_env=str(registration.get("api_key_env") or "").strip(),
        metadata_evidence=dict(evidence),
        metadata=dict(metadata) if metadata else {"capabilities": dict(capabilities), "pricing": dict(pricing)},
    )


def candidate_contract_fingerprint(candidate: Candidate | Mapping[str, Any]) -> str:
    normalized = candidate if isinstance(candidate, Candidate) else parse_candidate(candidate)
    contract = asdict(normalized)
    # 全量成本表 SHA 用于追溯本次快照，但其他厂商条目的变化不应让已经通过
    # 验收的候选失效。候选自身的价格、能力、cost_map_key 和 metadata 仍在
    # contract 中，相关条目变化会自然产生新指纹。
    contract["metadata_evidence"].pop("cost_map_sha256", None)
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _request_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _has_usage(payload: Mapping[str, Any]) -> bool:
    usage = payload.get("usage")
    return isinstance(usage, Mapping) and bool(usage)


def _has_cost(response: Any, payload: Mapping[str, Any]) -> bool:
    def valid(value: Any) -> bool:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(parsed) and parsed > 0

    headers = getattr(response, "headers", {})
    if valid(headers.get("x-litellm-response-cost")):
        return True
    if valid(payload.get("response_cost")):
        return True
    hidden = payload.get("_hidden_params")
    return isinstance(hidden, Mapping) and valid(hidden.get("response_cost"))


def _case_result(
    *,
    name: str,
    output_present: bool,
    usage_present: bool,
    cost_present: bool,
    stream_done: bool | None = None,
    output_text: str = "",
    reasoning_present: bool | None = None,
    extra_issues: tuple[str, ...] = (),
) -> CaseResult:
    issues: list[str] = list(extra_issues)
    if not output_present:
        issues.append("missing_output")
    if stream_done is False:
        issues.append("missing_done")
    if not usage_present:
        issues.append("missing_usage")
    if not cost_present:
        issues.append("missing_cost")
    return CaseResult(
        name=name,
        status="passed" if not issues else "failed",
        usage_present=usage_present,
        cost_present=cost_present,
        output_present=output_present,
        stream_done=stream_done,
        reasoning_present=reasoning_present,
        issues=tuple(issues),
        quality_flags=_detect_quality_flags(output_text),
    )


def _detect_quality_flags(text: str) -> tuple[str, ...]:
    normalized = text.casefold()
    if "<think>" in normalized or "</think>" in normalized:
        return ("reasoning_tag_leak",)
    return ()


def _failed_case(name: str, exc: Exception, *, stream: bool = False) -> CaseResult:
    return CaseResult(
        name=name,
        status="failed",
        stream_done=False if stream else None,
        issues=(type(exc).__name__,),
    )


def _run_text_case(
    *,
    candidate: Candidate,
    url: str,
    api_key: str,
    timeout_seconds: float,
    client: HttpClient,
) -> CaseResult:
    try:
        response = client.post(
            url,
            headers=_request_headers(api_key),
            json={
                "model": candidate.preflight_model,
                "messages": [{"role": "user", "content": "只回复 ok"}],
                "max_tokens": 8,
                "temperature": 0,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("响应不是 JSON 对象")
        choices = payload.get("choices")
        message = choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
        content = message.get("content") if isinstance(message, Mapping) else None
        return _case_result(
            name="text",
            output_present=isinstance(content, str) and bool(content.strip()),
            usage_present=_has_usage(payload),
            cost_present=_has_cost(response, payload),
            output_text=content if isinstance(content, str) else "",
        )
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        return _failed_case("text", exc)


def _stream_chunk_payload(line: str) -> Mapping[str, Any] | None:
    if not line.startswith("data:"):
        return None
    data = line.removeprefix("data:").strip()
    if not data or data == "[DONE]":
        return None
    payload = json.loads(data)
    return payload if isinstance(payload, Mapping) else None


def _run_stream_case(
    *,
    candidate: Candidate,
    url: str,
    api_key: str,
    timeout_seconds: float,
    client: HttpClient,
) -> CaseResult:
    text_parts: list[str] = []
    usage_present = False
    stream_done = False
    try:
        with client.stream(
            "POST",
            url,
            headers=_request_headers(api_key),
            json={
                "model": candidate.preflight_model,
                "messages": [{"role": "user", "content": "只回复 ok"}],
                "max_tokens": 8,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=timeout_seconds,
        ) as response:
            response.raise_for_status()
            cost_present = _has_cost(response, {})
            for line in response.iter_lines():
                stripped = line.strip()
                if stripped == "data: [DONE]":
                    stream_done = True
                    continue
                payload = _stream_chunk_payload(stripped)
                if payload is None:
                    continue
                usage_present = usage_present or _has_usage(payload)
                cost_present = cost_present or _has_cost(response, payload)
                choices = payload.get("choices")
                delta = choices[0].get("delta", {}) if isinstance(choices, list) and choices else {}
                content = delta.get("content") if isinstance(delta, Mapping) else None
                if isinstance(content, str):
                    text_parts.append(content)
        output_text = "".join(text_parts)
        return _case_result(
            name="stream",
            output_present=bool(output_text.strip()),
            usage_present=usage_present,
            cost_present=cost_present,
            stream_done=stream_done,
            output_text=output_text,
        )
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return _failed_case("stream", exc, stream=True)


def _run_tool_case(
    *,
    candidate: Candidate,
    url: str,
    api_key: str,
    timeout_seconds: float,
    client: HttpClient,
) -> CaseResult:
    try:
        response = client.post(
            url,
            headers=_request_headers(api_key),
            json={
                "model": candidate.preflight_model,
                "messages": [{"role": "user", "content": "查询上海天气"}],
                "max_tokens": 64,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "查询城市天气",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("响应不是 JSON 对象")
        choices = payload.get("choices")
        message = choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
        tool_calls = message.get("tool_calls") if isinstance(message, Mapping) else None
        return _case_result(
            name="tool_calling",
            output_present=isinstance(tool_calls, list) and bool(tool_calls),
            usage_present=_has_usage(payload),
            cost_present=_has_cost(response, payload),
        )
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        return _failed_case("tool_calling", exc)


def _run_vision_case(
    *,
    candidate: Candidate,
    url: str,
    api_key: str,
    timeout_seconds: float,
    client: HttpClient,
) -> CaseResult:
    try:
        response = client.post(
            url,
            headers=_request_headers(api_key),
            json={
                "model": candidate.preflight_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": RED_PIXEL_PNG_DATA_URL}},
                            {"type": "text", "text": "图片的主要颜色是什么？只回复 red 或 红色。"},
                        ],
                    }
                ],
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("响应不是 JSON 对象")
        choices = payload.get("choices")
        message = choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
        content = message.get("content") if isinstance(message, Mapping) else None
        text = content if isinstance(content, str) else ""
        normalized = text.casefold()
        matched = "red" in normalized or "红" in text
        return _case_result(
            name="vision",
            output_present=bool(text.strip()),
            usage_present=_has_usage(payload),
            cost_present=_has_cost(response, payload),
            output_text=text,
            extra_issues=() if matched else ("vision_expectation_mismatch",),
        )
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        return _failed_case("vision", exc)


def _run_reasoning_case(
    *,
    candidate: Candidate,
    url: str,
    api_key: str,
    timeout_seconds: float,
    client: HttpClient,
) -> CaseResult:
    try:
        response = client.post(
            url,
            headers=_request_headers(api_key),
            json={
                "model": candidate.preflight_model,
                "messages": [{"role": "user", "content": "计算 17×19，只回复答案。"}],
                "reasoning_effort": "low",
                "max_tokens": 64,
                "temperature": 0,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("响应不是 JSON 对象")
        choices = payload.get("choices")
        message = choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
        content = message.get("content") if isinstance(message, Mapping) else None
        reasoning = None
        if isinstance(message, Mapping):
            reasoning = message.get("reasoning_content") or message.get("reasoning")
        reasoning_present = isinstance(reasoning, str) and bool(reasoning.strip())
        return _case_result(
            name="reasoning",
            output_present=isinstance(content, str) and bool(content.strip()),
            usage_present=_has_usage(payload),
            cost_present=_has_cost(response, payload),
            output_text=content if isinstance(content, str) else "",
            reasoning_present=reasoning_present,
            extra_issues=() if reasoning_present else ("missing_reasoning",),
        )
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        return _failed_case("reasoning", exc)


def _run_preserved_tool_round_case(
    *,
    candidate: Candidate,
    url: str,
    api_key: str,
    timeout_seconds: float,
    client: HttpClient,
) -> CaseResult:
    try:
        first = client.post(
            url,
            headers=_request_headers(api_key),
            json={
                "model": candidate.preflight_model,
                "messages": [{"role": "user", "content": "查询上海天气，然后告诉我是否适合散步。"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "查询城市天气",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                "reasoning_effort": "low",
                "max_tokens": 128,
            },
            timeout=timeout_seconds,
        )
        first.raise_for_status()
        first_payload = first.json()
        choices = first_payload.get("choices") if isinstance(first_payload, Mapping) else None
        assistant = choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
        tool_calls = assistant.get("tool_calls") if isinstance(assistant, Mapping) else None
        if not isinstance(tool_calls, list) or not tool_calls:
            raise ValueError("首轮缺少 tool_calls")
        tool_call_id = tool_calls[0].get("id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("tool_call id 缺失")
        second = client.post(
            url,
            headers=_request_headers(api_key),
            json={
                "model": candidate.preflight_model,
                "messages": [
                    {"role": "user", "content": "查询上海天气，然后告诉我是否适合散步。"},
                    dict(assistant),
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": '{"weather":"sunny","temperature_c":24}',
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "查询城市天气",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "reasoning_effort": "low",
                "max_tokens": 128,
            },
            timeout=timeout_seconds,
        )
        second.raise_for_status()
        second_payload = second.json()
        second_choices = second_payload.get("choices") if isinstance(second_payload, Mapping) else None
        message = second_choices[0].get("message", {}) if isinstance(second_choices, list) and second_choices else {}
        content = message.get("content") if isinstance(message, Mapping) else None
        return _case_result(
            name="preserved_tool_round",
            output_present=isinstance(content, str) and bool(content.strip()),
            usage_present=_has_usage(first_payload) and _has_usage(second_payload),
            cost_present=_has_cost(first, first_payload) and _has_cost(second, second_payload),
            output_text=content if isinstance(content, str) else "",
        )
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        return _failed_case("preserved_tool_round", exc)


def _planned_cases(candidate: Candidate) -> list[CaseResult]:
    tool_status = "planned" if candidate.supports_tool_calling else "skipped"
    cases = [
        CaseResult(name="text", status="planned"),
        CaseResult(name="stream", status="planned"),
        CaseResult(name="tool_calling", status=tool_status),
    ]
    cases.append(CaseResult(name="vision", status="planned" if candidate.supports_vision else "skipped"))
    cases.append(CaseResult(name="reasoning", status="planned" if candidate.supports_reasoning else "skipped"))
    cases.append(
        CaseResult(
            name="preserved_tool_round",
            status="planned" if candidate.requires_preserved_tool_round else "skipped",
        )
    )
    return cases


def run_preflight(
    *,
    candidate: Candidate,
    base_url: str,
    api_key: str | None,
    apply: bool,
    client: HttpClient = httpx,
    timeout_seconds: float = 60.0,
) -> PreflightReport:
    """执行候选模型预准入；dry-run 时保证零请求。"""
    if not apply:
        return PreflightReport(
            healthy=True,
            dry_run=True,
            candidate=candidate,
            cases=_planned_cases(candidate),
        )
    if not api_key:
        raise ValueError("apply 模式缺少 LiteLLM 凭证")

    url = f"{base_url.rstrip('/')}/chat/completions"
    cases = [
        _run_text_case(
            candidate=candidate,
            url=url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            client=client,
        ),
        _run_stream_case(
            candidate=candidate,
            url=url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            client=client,
        ),
    ]
    if candidate.supports_tool_calling:
        cases.append(
            _run_tool_case(
                candidate=candidate,
                url=url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                client=client,
            )
        )
    else:
        cases.append(CaseResult(name="tool_calling", status="skipped"))
    if candidate.supports_vision:
        cases.append(
            _run_vision_case(
                candidate=candidate,
                url=url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                client=client,
            )
        )
    else:
        cases.append(CaseResult(name="vision", status="skipped"))
    if candidate.supports_reasoning:
        cases.append(
            _run_reasoning_case(
                candidate=candidate,
                url=url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                client=client,
            )
        )
    else:
        cases.append(CaseResult(name="reasoning", status="skipped"))
    if candidate.requires_preserved_tool_round:
        cases.append(
            _run_preserved_tool_round_case(
                candidate=candidate,
                url=url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                client=client,
            )
        )
    else:
        cases.append(CaseResult(name="preserved_tool_round", status="skipped"))
    return PreflightReport(
        healthy=all(case.status in {"passed", "skipped"} for case in cases),
        dry_run=False,
        candidate=candidate,
        cases=cases,
    )


def serialize_report(
    report: PreflightReport,
    *,
    context: Mapping[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """输出脱敏摘要，不序列化任何凭证字段。"""
    safe_context = {key: context[key] for key in ("litellm_base_url", "credential_source") if key in context}
    executed = [case for case in report.cases if case.status not in {"planned", "skipped"}]
    success_count = sum(1 for case in executed if case.status == "passed")
    failure_count = sum(1 for case in executed if case.status == "failed")
    case_by_name = {case.name: case for case in report.cases}
    quality_issues = [
        {
            "model_id": report.candidate.model_id,
            "scenario_id": case.name,
            "severity": "high",
            "flags": list(case.quality_flags),
        }
        for case in executed
        if case.quality_flags
    ]
    checks = {
        "text": case_by_name.get("text") is not None and case_by_name["text"].status == "passed",
        "stream": case_by_name.get("stream") is not None and case_by_name["stream"].status == "passed",
        "tool": case_by_name.get("tool_calling") is not None
        and case_by_name["tool_calling"].status in {"passed", "skipped"},
        "usage": bool(executed) and all(case.usage_present is True for case in executed),
        "cost": bool(executed) and all(case.cost_present is True for case in executed),
        "vision": not report.candidate.supports_vision
        or (case_by_name.get("vision") is not None and case_by_name["vision"].status == "passed"),
        "reasoning": not report.candidate.supports_reasoning
        or (
            case_by_name.get("reasoning") is not None
            and case_by_name["reasoning"].status == "passed"
            and case_by_name["reasoning"].reasoning_present is True
        ),
        "preserved_tool_round": not report.candidate.requires_preserved_tool_round
        or (
            case_by_name.get("preserved_tool_round") is not None
            and case_by_name["preserved_tool_round"].status == "passed"
        ),
    }
    generated_at = now or datetime.now(UTC)
    if generated_at.tzinfo is None:
        raise ValueError("验收时间必须包含时区")
    generated_at = generated_at.astimezone(UTC)
    expires_at = generated_at + timedelta(seconds=DEFAULT_ACCEPTANCE_TTL_SECONDS)
    return {
        "acceptance_stage": "candidate_pre_registration",
        "transport": "litellm_provider_wildcard",
        "generated_at": generated_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "healthy": report.healthy,
        "dry_run": report.dry_run,
        "candidate": asdict(report.candidate),
        "candidate_fingerprint": candidate_contract_fingerprint(report.candidate),
        "context": safe_context,
        "cases": [asdict(case) for case in report.cases],
        "by_model": {
            report.candidate.model_id: {
                "total": len(executed),
                "success_count": success_count,
                "failure_count": failure_count,
                "skipped_count": 0,
            }
        },
        "candidate_checks_by_model": {report.candidate.model_id: checks},
        "quality_issues": quality_issues,
        "tool_expectation_mismatch_count": 0 if checks["tool"] else 1,
    }


def _load_candidate(value: str) -> Candidate:
    if value == "-":
        raw = sys.stdin.read()
    else:
        path = Path(value)
        raw = path.read_text(encoding="utf-8") if path.is_file() else value
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("candidate JSON 必须是对象")
    return parse_candidate(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", help="candidate JSON 文件路径、内联 JSON 或 -（stdin）")
    parser.add_argument("--base-url", help="LiteLLM Proxy 地址")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只输出验收矩阵，不发送请求（默认）")
    mode.add_argument("--apply", action="store_true", help="显式执行会产生费用的真实请求")
    parser.add_argument("--timeout-seconds", type=float, default=60.0, help="单项 HTTP 超时秒数")
    return parser


def _print_error(code: str, exc: Exception) -> None:
    print(
        json.dumps(
            {"healthy": False, "error": {"code": code, "type": type(exc).__name__}},
            ensure_ascii=False,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        candidate = _load_candidate(args.candidate)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        _print_error("invalid_candidate", exc)
        return 2

    base_url = (
        args.base_url or os.getenv(LITELLM_BASE_URL_ENV) or os.getenv(LITELLM_PROXY_URL_ENV) or DEFAULT_LITELLM_BASE_URL
    )
    candidate_key = os.getenv(LITELLM_CANDIDATE_KEY_ENV)
    credential_source = LITELLM_CANDIDATE_KEY_ENV if candidate_key else "none"
    if args.apply and not candidate_key:
        _print_error("missing_candidate_key", ValueError("missing key"))
        return 2

    try:
        report = run_preflight(
            candidate=candidate,
            base_url=base_url,
            api_key=candidate_key,
            apply=args.apply,
            timeout_seconds=args.timeout_seconds,
        )
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        _print_error("preflight_failed", exc)
        return 1

    print(
        json.dumps(
            serialize_report(
                report,
                context={
                    "litellm_base_url": base_url,
                    "credential_source": credential_source,
                },
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report.healthy else 1


if __name__ == "__main__":
    sys.exit(main())
