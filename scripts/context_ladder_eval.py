#!/usr/bin/env python3
"""长上下文串行阶梯评测 runner；默认只生成脱敏计划，不访问网络。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import urlsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.perf.core import CleanupManifest, SSEParser, extract_agent_trace_ids, fingerprint
from scripts.perf.resource_guard import ResourceGuard, UrllibPrometheusClient
from scripts.perf.runner import (
    DEFAULT_AUTH_URL,
    DEFAULT_CLIENT_ID,
    DEFAULT_TARGET_URL,
    HttpClient,
    RunnerError,
    authenticate,
    cleanup_run,
    generate_identity,
    join_url,
)
from scripts.perf.sse_metrics import SSEFlowMetrics

LOW_TARGETS = (5_000, 10_000, 20_000, 40_000)
WINDOW_RATIOS = (Decimal("0.6"), Decimal("0.8"))
MANAGED_TRIM_RATIO = Decimal("0.9")
MANAGED_TARGET_RATIO = Decimal("0.75")
# Kimi K2.5 等推理模型即使关闭前端 reasoning 展示，仍可能消耗内部推理 Token；
# 128 会在产生可见答案前触顶。1024 仍是严格小输出上限，并已由费用门禁覆盖。
MAX_OUTPUT_TOKENS = 1024
PRODUCTION_TARGET_HOST = "fusion.seanfield.org"
PRODUCTION_AUTH_HOST = "auth.seanfield.org"
_LOKI_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_LOKI_MAX_ATTEMPTS = 10
_SAFE_AGENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SAFE_MODEL_ID = re.compile(r"^[A-Za-z0-9._:/-]{1,160}$")


@dataclass(frozen=True)
class StagePlan:
    case_id: str
    track: str
    target_context_tokens: int
    ratio: Decimal | None
    conversation_key: str = field(repr=False)
    prompt: str = field(repr=False)
    expected_canaries: dict[str, str] = field(repr=False)
    local_estimated_prompt_tokens: int
    request_bytes: int
    planned_input_tokens: int
    prompt_hash: str
    planned_output_tokens: int = MAX_OUTPUT_TOKENS
    expected_management_status: str | None = None

    def safe_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "track": self.track,
            "target_context_tokens": self.target_context_tokens,
            "window_ratio": str(self.ratio) if self.ratio is not None else None,
            "local_estimated_prompt_tokens": self.local_estimated_prompt_tokens,
            "request_bytes": self.request_bytes,
            "planned_input_tokens": self.planned_input_tokens,
            "planned_output_tokens": self.planned_output_tokens,
            "prompt_hash": self.prompt_hash,
            "expected_canary_count": len(self.expected_canaries),
            "expected_management_status": self.expected_management_status,
        }


@dataclass(frozen=True)
class LadderPlan:
    model_id: str
    context_window: int
    stages: tuple[StagePlan, ...]
    projected_input_tokens: int
    projected_output_tokens: int
    projected_cost_usd: Decimal

    def safe_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "context_window": self.context_window,
            "serial_only": True,
            "stage_count": len(self.stages),
            "projected_input_tokens": self.projected_input_tokens,
            "projected_output_tokens": self.projected_output_tokens,
            "projected_cost_usd": _decimal_text(self.projected_cost_usd),
            "stages": [stage.safe_dict() for stage in self.stages],
        }


@dataclass(frozen=True)
class StageResult:
    case_id: str
    success: bool
    done: bool
    round_count: int
    tool_event_count: int
    error_frame_count: int
    canary_hits: dict[str, bool]
    answer_hash: str
    answer_length: int
    metrics: dict[str, int | float | None]
    stop_reasons: tuple[str, ...]
    agent_run_id: str | None = None
    assistant_message_id: str | None = None
    round_context: dict[str, Any] | None = None

    def safe_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "success": self.success,
            "done": self.done,
            "round_count": self.round_count,
            "tool_event_count": self.tool_event_count,
            "error_frame_count": self.error_frame_count,
            "canary_hits": self.canary_hits,
            "answer_hash": self.answer_hash,
            "answer_length": self.answer_length,
            "metrics": self.metrics,
            "stop_reasons": list(self.stop_reasons),
            "agent_run_id": self.agent_run_id,
            "assistant_message_id": self.assistant_message_id,
            "round_context": self.round_context,
        }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def _positive_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("必须是正数") from error
    if not parsed.is_finite() or parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正数")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fusion 长上下文串行阶梯评测（默认 dry-run）")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--context-window", required=True, type=_positive_int)
    parser.add_argument("--input-usd-per-million", required=True, type=_positive_decimal)
    parser.add_argument("--output-usd-per-million", required=True, type=_positive_decimal)
    parser.add_argument("--max-cost-usd", required=True, type=_positive_decimal)
    parser.add_argument("--max-request-bytes", required=True, type=_positive_int)
    parser.add_argument("--target-url", default=DEFAULT_TARGET_URL)
    parser.add_argument("--auth-url", default=DEFAULT_AUTH_URL)
    parser.add_argument("--prometheus-url", required=True)
    parser.add_argument("--loki-url", required=True)
    parser.add_argument("--client-id", default=DEFAULT_CLIENT_ID)
    parser.add_argument("--timeout-seconds", type=_positive_int, default=180)
    parser.add_argument("--start-case", default="", help="从指定 case 开始，供硬停后的安全续跑")
    parser.add_argument("--apply", action="store_true", help="实际串行执行；缺省仅输出计划")
    parser.add_argument("--confirm-production", action="store_true")
    parser.add_argument(
        "--generation-controls-verified",
        action="store_true",
        help="确认目标环境已真实支持 disable_tools 与 max_tokens 硬约束",
    )
    parser.add_argument(
        "--allow-account-residue",
        action="store_true",
        help="确认认证服务不支持删除压测账号，实际执行会永久保留账号行",
    )
    return parser


def _normalized_host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().rstrip(".")


def validate_args(args: argparse.Namespace) -> None:
    if not _SAFE_MODEL_ID.fullmatch(args.model_id):
        raise RunnerError("model-id 格式无效")
    _validate_monitoring_url(args.prometheus_url, "Prometheus")
    _validate_monitoring_url(args.loki_url, "Loki")
    target_host = _normalized_host(args.target_url)
    auth_host = _normalized_host(args.auth_url)
    production_requested = target_host == PRODUCTION_TARGET_HOST or auth_host == PRODUCTION_AUTH_HOST
    if args.apply and production_requested and not args.confirm_production:
        raise RunnerError("生产执行必须显式传入 --confirm-production")
    if args.apply and production_requested:
        if target_host != PRODUCTION_TARGET_HOST or auth_host != PRODUCTION_AUTH_HOST:
            raise RunnerError("生产执行必须同时使用预期的 Fusion 与认证域名")
    if args.apply and not args.generation_controls_verified:
        raise RunnerError("实际执行前必须传入 --generation-controls-verified")
    if args.apply and not args.allow_account_residue:
        raise RunnerError("实际执行会永久保留账号行，必须显式传入 --allow-account-residue")


def _validate_monitoring_url(url: str, label: str) -> None:
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError as error:
        raise RunnerError(f"{label} URL 无效") from error
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RunnerError(f"{label} URL 必须是 HTTP(S)")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RunnerError(f"{label} URL 不能包含凭据、query 或 fragment")


def build_ladder_plan(
    *,
    model_id: str,
    context_window: int,
    input_price: Decimal,
    output_price: Decimal,
    max_cost: Decimal,
    max_request_bytes: int,
    low_targets: tuple[int, ...] = LOW_TARGETS,
    token_estimator: Callable[[str], int] | None = None,
) -> LadderPlan:
    if context_window <= 0 or max_request_bytes <= 0 or not low_targets:
        raise RunnerError("上下文窗口、请求体上限与阶梯必须是正整数")
    if any(target <= 0 for target in low_targets):
        raise RunnerError("上下文阶梯必须是正整数")
    if max(low_targets) >= context_window:
        raise RunnerError("context window 必须大于最大固定阶梯")
    estimator = token_estimator or _build_litellm_estimator(model_id)
    specs: list[tuple[str, str, int, Decimal | None, int, str | None]] = []
    for target in low_targets:
        specs.append((f"cold-{target}", "cold", target, None, target, None))
    previous = 0
    for target in low_targets:
        specs.append((f"multi-{target}", "multi_turn", target, None, target - previous, None))
        previous = target
    for ratio in WINDOW_RATIOS:
        target = int(Decimal(context_window) * ratio)
        specs.append((f"window-{int(ratio * 100)}", "window", target, ratio, target, None))
    managed_seed_target = low_targets[-1]
    managed_trim_target = int(Decimal(context_window) * MANAGED_TRIM_RATIO)
    managed_reserve = min(5_000, max(1, int(Decimal(context_window) * Decimal("0.05"))))
    managed_current_target = int(Decimal(context_window) * MANAGED_TARGET_RATIO) - managed_reserve
    if managed_current_target <= 0:
        raise RunnerError("context window 太小，无法生成裁剪验收阶梯")
    specs.extend(
        [
            ("managed-seed", "managed_multi", managed_seed_target, None, managed_seed_target, None),
            (
                "managed-trim-90",
                "managed_multi",
                managed_trim_target,
                MANAGED_TRIM_RATIO,
                managed_current_target,
                "trimmed",
            ),
        ]
    )

    multi_canaries: dict[int, dict[str, str]] = {}
    stages: list[StagePlan] = []
    for index, (case_id, track, target, ratio, prompt_target, expected_management_status) in enumerate(specs):
        canaries = _canaries_for_stage(case_id, index)
        if track == "multi_turn":
            multi_canaries[target] = canaries
            expected = (
                {
                    "early": multi_canaries[low_targets[0]]["early"],
                    "middle": multi_canaries[low_targets[len(low_targets) // 2]]["middle"],
                    "recent": canaries["recent"],
                }
                if target == low_targets[-1]
                else {}
            )
            expected_sources = (
                {
                    "early": f"multi-{low_targets[0]}",
                    "middle": f"multi-{low_targets[len(low_targets) // 2]}",
                    "recent": case_id,
                }
                if expected
                else {}
            )
        elif track == "managed_multi":
            expected = canaries if expected_management_status else {}
            expected_sources = {position: case_id for position in expected}
        else:
            expected = canaries
            expected_sources = {position: case_id for position in expected}
        prompt, estimated = _fit_prompt(
            target_tokens=prompt_target,
            case_id=case_id,
            canaries=canaries,
            expected=expected,
            expected_sources=expected_sources,
            estimator=estimator,
        )
        if track == "multi_turn":
            conversation_key = "multi-turn"
        elif track == "managed_multi":
            conversation_key = "managed-trim"
        else:
            conversation_key = case_id
        provisional = StagePlan(
            case_id=case_id,
            track=track,
            target_context_tokens=target,
            ratio=ratio,
            conversation_key=conversation_key,
            prompt=prompt,
            expected_canaries=expected,
            local_estimated_prompt_tokens=estimated,
            request_bytes=0,
            planned_input_tokens=target,
            prompt_hash=_sha256(prompt),
            expected_management_status=expected_management_status,
        )
        request_bytes = len(
            json.dumps(
                build_chat_payload(provisional, model_id, "00000000-0000-0000-0000-000000000000"),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if request_bytes > max_request_bytes:
            raise RunnerError(f"请求体超过上限: {case_id}")
        stages.append(
            StagePlan(
                **{
                    **provisional.__dict__,
                    "request_bytes": request_bytes,
                }
            )
        )

    projected_input = sum(stage.planned_input_tokens for stage in stages)
    projected_output = len(stages) * MAX_OUTPUT_TOKENS
    projected_cost = (Decimal(projected_input) * input_price + Decimal(projected_output) * output_price) / Decimal(
        1_000_000
    )
    if projected_cost > max_cost:
        raise RunnerError(f"预计成本 {_decimal_text(projected_cost)} USD 超过上限 {_decimal_text(max_cost)} USD")
    return LadderPlan(
        model_id=model_id,
        context_window=context_window,
        stages=tuple(stages),
        projected_input_tokens=projected_input,
        projected_output_tokens=projected_output,
        projected_cost_usd=projected_cost,
    )


def _build_litellm_estimator(model_id: str) -> Callable[[str], int]:
    try:
        from litellm import token_counter
    except ImportError as error:
        raise RunnerError("缺少 litellm，无法生成上下文阶梯") from error

    def estimate(text: str) -> int:
        try:
            return int(token_counter(model=model_id, text=text))
        except Exception as error:  # LiteLLM 对未知模型可能抛出不同异常。
            raise RunnerError("模型 tokenizer 不可用，无法安全估算上下文") from error

    return estimate


def _canaries_for_stage(case_id: str, index: int) -> dict[str, str]:
    return {
        position: hashlib.sha256(f"{case_id}:{index}:{position}".encode()).hexdigest()[:12]
        for position in ("early", "middle", "recent")
    }


def _fit_prompt(
    *,
    target_tokens: int,
    case_id: str,
    canaries: dict[str, str],
    expected: dict[str, str],
    expected_sources: dict[str, str],
    estimator: Callable[[str], int],
) -> tuple[str, int]:
    def compose(records: int) -> str:
        filler = _neutral_records(case_id, records)
        insertion_points = (0, len(filler) // 2, len(filler))
        marked = list(filler)
        offset = 0
        for position, point in zip(("early", "middle", "recent"), insertion_points):
            marked.insert(point + offset, f"轮次{case_id}标记{position}:{canaries[position]}")
            offset += 1
        if expected:
            requested = "、".join(
                f"{expected_sources[position]} 的 {position}" for position in ("early", "middle", "recent")
            )
            query = f"请仅按 early,middle,recent 顺序返回以下三项标记值：{requested}。"
        else:
            query = "请仅回复：确认。"
        return "\n".join([*marked, query])

    low, high = 0, 1
    prompt = compose(high)
    estimate = estimator(prompt)
    while estimate < target_tokens:
        low = high
        high *= 2
        prompt = compose(high)
        estimate = estimator(prompt)
        if high > max(1_000_000, target_tokens * 20):
            raise RunnerError("无法在安全范围内生成目标上下文")
    while low + 1 < high:
        middle = (low + high) // 2
        candidate = compose(middle)
        candidate_estimate = estimator(candidate)
        if candidate_estimate < target_tokens:
            low = middle
        else:
            high = middle
            prompt = candidate
            estimate = candidate_estimate
    return prompt, estimate


def _neutral_records(case_id: str, count: int) -> list[str]:
    return [f"中性记录{i:06d}:{hashlib.sha256(f'{case_id}:{i}'.encode()).hexdigest()[:16]}" for i in range(count)]


def build_chat_payload(stage: StagePlan, model_id: str, conversation_id: str) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "message": stage.prompt,
        "conversation_id": conversation_id,
        "stream": True,
        "options": {
            "use_reasoning": False,
            "disable_tools": True,
            "max_tokens": MAX_OUTPUT_TOKENS,
        },
        "file_ids": [],
    }


def consume_sse_lines(
    stage: StagePlan,
    lines: Iterable[bytes | str],
    *,
    observed_times: Iterator[float] | None = None,
    manifest: CleanupManifest | None = None,
    max_duration_seconds: float | None = None,
    deadline_clock_ms: Callable[[], float] | None = None,
) -> StageResult:
    parser = SSEParser()
    started_at_ms = 0.0 if observed_times is not None else time.perf_counter() * 1000
    metrics = SSEFlowMetrics(started_at_ms=started_at_ms)
    answer_parts: list[str] = []
    done = False
    rounds = 0
    tools = 0
    errors = 0
    agent_run_id: str | None = None
    assistant_message_id: str | None = None
    last_observed = started_at_ms
    stop_reasons: list[str] = []
    deadline_clock = deadline_clock_ms or (lambda: time.perf_counter() * 1000)
    deadline_started_ms = deadline_clock() if max_duration_seconds is not None else None
    for raw_line in lines:
        if (
            max_duration_seconds is not None
            and deadline_started_ms is not None
            and deadline_clock() - deadline_started_ms >= max_duration_seconds * 1000
        ):
            _append_once(stop_reasons, "timeout")
            break
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        event = parser.feed_line(line)
        if event is None:
            continue
        last_observed = next(observed_times) if observed_times is not None else time.perf_counter() * 1000
        if event.done:
            done = True
            break
        payload = event.payload or {}
        metrics.observe_envelope(payload, observed_at_ms=last_observed)
        run_id, trace_id = extract_agent_trace_ids(payload)
        if manifest is not None:
            manifest.add_agent_trace(run_id, trace_id)
        observed_run_id, observed_message_id = _run_started_identifiers(payload)
        if observed_run_id:
            if not _SAFE_AGENT_ID.fullmatch(observed_run_id):
                _append_once(stop_reasons, "invalid_run_id")
            else:
                if agent_run_id and agent_run_id != observed_run_id:
                    _append_once(stop_reasons, "multiple_run_ids")
                agent_run_id = observed_run_id
        if observed_message_id:
            if not _SAFE_AGENT_ID.fullmatch(observed_message_id):
                _append_once(stop_reasons, "invalid_message_id")
            else:
                if assistant_message_id and assistant_message_id != observed_message_id:
                    _append_once(stop_reasons, "multiple_message_ids")
                assistant_message_id = observed_message_id
        event_type = _agent_event_type(payload)
        abort_stream = False
        if event_type == "step_started":
            rounds += 1
            if rounds > 1:
                _append_once(stop_reasons, "second_round")
                abort_stream = True
        if event_type and event_type.startswith("tool_"):
            tools += 1
            _append_once(stop_reasons, "tool_event")
            abort_stream = True
        if payload.get("chunk_type") == "error" or event_type in {"run_failed", "step_failed"}:
            errors += 1
            _append_once(stop_reasons, "error_frame")
            abort_stream = True
        if payload.get("chunk_type") == "answering":
            data = payload.get("data")
            delta = data.get("delta") if isinstance(data, dict) else None
            if isinstance(delta, str):
                answer_parts.append(delta)
        if abort_stream:
            break

    answer = "".join(answer_parts)
    canary_hits = {label: value in answer for label, value in stage.expected_canaries.items()}
    if not done:
        _append_once(stop_reasons, "incomplete_stream")
    if rounds == 0:
        _append_once(stop_reasons, "missing_round")
    if not agent_run_id:
        _append_once(stop_reasons, "missing_run_id")
    if any(not hit for hit in canary_hits.values()):
        _append_once(stop_reasons, "canary_miss")
    summary = metrics.build_summary(finished_at_ms=max(last_observed, started_at_ms))
    return StageResult(
        case_id=stage.case_id,
        success=not stop_reasons,
        done=done,
        round_count=rounds,
        tool_event_count=tools,
        error_frame_count=errors,
        canary_hits=canary_hits,
        answer_hash=_sha256(answer),
        answer_length=len(answer),
        metrics=summary,
        stop_reasons=tuple(stop_reasons),
        agent_run_id=agent_run_id,
        assistant_message_id=assistant_message_id,
    )


def _agent_event_type(payload: dict[str, Any]) -> str | None:
    if payload.get("chunk_type") != "agent_event":
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    nested = data.get("event")
    candidate = nested if isinstance(nested, dict) else data
    value = candidate.get("type")
    return value if isinstance(value, str) else None


def _run_started_identifiers(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    if payload.get("chunk_type") != "agent_event":
        return None, None
    data = payload.get("data")
    candidates = [data.get("event") if isinstance(data, dict) else None, data, payload]
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("type") != "run_started":
            continue
        run_id = candidate.get("run_id")
        message_id = candidate.get("message_id")
        return (
            run_id if isinstance(run_id, str) and run_id else None,
            message_id if isinstance(message_id, str) and message_id else None,
        )
    return None, None


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def build_loki_query_range_url(
    loki_url: str,
    run_id: str,
    *,
    start_ns: int,
    end_ns: int,
) -> str:
    _validate_monitoring_url(loki_url, "Loki")
    if not _SAFE_AGENT_ID.fullmatch(run_id):
        raise RunnerError("agent run_id 格式无效")
    if start_ns < 0 or end_ns <= start_ns:
        raise RunnerError("Loki 查询时间窗口无效")
    parsed = urlsplit(loki_url)
    base_path = parsed.path.rstrip("/")
    if base_path.endswith("/loki/api/v1/query_range"):
        endpoint_path = base_path
    else:
        endpoint_path = f"{base_path}/loki/api/v1/query_range"
    query = f'{{container="fusion-api"}} |= "LLM_ROUND_CONTEXT" |= "\\"run_id\\":\\"{run_id}\\""'
    params = urllib.parse.urlencode(
        {
            "query": query,
            "start": str(start_ns),
            "end": str(end_ns),
            "direction": "forward",
            "limit": "20",
        }
    )
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, params, ""))


class LokiRoundContextClient:
    """只读取单个 agent run 对应的脱敏 round context 日志。"""

    def query_range(
        self,
        loki_url: str,
        run_id: str,
        *,
        start_ns: int,
        end_ns: int,
        timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        if timeout_seconds <= 0:
            raise RunnerError("Loki timeout 必须为正数")
        url = build_loki_query_range_url(loki_url, run_id, start_ns=start_ns, end_ns=end_ns)
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "fusion-context-ladder/1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read(_LOKI_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _LOKI_MAX_RESPONSE_BYTES:
                raise RunnerError("Loki 响应超过安全上限")
            payload = json.loads(raw.decode("utf-8"))
        except RunnerError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RunnerError("Loki 查询不可用") from error
        return _parse_loki_round_payloads(payload, run_id)


def _parse_loki_round_payloads(payload: Any, run_id: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise RunnerError("Loki 响应失败")
    data = payload.get("data")
    result = data.get("result") if isinstance(data, dict) else None
    if not isinstance(result, list):
        raise RunnerError("Loki 响应缺少 result")
    matches: list[dict[str, Any]] = []
    for stream in result:
        values = stream.get("values") if isinstance(stream, dict) else None
        if not isinstance(values, list):
            continue
        for sample in values:
            if not isinstance(sample, list) or len(sample) != 2 or not isinstance(sample[1], str):
                continue
            for decoded in _decode_round_log_line(sample[1]):
                if decoded.get("event") == "llm_round_context" and decoded.get("run_id") == run_id:
                    matches.append(decoded)
    return matches


def _decode_round_log_line(line: str) -> list[dict[str, Any]]:
    candidates = [line]
    try:
        outer = json.loads(line)
    except json.JSONDecodeError:
        outer = None
    if isinstance(outer, dict):
        candidates.extend(
            value for key, value in outer.items() if key in {"log", "message", "msg"} and isinstance(value, str)
        )
    decoded: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for candidate in candidates:
        marker = "LLM_ROUND_CONTEXT"
        offset = candidate.find(marker)
        if offset < 0:
            continue
        tail = candidate[offset + len(marker) :].lstrip()
        try:
            value, _ = decoder.raw_decode(tail)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            decoded.append(value)
    return decoded


def _lookup_round_context(
    client: LokiRoundContextClient,
    loki_url: str,
    run_id: str,
    *,
    start_ns: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any] | None, str | None]:
    for attempt in range(_LOKI_MAX_ATTEMPTS):
        try:
            matches = client.query_range(
                loki_url,
                run_id,
                start_ns=max(0, start_ns - 5_000_000_000),
                end_ns=time.time_ns() + 1_000_000_000,
                timeout_seconds=timeout_seconds,
            )
        except RunnerError:
            return None, "loki_unavailable"
        if len(matches) > 1:
            return None, "round_context_duplicate"
        if len(matches) == 1:
            return matches[0], None
        if attempt < _LOKI_MAX_ATTEMPTS - 1:
            time.sleep(0.5 * (attempt + 1))
    return None, "round_context_missing"


_ROUND_CONTEXT_SAFE_FIELDS = (
    "schema_version",
    "usage_scope",
    "round_index",
    "round_kind",
    "model_id",
    "message_count",
    "role_counts",
    "content_part_counts",
    "assistant_tool_call_count",
    "request_tool_definition_count",
    "estimated_prompt_tokens",
    "estimator_method",
    "estimator_status",
    "context_window_tokens",
    "context_window_source",
    "context_window_status",
    "estimated_utilization_ratio",
    "estimated_over_budget",
    "round_prompt_tokens",
    "round_completion_tokens",
    "actual_utilization_ratio",
    "actual_over_budget",
    "first_model_text_delta_ms",
    "total_duration_ms",
    "finish_reason",
    "outcome",
    "error_type",
    "context_management_status",
    "context_management_context_window_tokens",
    "context_management_context_window_source",
    "context_management_context_window_status",
    "context_management_trigger_tokens",
    "context_management_target_tokens",
    "context_management_estimated_tokens_before",
    "context_management_estimated_tokens_after",
    "context_management_removed_turns",
    "context_management_removed_tool_transactions",
    "context_management_removed_messages",
)


def _validate_round_context(
    stage: StagePlan,
    result: StageResult,
    payload: dict[str, Any],
    *,
    model_id: str,
    context_window: int,
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    if payload.get("run_id") != result.agent_run_id:
        reasons.append("round_context_run_mismatch")
    if payload.get("assistant_message_id") != result.assistant_message_id:
        reasons.append("round_context_message_mismatch")
    if payload.get("model_id") != model_id:
        reasons.append("round_context_model_mismatch")
    if payload.get("context_window_tokens") != context_window:
        reasons.append("round_context_window_mismatch")
    if payload.get("context_window_status") != "known":
        reasons.append("round_context_window_not_fresh")
    if payload.get("round_index") != 1 or payload.get("round_kind") != "agent":
        reasons.append("round_context_not_single_agent_round")
    estimated = payload.get("estimated_prompt_tokens")
    if isinstance(estimated, bool) or not isinstance(estimated, int) or estimated <= 0:
        reasons.append("round_context_estimate_missing")
    if payload.get("estimator_status") not in {"success", "reused_context_manager"}:
        reasons.append("round_context_estimator_failed")
    if payload.get("outcome") != "success":
        reasons.append("round_context_outcome_failed")
    if payload.get("request_tool_definition_count") != 0:
        reasons.append("round_context_tools_enabled")
    round_prompt_tokens = payload.get("round_prompt_tokens")
    expected_round_tokens = stage.target_context_tokens
    if stage.expected_management_status:
        expected_round_tokens = payload.get("context_management_estimated_tokens_after")
    allowed_deviation = (
        max(int(expected_round_tokens * 0.1), 3_000)
        if isinstance(expected_round_tokens, int) and not isinstance(expected_round_tokens, bool)
        else 3_000
    )
    if (
        isinstance(round_prompt_tokens, bool)
        or not isinstance(round_prompt_tokens, int)
        or isinstance(expected_round_tokens, bool)
        or not isinstance(expected_round_tokens, int)
        or abs(round_prompt_tokens - expected_round_tokens) > allowed_deviation
    ):
        reasons.append("round_context_target_deviation")
    round_completion_tokens = payload.get("round_completion_tokens")
    if (
        isinstance(round_completion_tokens, bool)
        or not isinstance(round_completion_tokens, int)
        or round_completion_tokens > MAX_OUTPUT_TOKENS
    ):
        reasons.append("round_context_output_cap_failed")
    if stage.expected_management_status:
        _validate_managed_trim_context(stage, payload, reasons)
    safe = {key: payload.get(key) for key in _ROUND_CONTEXT_SAFE_FIELDS}
    safe["target_context_tokens"] = stage.target_context_tokens
    return safe, reasons


def _validate_managed_trim_context(
    stage: StagePlan,
    payload: dict[str, Any],
    reasons: list[str],
) -> None:
    if payload.get("context_management_status") != stage.expected_management_status:
        reasons.append("context_management_status_mismatch")
    before = payload.get("context_management_estimated_tokens_before")
    after = payload.get("context_management_estimated_tokens_after")
    target = payload.get("context_management_target_tokens")
    removed_turns = payload.get("context_management_removed_turns")
    removed_messages = payload.get("context_management_removed_messages")
    allowed_before_deviation = max(int(stage.target_context_tokens * 0.1), 3_000)
    if (
        isinstance(before, bool)
        or not isinstance(before, int)
        or abs(before - stage.target_context_tokens) > allowed_before_deviation
    ):
        reasons.append("context_management_before_target_deviation")
    if (
        isinstance(after, bool)
        or not isinstance(after, int)
        or isinstance(target, bool)
        or not isinstance(target, int)
        or after > target
        or (isinstance(before, int) and not isinstance(before, bool) and after >= before)
    ):
        reasons.append("context_management_after_target_failed")
    if isinstance(removed_turns, bool) or not isinstance(removed_turns, int) or removed_turns < 1:
        reasons.append("context_management_turn_not_removed")
    if isinstance(removed_messages, bool) or not isinstance(removed_messages, int) or removed_messages < 2:
        reasons.append("context_management_messages_not_removed")


def _verify_live_model(client: HttpClient, args: argparse.Namespace) -> None:
    response = client.request_json("GET", join_url(args.target_url, "/api/models/"))
    data = response.data.get("data")
    models = data.get("models") if isinstance(data, dict) else response.data.get("models")
    if not isinstance(models, list):
        raise RunnerError("模型目录响应缺少 models")
    selected = next(
        (item for item in models if isinstance(item, dict) and item.get("modelId") == args.model_id),
        None,
    )
    if selected is None:
        raise RunnerError("模型目录中不存在指定模型")
    live_window = selected.get("contextWindowTokens")
    if not isinstance(live_window, int) or live_window <= 0 or live_window != args.context_window:
        raise RunnerError("模型实时 context window 与显式参数不一致")
    pricing = selected.get("pricing")
    if not isinstance(pricing, dict):
        raise RunnerError("模型目录缺少实时价格")
    if pricing.get("unit") != "USD":
        raise RunnerError("模型价格单位不是 USD")
    try:
        live_input = Decimal(str(pricing["input"]))
        live_output = Decimal(str(pricing["output"]))
    except (KeyError, InvalidOperation) as error:
        raise RunnerError("模型目录价格无效") from error
    if live_input != args.input_usd_per_million or live_output != args.output_usd_per_million:
        raise RunnerError("模型实时价格与显式参数不一致")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    plan = build_ladder_plan(
        model_id=args.model_id,
        context_window=args.context_window,
        input_price=args.input_usd_per_million,
        output_price=args.output_usd_per_million,
        max_cost=args.max_cost_usd,
        max_request_bytes=args.max_request_bytes,
    )
    execution_stages = _execution_stages(plan, args.start_case)
    base = {
        "schema_version": 1,
        "mode": "apply" if args.apply else "dry_run",
        "executed": bool(args.apply),
        "live_model_verified": False,
        "plan": plan.safe_dict(),
        "execution_start_case": execution_stages[0].case_id,
        "monitoring": {
            "prometheus": "planned_not_executed" if not args.apply else "pending",
            "loki": "planned_not_executed" if not args.apply else "pending",
        },
    }
    if not args.apply:
        return {
            **base,
            "status": "planned_not_executed",
            "results": [],
            "stopped": False,
            "stop_reasons": [],
            "perf_run_id": None,
            "account_fingerprint": None,
            "actual_cost_usd": None,
            "resources": {},
            "cleanup": {
                "attempted": False,
                "conversations_deleted": 0,
                "tokens_revoked": 0,
                "account_cleanup_supported": False,
                "account_rows_retained": 0,
                "errors": [],
            },
        }

    client = HttpClient(args.timeout_seconds)
    run_id, email, password = generate_identity()
    manifest = CleanupManifest(run_id=run_id, email=email)
    loki_client = LokiRoundContextClient()
    token: str | None = None
    results: list[dict[str, Any]] = []
    stop_reasons: list[str] = []
    conversation_ids: dict[str, str] = {}
    actual_cost = Decimal("0")
    guard: ResourceGuard | None = None
    cleanup: dict[str, Any]
    setup_completed = False
    account_registration_attempted = False
    active_conversation_id: str | None = None
    active_result: StageResult | None = None
    try:
        try:
            guard = ResourceGuard(
                UrllibPrometheusClient(),
                args.prometheus_url,
                timeout_seconds=min(float(args.timeout_seconds), 10.0),
            )
            baseline_reasons = guard.check()
        except Exception:  # noqa: BLE001 — 监控门禁只输出固定安全原因码。
            baseline_reasons = ["resource:monitoring_unavailable"]
        if baseline_reasons:
            base["monitoring"] = {"prometheus": "fail_closed", "loki": "not_started"}
            stop_reasons.extend(f"baseline:{reason}" for reason in baseline_reasons)
        else:
            base["monitoring"] = {"prometheus": "active", "loki": "active"}
            _verify_live_model(client, args)
            base["live_model_verified"] = True
            account_registration_attempted = True
            token = authenticate(client, args.auth_url, args.client_id, manifest, password)
            setup_completed = True
            for stage in execution_stages:
                reserved_cost = _planned_stage_cost(
                    stage,
                    input_price=args.input_usd_per_million,
                    output_price=args.output_usd_per_million,
                )
                if actual_cost + reserved_cost > args.max_cost_usd:
                    stop_reasons.append(f"{stage.case_id}:cost:insufficient_remaining_budget")
                    break
                conversation_id = conversation_ids.setdefault(stage.conversation_key, str(uuid.uuid4()))
                active_conversation_id = conversation_id
                active_result = None
                manifest.add_conversation(conversation_id)
                payload = build_chat_payload(stage, args.model_id, conversation_id)
                stage_started_ns = time.time_ns()
                try:
                    with client.open_sse(join_url(args.target_url, "/api/chat/send"), payload, token) as response:
                        result = consume_sse_lines(
                            stage,
                            response,
                            manifest=manifest,
                            max_duration_seconds=float(args.timeout_seconds),
                        )
                except (
                    RunnerError,
                    ValueError,
                    UnicodeDecodeError,
                    urllib.error.URLError,
                    TimeoutError,
                    socket.timeout,
                ) as error:
                    result = _failed_stage_result(stage, type(error).__name__)
                active_result = result

                result_reasons = list(result.stop_reasons)
                stage_stop_attempted = False
                if result_reasons:
                    stop_error = _stop_active_stage(
                        client,
                        args.target_url,
                        token,
                        conversation_id,
                        result.assistant_message_id,
                    )
                    stage_stop_attempted = True
                    if stop_error:
                        _append_once(result_reasons, stop_error)
                cost_context: dict[str, Any] | None = None
                if result.agent_run_id:
                    round_payload, lookup_error = _lookup_round_context(
                        loki_client,
                        args.loki_url,
                        result.agent_run_id,
                        start_ns=stage_started_ns,
                        timeout_seconds=min(float(args.timeout_seconds), 10.0),
                    )
                    if lookup_error:
                        _append_once(result_reasons, lookup_error)
                    elif round_payload is not None:
                        safe_context, context_reasons = _validate_round_context(
                            stage,
                            result,
                            round_payload,
                            model_id=args.model_id,
                            context_window=args.context_window,
                        )
                        result_reasons.extend(reason for reason in context_reasons if reason not in result_reasons)
                        result = replace(result, round_context=safe_context)
                        cost_context = safe_context
                actual_cost += _round_cost(
                    cost_context or {},
                    input_price=args.input_usd_per_million,
                    output_price=args.output_usd_per_million,
                    fallback_input_tokens=stage.planned_input_tokens,
                )
                resource_reasons = guard.check()
                result_reasons.extend(reason for reason in resource_reasons if reason not in result_reasons)
                if actual_cost > args.max_cost_usd:
                    _append_once(result_reasons, "cost:actual_budget_exceeded")
                if result_reasons and not stage_stop_attempted:
                    stop_error = _stop_active_stage(
                        client,
                        args.target_url,
                        token,
                        conversation_id,
                        result.assistant_message_id,
                    )
                    if stop_error:
                        _append_once(result_reasons, stop_error)
                result = replace(
                    result,
                    success=not result_reasons,
                    stop_reasons=tuple(result_reasons),
                )
                results.append(result.safe_dict())
                if not result.success:
                    stop_reasons.extend(f"{stage.case_id}:{reason}" for reason in result.stop_reasons)
                    break
                active_conversation_id = None
                active_result = None
    except Exception as error:  # noqa: BLE001 — 输出仅保留异常类型，确保 setup 也有审计结果。
        scope = "execution" if setup_completed else "setup"
        _append_once(stop_reasons, f"{scope}:{_safe_error_type(error)}")
        if setup_completed and active_conversation_id and token:
            stop_error = _stop_active_stage(
                client,
                args.target_url,
                token,
                active_conversation_id,
                active_result.assistant_message_id if active_result is not None else None,
            )
            if stop_error:
                _append_once(stop_reasons, f"execution:{stop_error}")
    finally:
        try:
            cleanup = cleanup_run(client, args.target_url, args.auth_url, token, manifest)
        except Exception:  # noqa: BLE001 — cleanup 失败也只能输出固定原因码。
            cleanup = {"conversations_deleted": 0, "tokens_revoked": 0, "errors": ["cleanup_failed"]}
    return {
        **base,
        "status": "stopped" if stop_reasons else "completed",
        "results": results,
        "stopped": bool(stop_reasons),
        "stop_reasons": stop_reasons,
        "perf_run_id": run_id,
        "account_fingerprint": fingerprint(email),
        "actual_cost_usd": _decimal_text(actual_cost),
        "resources": guard.resources_summary() if guard is not None else {},
        "cleanup": {
            "attempted": True,
            **cleanup,
            "account_cleanup_supported": False,
            "account_rows_retained": 1 if setup_completed else None if account_registration_attempted else 0,
        },
    }


def _execution_stages(plan: LadderPlan, start_case: str) -> tuple[StagePlan, ...]:
    if not start_case:
        return plan.stages
    for index, stage in enumerate(plan.stages):
        if stage.case_id == start_case:
            if any(previous.conversation_key == stage.conversation_key for previous in plan.stages[:index]):
                raise RunnerError("start-case 存在未重放的会话前置依赖")
            return plan.stages[index:]
    raise RunnerError("start-case 不在阶梯计划中")


def _round_cost(
    context: dict[str, Any],
    *,
    input_price: Decimal,
    output_price: Decimal,
    fallback_input_tokens: int,
) -> Decimal:
    raw_input = context.get("round_prompt_tokens")
    if isinstance(raw_input, bool) or not isinstance(raw_input, int) or raw_input < 0:
        raw_input = context.get("estimated_prompt_tokens")
    if isinstance(raw_input, bool) or not isinstance(raw_input, int) or raw_input < 0:
        raw_input = fallback_input_tokens
    raw_output = context.get("round_completion_tokens")
    if isinstance(raw_output, bool) or not isinstance(raw_output, int) or raw_output < 0:
        raw_output = MAX_OUTPUT_TOKENS
    return (Decimal(raw_input) * input_price + Decimal(raw_output) * output_price) / Decimal(1_000_000)


def _planned_stage_cost(stage: StagePlan, *, input_price: Decimal, output_price: Decimal) -> Decimal:
    return _round_cost(
        {},
        input_price=input_price,
        output_price=output_price,
        fallback_input_tokens=stage.planned_input_tokens,
    )


def _stop_active_stage(
    client: HttpClient,
    target_url: str,
    token: str,
    conversation_id: str,
    assistant_message_id: str | None,
) -> str | None:
    """停止仍在运行的 stage；所有失败仅收敛为固定脱敏原因码。"""
    try:
        status_response = client.request_json(
            "GET",
            join_url(target_url, f"/api/chat/stream-status/{urllib.parse.quote(conversation_id, safe='')}"),
            token=token,
        )
        wrapped = status_response.data.get("data")
        if not isinstance(wrapped, dict):
            return "active_stream_stop_failed"
        status = wrapped.get("status")
        if status in {"completed", "cancelled", "error", "not_found"}:
            return None
        if status != "streaming":
            return "active_stream_stop_failed"
        status_message_id = wrapped.get("message_id")
        if assistant_message_id and status_message_id and assistant_message_id != status_message_id:
            return "active_stream_stop_failed"
        message_id = assistant_message_id or status_message_id
        if not isinstance(message_id, str) or not _SAFE_AGENT_ID.fullmatch(message_id):
            return "active_stream_stop_failed"
        stop_response = client.request_json(
            "POST",
            join_url(
                target_url,
                f"/api/chat/stop/{urllib.parse.quote(conversation_id, safe='')}"
                f"?message_id={urllib.parse.quote(message_id, safe='')}",
            ),
            payload={"partial_content": []},
            token=token,
        )
        stop_data = stop_response.data.get("data")
        if not isinstance(stop_data, dict) or stop_data.get("cancelled") is not True:
            return "active_stream_stop_failed"
        return None
    except Exception:  # noqa: BLE001 — 绝不输出 URL、ID、token 或异常正文。
        return "active_stream_stop_failed"


def _safe_error_type(error: Exception) -> str:
    error_type = type(error).__name__
    return error_type if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", error_type) else "Exception"


def _failed_stage_result(stage: StagePlan, error_type: str) -> StageResult:
    reason = "timeout" if error_type in {"TimeoutError", "timeout"} else "request_failed"
    return StageResult(
        case_id=stage.case_id,
        success=False,
        done=False,
        round_count=0,
        tool_event_count=0,
        error_frame_count=0,
        canary_hits={label: False for label in stage.expected_canaries},
        answer_hash=_sha256(""),
        answer_length=0,
        metrics=SSEFlowMetrics(started_at_ms=0).build_summary(finished_at_ms=0),
        stop_reasons=(reason,),
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _decimal_text(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000001")), "f")


def main() -> int:
    try:
        result = execute(build_parser().parse_args())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result["stopped"] or result["cleanup"]["errors"] else 0
    except (RunnerError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
