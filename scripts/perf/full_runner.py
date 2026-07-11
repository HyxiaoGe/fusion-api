#!/usr/bin/env python3
"""Fusion L1-L4 生产全链路压测 runner；所有凭据与正文只存在进程内存。"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.perf.core import CleanupManifest, SSEParser, StopPolicy, extract_agent_trace_ids, percentile
from scripts.perf.http_scenarios import build_l1_scenarios, run_scenario_stage
from scripts.perf.reliability_scenarios import (
    SoakPolicy,
    SoakSample,
    StopAck,
    StreamReadObservation,
    StreamStatusObservation,
    run_concurrent_recovery,
    run_soak,
    run_stop_scenario,
)
from scripts.perf.resource_guard import ResourceGuard, UrllibPrometheusClient
from scripts.perf.runner import (
    DEFAULT_AUTH_URL,
    DEFAULT_CLIENT_ID,
    DEFAULT_TARGET_URL,
    HttpClient,
    JsonResponse,
    RunnerError,
    authenticate,
    generate_identity,
    join_url,
    parse_concurrency,
)
from scripts.perf.sse_metrics import SSEFlowMetrics, summarize_sse_stage

_STAGE_FIELDS = {
    "scenario",
    "kind",
    "concurrency",
    "duration_seconds",
    "elapsed_seconds",
    "cadence_seconds",
    "window_seconds",
    "total",
    "requests",
    "flows",
    "flows_with_output",
    "successful",
    "failed",
    "success_rate",
    "requests_per_second",
    "rps",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "max_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "p95_total_ms",
    "error_rate",
    "timeout_rate",
    "error_frames",
    "output_chunks",
    "reasoning_chunks",
    "answering_chunks",
    "visible_chars",
    "reasoning_visible_chars",
    "answering_visible_chars",
    "approx_tokens",
    "first_output_p50_ms",
    "first_output_p95_ms",
    "first_output_max_ms",
    "chunk_interval_count",
    "chunk_interval_p50_ms",
    "chunk_interval_p95_ms",
    "chunk_interval_max_ms",
    "output_window_p50_ms",
    "output_window_p95_ms",
    "output_window_max_ms",
    "tokens_per_second",
    "tokens_per_second_p50",
    "tokens_per_second_p95",
    "tokens_per_second_max",
    "initial_events",
    "recovered_events",
    "duplicate_events",
    "lost_events",
    "ordering_errors",
    "recovery_latency_ms",
    "recovery_latency_p50_ms",
    "recovery_latency_p95_ms",
    "recovery_latency_max_ms",
    "stop_attempted",
    "cancelled",
    "persistence_verified",
    "stop_attempts",
    "cancelled_count",
    "persistence_verified_count",
    "stop_latency_ms",
    "stop_latency_p50_ms",
    "stop_latency_p95_ms",
    "stop_latency_max_ms",
    "executed_ticks",
    "skipped_ticks",
    "window_count",
    "consecutive_failures",
}


class CapturingLoginClient:
    """拦截登录 refresh token 供 finally 吊销，不把凭据放入 repr 或结果。"""

    def __init__(self, client: HttpClient, manifest: CleanupManifest) -> None:
        self._client = client
        self._manifest = manifest
        self._lock = threading.Lock()
        self._sequence = 0

    def request_json(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> JsonResponse:
        response = self._client.request_json(method, url, payload=payload, token=token)
        refresh_token = response.data.get("refresh_token") if url.endswith("/auth/login") else None
        if isinstance(refresh_token, str) and refresh_token:
            with self._lock:
                self._sequence += 1
                self._manifest.add_refresh_token(f"l1_login_{self._sequence:04d}", refresh_token)
        return response


@dataclass
class LiveFlow:
    metrics: SSEFlowMetrics = field(repr=False)
    total_ms: float
    done: bool
    disconnected: bool
    error_frames: int
    error_code: str | None
    event_ids: tuple[str, ...] = field(repr=False)
    chunk_types: tuple[str, ...]
    message_id: str | None = field(default=None, repr=False)
    partial_blocks: dict[tuple[str, str], str] = field(default_factory=dict, repr=False)

    @property
    def success(self) -> bool:
        summary = self.metrics.build_summary()
        return self.done and bool(summary["output_chunks"]) and not self.error_frames and self.error_code is None

    @property
    def timed_out(self) -> bool:
        return self.error_code in {"TimeoutError", "timeout", "socket.timeout"}

    def observation(self) -> StreamReadObservation:
        return StreamReadObservation(
            event_ids=self.event_ids,
            chunk_types=self.chunk_types,
            message_id=self.message_id,
            done=self.done,
            disconnected=self.disconnected,
            error_frames=self.error_frames,
            error_code=self.error_code,
        )


def extract_run_started_message_id(payload: dict[str, Any]) -> str | None:
    if payload.get("chunk_type") != "agent_event":
        return None
    data = payload.get("data")
    candidates = [data.get("event") if isinstance(data, dict) else None, data, payload]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("type") == "run_started":
            value = candidate.get("message_id")
            return value if isinstance(value, str) and value else None
    return None


def normalize_http_stage(stage: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stage.items() if key in _STAGE_FIELDS and value is not None}


def build_import_payload(
    *,
    run_id: str,
    model_id: str,
    stages: list[dict[str, Any]],
    stopped: bool,
    stop_reasons: list[str],
    cleanup: dict[str, Any],
    resources: dict[str, Any] | None,
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    safe_stages = [normalize_http_stage(stage) for stage in stages]
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "environment": "production",
        "model_id": model_id,
        "status": "stopped" if stopped else "completed",
        "safe_summary": {
            "stages": safe_stages,
            "stopped": stopped,
            "stop_reasons": stop_reasons,
            "cleanup": cleanup,
            "resources": resources,
        },
        "started_at": started_at,
        "finished_at": finished_at,
    }
    from app.schemas.admin_audit import AdminPerformanceRunImport

    AdminPerformanceRunImport.model_validate(payload)
    return payload


def _cleanup_full(
    client: HttpClient,
    target_url: str,
    auth_url: str,
    token: str | None,
    manifest: CleanupManifest,
    *,
    workers: int = 10,
) -> dict[str, Any]:
    """并行执行精确会话删除与 token 吊销；错误只返回去重安全码。"""

    errors: list[str] = []

    def delete_conversation(conversation_id: str) -> bool:
        if not token:
            return False
        try:
            client.request_json(
                "DELETE",
                join_url(target_url, f"/api/chat/conversations/{urllib.parse.quote(conversation_id, safe='')}"),
                token=token,
            )
            return True
        except RunnerError as error:
            return str(error).startswith("HTTP 404:")
        except (ValueError, urllib.error.URLError, TimeoutError, socket.timeout):
            return False

    conversation_ids = sorted(manifest.conversation_ids)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, max(1, len(conversation_ids)))) as executor:
        deleted_results = list(executor.map(delete_conversation, conversation_ids))
    conversations_deleted = sum(deleted_results)
    if conversations_deleted != len(conversation_ids):
        errors.append("conversation_delete_failed")

    def revoke_token(item: tuple[str, str]) -> bool:
        _, refresh_token = item
        try:
            client.request_json(
                "POST",
                join_url(auth_url, "/auth/token/revoke"),
                payload={"refresh_token": refresh_token},
            )
            return True
        except (RunnerError, ValueError, urllib.error.URLError, TimeoutError, socket.timeout, RuntimeError):
            return False

    refresh_tokens = manifest.refresh_tokens()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, max(1, len(refresh_tokens)))) as executor:
        revoked_results = list(executor.map(revoke_token, refresh_tokens))
    tokens_revoked = sum(revoked_results)
    if tokens_revoked != len(refresh_tokens):
        errors.append("token_revoke_failed")
    return {
        "conversations_deleted": conversations_deleted,
        "tokens_revoked": tokens_revoked,
        "errors": list(dict.fromkeys(errors)),
    }


def _chat_payload(run_id: str, conversation_id: str, model_id: str, max_tokens: int, prompt: str) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "message": f"[PERF:{run_id}] {prompt}",
        "conversation_id": conversation_id,
        "stream": True,
        "options": {"max_tokens": max_tokens, "use_reasoning": False},
        "file_ids": [],
    }


def _now_ms() -> float:
    return time.perf_counter() * 1000


def _read_sse_response(
    response,
    manifest: CleanupManifest,
    *,
    disconnect_after_output_chunks: int | None = None,
    known_message_id: str | None = None,
    started_ms: float | None = None,
) -> LiveFlow:
    started_ms = _now_ms() if started_ms is None else started_ms
    metrics = SSEFlowMetrics(started_at_ms=started_ms)
    parser = SSEParser()
    event_ids: list[str] = []
    chunk_types: list[str] = []
    message_id = known_message_id
    partial_blocks: dict[tuple[str, str], str] = {}
    done = False
    disconnected = False
    error_frames = 0
    error_code: str | None = None
    try:
        for raw_line in response:
            event = parser.feed_line(raw_line.decode("utf-8"))
            if event is None:
                continue
            if event.done:
                done = True
                break
            payload = event.payload or {}
            chunk_type = str(payload.get("chunk_type", ""))
            chunk_types.append(chunk_type)
            if event.event_id:
                event_ids.append(event.event_id)
            run_id, trace_id = extract_agent_trace_ids(payload)
            manifest.add_agent_trace(run_id, trace_id)
            message_id = message_id or extract_run_started_message_id(payload)
            metrics.observe_envelope(payload, observed_at_ms=_now_ms())
            data = payload.get("data")
            if chunk_type in {"reasoning", "answering"} and isinstance(data, dict):
                delta = data.get("delta")
                block_id = data.get("block_id")
                if isinstance(delta, str) and isinstance(block_id, str):
                    key = (chunk_type, block_id)
                    partial_blocks[key] = partial_blocks.get(key, "") + delta
            if chunk_type == "error":
                error_frames += 1
                if isinstance(data, dict) and isinstance(data.get("code"), str):
                    error_code = data["code"]
            if disconnect_after_output_chunks is not None:
                output_chunks = int(metrics.build_summary()["output_chunks"] or 0)
                if output_chunks >= disconnect_after_output_chunks:
                    disconnected = True
                    break
    except (ValueError, UnicodeDecodeError, urllib.error.URLError, TimeoutError, socket.timeout) as error:
        error_code = type(error).__name__
    finished_ms = _now_ms()
    metrics.build_summary(finished_at_ms=finished_ms)
    return LiveFlow(
        metrics=metrics,
        total_ms=round(finished_ms - started_ms, 2),
        done=done,
        disconnected=disconnected,
        error_frames=error_frames,
        error_code=error_code,
        event_ids=tuple(event_ids),
        chunk_types=tuple(chunk_types),
        message_id=message_id,
        partial_blocks=partial_blocks,
    )


def _run_new_flow(
    client: HttpClient,
    target_url: str,
    token: str,
    model_id: str,
    manifest: CleanupManifest,
    *,
    max_tokens: int,
    prompt: str,
    conversation_id: str | None = None,
    disconnect_after_output_chunks: int | None = None,
) -> LiveFlow:
    conversation_id = conversation_id or str(uuid.uuid4())
    manifest.add_conversation(conversation_id)
    payload = _chat_payload(
        manifest.run_id,
        conversation_id,
        model_id,
        max_tokens,
        _with_cache_bypass_nonce(prompt, conversation_id),
    )
    started_ms = _now_ms()
    try:
        with client.open_sse(join_url(target_url, "/api/chat/send"), payload, token) as response:
            return _read_sse_response(
                response,
                manifest,
                disconnect_after_output_chunks=disconnect_after_output_chunks,
                started_ms=started_ms,
            )
    except (RunnerError, urllib.error.URLError, TimeoutError, socket.timeout) as error:
        started = _now_ms()
        metrics = SSEFlowMetrics(started_at_ms=started)
        metrics.build_summary(finished_at_ms=started)
        return LiveFlow(metrics, 0.0, False, False, 0, type(error).__name__, (), ())


def _with_cache_bypass_nonce(prompt: str, conversation_id: str) -> str:
    """为每条压测会话注入唯一标识，避免并发生成命中上游 prompt cache。"""

    return f"{prompt}\n\n内部压测标识（无需复述）：{conversation_id}"


def _run_sse_stage(
    client: HttpClient,
    target_url: str,
    token: str,
    model_id: str,
    manifest: CleanupManifest,
    *,
    scenario: str,
    concurrency: int,
    max_tokens: int,
    prompt: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _run_new_flow,
                client,
                target_url,
                token,
                model_id,
                manifest,
                max_tokens=max_tokens,
                prompt=prompt,
            )
            for _ in range(concurrency)
        ]
        flows = [future.result() for future in futures]
    successful = sum(flow.success for flow in flows)
    timeouts = sum(flow.timed_out for flow in flows)
    metrics = summarize_sse_stage(flow.metrics for flow in flows)
    return {
        "scenario": scenario,
        "kind": "sse",
        "concurrency": concurrency,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "successful": successful,
        "failed": len(flows) - successful,
        "success_rate": round(successful / len(flows), 4),
        "timeout_rate": round(timeouts / len(flows), 4),
        "p50_ttft_ms": metrics["first_output_p50_ms"],
        "p95_ttft_ms": metrics["first_output_p95_ms"],
        "p95_total_ms": percentile([flow.total_ms for flow in flows], 0.95),
        "error_frames": sum(flow.error_frames for flow in flows),
        **metrics,
    }


class _ReliabilityClient:
    def __init__(
        self,
        client: HttpClient,
        target_url: str,
        token: str,
        model_id: str,
        manifest: CleanupManifest,
        *,
        max_tokens: int,
    ) -> None:
        self.client = client
        self.target_url = target_url
        self.token = token
        self.model_id = model_id
        self.manifest = manifest
        self.max_tokens = max_tokens
        self.states: dict[str, LiveFlow] = {}
        self.expected_partials: dict[str, list[tuple[str, str]]] = {}
        self._lock = threading.Lock()

    def initial_read(self, conversation_id: str) -> StreamReadObservation:
        flow = _run_new_flow(
            self.client,
            self.target_url,
            self.token,
            self.model_id,
            self.manifest,
            max_tokens=self.max_tokens,
            prompt="不要使用工具。连续写一篇约 1500 个中文字符的科普文章，不要列提纲。",
            conversation_id=conversation_id,
            disconnect_after_output_chunks=2,
        )
        with self._lock:
            self.states[conversation_id] = flow
        return flow.observation()

    def read_status(self, conversation_id: str) -> StreamStatusObservation:
        response = self.client.request_json(
            "GET",
            join_url(self.target_url, f"/api/chat/stream-status/{urllib.parse.quote(conversation_id, safe='')}"),
            token=self.token,
        )
        data = response.data.get("data")
        if not isinstance(data, dict):
            return StreamStatusObservation(status="invalid_response", error_code="invalid_response")
        return StreamStatusObservation(
            status=str(data.get("status", "invalid_response")),
            message_id=data.get("message_id") if isinstance(data.get("message_id"), str) else None,
            last_entry_id=data.get("last_entry_id") if isinstance(data.get("last_entry_id"), str) else None,
            stream_mode=data.get("stream_mode") if isinstance(data.get("stream_mode"), str) else None,
        )

    def reconnect_read(self, conversation_id: str, cursor: str) -> StreamReadObservation:
        known = self.states.get(conversation_id)
        url = join_url(
            self.target_url,
            f"/api/chat/stream/{urllib.parse.quote(conversation_id, safe='')}?last_entry_id={urllib.parse.quote(cursor, safe='')}",
        )
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {self.token}",
                "User-Agent": "fusion-perf-runner/1",
            },
            method="GET",
        )
        started_ms = _now_ms()
        with urllib.request.urlopen(request, timeout=self.client.timeout_seconds) as response:
            flow = _read_sse_response(
                response,
                self.manifest,
                known_message_id=known.message_id if known else None,
                started_ms=started_ms,
            )
        return flow.observation()

    def stop_stream(self, conversation_id: str, message_id: str) -> StopAck:
        flow = self.states.get(conversation_id)
        partial_content: list[dict[str, Any]] = []
        if flow:
            for (chunk_type, _), value in flow.partial_blocks.items():
                if not value:
                    continue
                if chunk_type == "answering":
                    partial_content.append({"type": "text", "text": value})
                else:
                    partial_content.append({"type": "thinking", "thinking": value})
        expected = [
            (str(block["type"]), str(block.get("text") or block.get("thinking") or "")) for block in partial_content
        ]
        with self._lock:
            self.expected_partials[conversation_id] = expected
        response = self.client.request_json(
            "POST",
            join_url(
                self.target_url,
                f"/api/chat/stop/{urllib.parse.quote(conversation_id, safe='')}?message_id={urllib.parse.quote(message_id, safe='')}",
            ),
            payload={"partial_content": partial_content},
            token=self.token,
        )
        data = response.data.get("data")
        return StopAck(cancelled=bool(isinstance(data, dict) and data.get("cancelled")))

    def verify_persisted(self, conversation_id: str, message_id: str) -> bool:
        expected = self.expected_partials.get(conversation_id)
        if not expected:
            return False
        observed_versions: list[list[tuple[str, str]]] = []
        for _ in range(2):
            status = self.read_status(conversation_id)
            if status.status != "cancelled":
                return False
            response = self.client.request_json(
                "GET",
                join_url(
                    self.target_url,
                    f"/api/chat/conversations/{urllib.parse.quote(conversation_id, safe='')}",
                ),
                token=self.token,
            )
            data = response.data.get("data")
            messages = data.get("messages") if isinstance(data, dict) else None
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict) and message.get("id") == message_id and message.get("content"):
                        blocks = []
                        for block in message["content"]:
                            if not isinstance(block, dict):
                                continue
                            block_type = str(block.get("type", ""))
                            value = block.get("text") if block_type == "text" else block.get("thinking")
                            if isinstance(value, str):
                                blocks.append((block_type, value))
                        observed_versions.append(blocks)
                        break
            time.sleep(0.5)
        return len(observed_versions) == 2 and observed_versions[0] == expected and observed_versions[1] == expected

    def quiesce(self) -> list[str]:
        """停止异常路径仍在生成的流，并确认不再处于 streaming。"""

        errors: list[str] = []
        for conversation_id, flow in list(self.states.items()):
            try:
                status = self.read_status(conversation_id)
                message_id = flow.message_id or status.message_id
                if status.status == "streaming" and message_id:
                    ack = self.stop_stream(conversation_id, message_id)
                    if not ack.cancelled or self.read_status(conversation_id).status == "streaming":
                        errors.append("active_stream_stop_failed")
            except Exception:  # noqa: BLE001 - finally 清理必须收敛为安全错误码
                errors.append("active_stream_stop_failed")
        return list(dict.fromkeys(errors))


def _recovery_stage(
    reliability: _ReliabilityClient,
    manifest: CleanupManifest,
    concurrency: int,
) -> dict[str, Any]:
    refs = [str(uuid.uuid4()) for _ in range(concurrency)]
    for ref in refs:
        manifest.add_conversation(ref)
    batch = run_concurrent_recovery(
        refs,
        initial_read=reliability.initial_read,
        read_status=reliability.read_status,
        reconnect_read=reliability.reconnect_read,
        max_workers=concurrency,
    )
    durations = [case.duration_ms for case in batch.cases]
    return {
        "scenario": "disconnect_reconnect",
        "kind": "recovery",
        "concurrency": concurrency,
        "total": batch.total,
        "successful": batch.successful,
        "failed": batch.failed,
        "success_rate": round(batch.successful / batch.total, 4) if batch.total else 0,
        "error_rate": round(batch.error_rate, 4),
        "initial_events": sum(case.initial_events for case in batch.cases),
        "recovered_events": sum(case.recovered_events for case in batch.cases),
        "duplicate_events": sum(case.duplicate_events for case in batch.cases),
        "lost_events": batch.lost_events,
        "ordering_errors": batch.ordering_errors,
        "error_frames": sum(case.error_frames for case in batch.cases),
        "recovery_latency_p50_ms": percentile(durations, 0.50),
        "recovery_latency_p95_ms": percentile(durations, 0.95),
        "recovery_latency_max_ms": round(max(durations), 2) if durations else 0,
        "_reason_counts": dict(batch.reason_counts),
    }


def _stop_stage(reliability: _ReliabilityClient, manifest: CleanupManifest, concurrency: int) -> dict[str, Any]:
    refs = [str(uuid.uuid4()) for _ in range(concurrency)]
    for ref in refs:
        manifest.add_conversation(ref)

    def execute(index: int, ref: str):
        return run_stop_scenario(
            ref,
            initial_read=reliability.initial_read,
            read_status=reliability.read_status,
            stop_stream=reliability.stop_stream,
            read_status_after_stop=reliability.read_status,
            verify_persisted=reliability.verify_persisted,
            case_id=f"stop-{index + 1}",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        outcomes = list(executor.map(lambda pair: execute(*pair), enumerate(refs)))
    successful = sum(outcome.success for outcome in outcomes)
    durations = [outcome.duration_ms for outcome in outcomes]
    return {
        "scenario": "stop_generation",
        "kind": "stop",
        "concurrency": concurrency,
        "total": len(outcomes),
        "successful": successful,
        "failed": len(outcomes) - successful,
        "success_rate": round(successful / len(outcomes), 4),
        "error_rate": round((len(outcomes) - successful) / len(outcomes), 4),
        "stop_attempts": sum(outcome.stop_attempted for outcome in outcomes),
        "cancelled_count": sum(outcome.cancelled for outcome in outcomes),
        "persistence_verified_count": sum(outcome.persistence_verified is True for outcome in outcomes),
        "error_frames": sum(outcome.error_frames for outcome in outcomes),
        "stop_latency_p50_ms": percentile(durations, 0.50),
        "stop_latency_p95_ms": percentile(durations, 0.95),
        "stop_latency_max_ms": round(max(durations), 2) if durations else 0,
    }


def _soak_stage(
    args, client, token, manifest, resource_guard: ResourceGuard | None
) -> tuple[dict[str, Any], list[str]]:
    window_updates: list[dict[str, Any]] = []

    def execute_tick(_slot: int) -> SoakSample:
        stage = _run_sse_stage(
            client,
            args.target_url,
            token,
            args.model_id,
            manifest,
            scenario="soak_tick",
            concurrency=args.soak_concurrency,
            max_tokens=args.short_max_tokens,
            prompt="不要使用工具。请用一句简短中文回答：水的化学式是什么？",
        )
        return SoakSample(
            latency_ms=float(stage.get("p95_total_ms") or 0),
            success=stage["failed"] == 0 and stage["error_frames"] == 0,
            timed_out=bool(stage["timeout_rate"]),
            error_code=(None if stage["failed"] == 0 else "timeout" if stage["timeout_rate"] else "sse_flow_failed"),
            requests=stage["flows"],
            failures=stage["failed"],
            timeouts=round(stage["timeout_rate"] * stage["flows"]),
        )

    def on_window(window) -> None:
        safe = window.to_safe_dict()
        window_updates.append(safe)
        _progress("l4_window", {"index": safe["index"], "samples": safe["samples"], "failed": safe["failed"]})

    policy = SoakPolicy(
        duration_seconds=args.soak_duration_seconds,
        cadence_seconds=args.soak_cadence_seconds,
        window_seconds=300,
        min_samples=5,
        max_error_rate=0.05,
        max_timeout_rate=0.05,
        max_consecutive_failures=2,
        max_p95_ms=args.soak_max_p95_ms,
    )
    result = run_soak(
        execute_tick,
        policy=policy,
        on_window=on_window,
        hard_stop=resource_guard.check if resource_guard is not None else None,
    )
    windows = result.windows
    total_samples = sum(window.samples for window in windows)
    successful = sum(window.successful for window in windows)
    stage = {
        "scenario": "steady_state_30m",
        "kind": "soak",
        "concurrency": args.soak_concurrency,
        "duration_seconds": args.soak_duration_seconds,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "cadence_seconds": args.soak_cadence_seconds,
        "window_seconds": 300,
        "total": total_samples,
        "successful": successful,
        "failed": total_samples - successful,
        "success_rate": round(successful / total_samples, 4) if total_samples else 0,
        "error_rate": round((total_samples - successful) / total_samples, 4) if total_samples else 0,
        "timeout_rate": round(sum(window.timeouts for window in windows) / total_samples, 4) if total_samples else 0,
        "p50_ms": percentile([window.p50_ms for window in windows], 0.50),
        "p95_ms": percentile([window.p95_ms for window in windows], 0.95),
        "max_ms": round(max((window.max_ms for window in windows), default=0), 2),
        "executed_ticks": result.executed_ticks,
        "skipped_ticks": result.skipped_ticks,
        "window_count": len(windows),
        "consecutive_failures": max((window.peak_consecutive_failures for window in windows), default=0),
    }
    return stage, list(result.stop_reasons)


def _progress(stage: str, data: dict[str, Any]) -> None:
    print(json.dumps({"stage": stage, **data}, ensure_ascii=False), file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fusion 生产 L1-L4 全链路压测")
    parser.add_argument("--target-url", default=DEFAULT_TARGET_URL)
    parser.add_argument("--auth-url", default=DEFAULT_AUTH_URL)
    parser.add_argument("--client-id", default=DEFAULT_CLIENT_ID)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--l1-concurrency", type=parse_concurrency, default=parse_concurrency("10,25,50"))
    parser.add_argument("--short-sse-concurrency", type=parse_concurrency, default=parse_concurrency("1,3,5,10"))
    parser.add_argument("--long-sse-concurrency", type=parse_concurrency, default=parse_concurrency("1,3,5"))
    parser.add_argument("--l3-concurrency", type=parse_concurrency, default=parse_concurrency("1,3,5"))
    parser.add_argument("--requests-per-stage", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--short-max-tokens", type=int, default=64)
    parser.add_argument("--long-max-tokens", type=int, default=512)
    parser.add_argument("--soak-duration-seconds", type=float, default=1800)
    parser.add_argument("--soak-cadence-seconds", type=float, default=60)
    parser.add_argument("--soak-concurrency", type=int, default=2)
    parser.add_argument("--soak-max-p95-ms", type=float, default=30000)
    parser.add_argument("--prometheus-url")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-production", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    target_host = (urllib.parse.urlparse(args.target_url).hostname or "").lower().rstrip(".")
    auth_host = (urllib.parse.urlparse(args.auth_url).hostname or "").lower().rstrip(".")
    production_requested = target_host == "fusion.seanfield.org" or auth_host == "auth.seanfield.org"
    if production_requested and not args.confirm_production:
        raise RunnerError("生产域名压测必须显式确认")
    if args.confirm_production and production_requested:
        if target_host != "fusion.seanfield.org" or auth_host != "auth.seanfield.org":
            raise RunnerError("生产压测只允许预期的 Fusion 与 Auth 域名组合")
        reviewed_caps = (
            ("L1", args.l1_concurrency, 50),
            ("短 SSE", args.short_sse_concurrency, 10),
            ("长 SSE", args.long_sse_concurrency, 5),
            ("L3", args.l3_concurrency, 5),
        )
        for label, levels, maximum in reviewed_caps:
            if not levels or max(levels) > maximum:
                raise RunnerError(f"{label} 生产并发超过已审查上限 {maximum}")
        if args.soak_concurrency > 2:
            raise RunnerError("L4 生产并发超过已审查上限 2")
        if not args.prometheus_url:
            raise RunnerError("生产全链路压测必须配置 Prometheus 资源硬门禁")
    positive = (
        args.requests_per_stage,
        args.timeout_seconds,
        args.short_max_tokens,
        args.long_max_tokens,
        args.soak_duration_seconds,
        args.soak_cadence_seconds,
        args.soak_concurrency,
    )
    if any(value <= 0 for value in positive):
        raise RunnerError("压测数量、时长和并发必须为正数")
    if production_requested and args.soak_duration_seconds != 1800:
        raise RunnerError("完整生产稳态压测必须执行 1800 秒")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    client = HttpClient(args.timeout_seconds)
    resource_guard = (
        ResourceGuard(UrllibPrometheusClient(), args.prometheus_url, timeout_seconds=5) if args.prometheus_url else None
    )
    run_id, email, password = generate_identity()
    manifest = CleanupManifest(run_id=run_id, email=email)
    stages: list[dict[str, Any]] = []
    stop_reasons: list[str] = []
    token: str | None = None
    reliability: _ReliabilityClient | None = None
    quiesce_errors: list[str] = []
    try:
        if resource_guard is not None:
            stop_reasons.extend(resource_guard.check())
        if stop_reasons:
            raise RunnerError("生产资源门禁不可用")
        token = authenticate(client, args.auth_url, args.client_id, manifest, password)
        seed = _run_new_flow(
            client,
            args.target_url,
            token,
            args.model_id,
            manifest,
            max_tokens=32,
            prompt="不要使用工具。请用一句简短中文回答：1+1 等于多少？",
        )
        if not seed.success:
            stop_reasons.append("seed_sse_failed")
        conversation_id = next(iter(manifest.conversation_ids), None)

        if not stop_reasons:
            scenarios = build_l1_scenarios(
                target_url=args.target_url,
                auth_url=args.auth_url,
                email=email,
                password=password,
                client_id=args.client_id,
                access_token=token,
                conversation_id=conversation_id,
            )
            login_client = CapturingLoginClient(client, manifest)
            for name, scenario in scenarios.items():
                scenario_client = login_client if name == "auth_login" else client
                scenario_stages = 0
                for concurrency in args.l1_concurrency:
                    stage, consecutive = run_scenario_stage(
                        scenario_client,
                        scenario,
                        concurrency=concurrency,
                        requests=args.requests_per_stage,
                    )
                    stages.append(normalize_http_stage(stage))
                    scenario_stages += 1
                    reasons = StopPolicy().evaluate(stage, consecutive_failures=consecutive)
                    stop_reasons.extend(f"{name}:{reason}" for reason in reasons)
                    if resource_guard is not None:
                        stop_reasons.extend(resource_guard.check())
                    if stop_reasons:
                        break
                _progress("l1", {"scenario": name, "stages": scenario_stages, "stopped": bool(stop_reasons)})
                if stop_reasons:
                    break

        if not stop_reasons:
            for concurrency in args.short_sse_concurrency:
                stage = _run_sse_stage(
                    client,
                    args.target_url,
                    token,
                    args.model_id,
                    manifest,
                    scenario="sse_short",
                    concurrency=concurrency,
                    max_tokens=args.short_max_tokens,
                    prompt="不要使用工具。请用两句简短中文说明为什么天空看起来是蓝色。",
                )
                stages.append(normalize_http_stage(stage))
                _progress("l2_short", {"concurrency": concurrency, "successful": stage["successful"]})
                if stage["failed"] or stage["error_frames"]:
                    stop_reasons.append("sse_short_failed")
                if resource_guard is not None:
                    stop_reasons.extend(resource_guard.check())
                if stop_reasons:
                    break

        if not stop_reasons:
            for concurrency in args.long_sse_concurrency:
                stage = _run_sse_stage(
                    client,
                    args.target_url,
                    token,
                    args.model_id,
                    manifest,
                    scenario="sse_long",
                    concurrency=concurrency,
                    max_tokens=args.long_max_tokens,
                    prompt="不要使用工具。连续写一篇约 800 个中文字符的海洋科普短文，不要列提纲。",
                )
                stages.append(normalize_http_stage(stage))
                _progress("l2_long", {"concurrency": concurrency, "successful": stage["successful"]})
                if stage["failed"] or stage["error_frames"]:
                    stop_reasons.append("sse_long_failed")
                if resource_guard is not None:
                    stop_reasons.extend(resource_guard.check())
                if stop_reasons:
                    break

        reliability = _ReliabilityClient(
            client,
            args.target_url,
            token,
            args.model_id,
            manifest,
            max_tokens=max(args.long_max_tokens, 1024),
        )
        if not stop_reasons:
            for concurrency in args.l3_concurrency:
                stage = _recovery_stage(reliability, manifest, concurrency)
                stages.append(normalize_http_stage(stage))
                _progress(
                    "l3_recovery",
                    {
                        "concurrency": concurrency,
                        "successful": stage["successful"],
                        "reasons": stage.get("_reason_counts", {}),
                    },
                )
                if stage["failed"]:
                    stop_reasons.append("recovery_failed")
                if resource_guard is not None:
                    stop_reasons.extend(resource_guard.check())
                if stop_reasons:
                    break

        if not stop_reasons:
            for concurrency in args.l3_concurrency:
                stage = _stop_stage(reliability, manifest, concurrency)
                stages.append(normalize_http_stage(stage))
                _progress("l3_stop", {"concurrency": concurrency, "successful": stage["successful"]})
                if stage["failed"]:
                    stop_reasons.append("stop_failed")
                if resource_guard is not None:
                    stop_reasons.extend(resource_guard.check())
                if stop_reasons:
                    break

        if not stop_reasons:
            soak_stage, soak_reasons = _soak_stage(args, client, token, manifest, resource_guard)
            stages.append(normalize_http_stage(soak_stage))
            stop_reasons.extend(soak_reasons)
            _progress("l4", {"ticks": soak_stage["executed_ticks"], "failed": soak_stage["failed"]})
    finally:
        if reliability is not None:
            quiesce_errors = reliability.quiesce()
        cleanup = _cleanup_full(client, args.target_url, args.auth_url, token, manifest)

    stop_reasons.extend(quiesce_errors)
    if cleanup["errors"]:
        stop_reasons.append("cleanup_failed")
    finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return build_import_payload(
        run_id=run_id,
        model_id=args.model_id,
        stages=stages,
        stopped=bool(stop_reasons),
        stop_reasons=list(dict.fromkeys(stop_reasons)),
        cleanup=cleanup,
        resources=resource_guard.resources_summary() if resource_guard is not None else None,
        started_at=started_at,
        finished_at=finished_at,
    )


def main() -> int:
    try:
        args = build_parser().parse_args()
        result = execute(args)
        serialized = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
            print(json.dumps({"run_id": result["run_id"], "status": result["status"]}, ensure_ascii=False))
        else:
            print(serialized)
        return 2 if result["status"] == "stopped" else 0
    except (RunnerError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
