#!/usr/bin/env python3
"""Fusion HTTP/SSE 阶梯压测 runner；凭据仅存在于进程内存。"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import secrets
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from scripts.perf.core import (
    CleanupManifest,
    RequestSample,
    SSEParser,
    StopPolicy,
    build_safe_result,
    extract_agent_trace_ids,
    percentile,
)

DEFAULT_TARGET_URL = "https://fusion.seanfield.org"
DEFAULT_AUTH_URL = "https://auth.seanfield.org"
DEFAULT_CLIENT_ID = os.getenv("FUSION_PERF_CLIENT_ID", "app_a93ea0569cafafe6299c7f660669a5b7")


class RunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class JsonResponse:
    status: int
    data: dict[str, Any]


class HttpClient:
    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds

    def request_json(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> JsonResponse:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Accept": "application/json", "User-Agent": "fusion-perf-runner/1"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url=url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return JsonResponse(response.status, _decode_json(response.read()))
        except urllib.error.HTTPError as error:
            error.read()
            raise RunnerError(f"HTTP {error.code}: 请求失败") from error

    def open_sse(self, url: str, payload: dict[str, Any], token: str):
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "fusion-perf-runner/1",
        }
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        return urllib.request.urlopen(request, timeout=self.timeout_seconds)


def _decode_json(raw: bytes) -> dict[str, Any]:
    decoded = json.loads(raw.decode("utf-8")) if raw else {}
    if not isinstance(decoded, dict):
        raise RunnerError("接口响应不是 JSON object")
    return decoded


def join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def generate_identity() -> tuple[str, str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"perf-{timestamp}-{secrets.token_hex(4)}"
    email = f"fusion-perf+{run_id}@seanfield.org"
    password = secrets.token_urlsafe(32)
    return run_id, email, password


def authenticate(
    client: HttpClient,
    auth_url: str,
    client_id: str,
    manifest: CleanupManifest,
    password: str,
) -> str:
    registration = client.request_json(
        "POST",
        join_url(auth_url, "/auth/register"),
        payload={"email": manifest.email, "password": password, "name": f"Fusion Perf {manifest.run_id}"},
    ).data
    manifest.add_refresh_token("registration", _required_token(registration, "refresh_token"))
    login = client.request_json(
        "POST",
        join_url(auth_url, "/auth/login"),
        payload={"email": manifest.email, "password": password, "client_id": client_id},
    ).data
    manifest.add_refresh_token("fusion_login", _required_token(login, "refresh_token"))
    return _required_token(login, "access_token")


def _required_token(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise RunnerError(f"认证响应缺少 {key}")
    return value


def perform_get(client: HttpClient, url: str) -> RequestSample:
    started = time.perf_counter()
    try:
        response = client.request_json("GET", url)
        error = None if response.status < 400 else f"http_{response.status}"
        return RequestSample(_elapsed_ms(started), response.status, error=error)
    except RunnerError as error:
        return RequestSample(_elapsed_ms(started), None, error=type(error).__name__)
    except (TimeoutError, socket.timeout):
        return RequestSample(_elapsed_ms(started), None, error="timeout", timed_out=True)
    except urllib.error.URLError as error:
        timed_out = isinstance(error.reason, (TimeoutError, socket.timeout))
        return RequestSample(
            _elapsed_ms(started), None, error="timeout" if timed_out else "network", timed_out=timed_out
        )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def run_http_stage(client: HttpClient, url: str, concurrency: int, requests: int) -> tuple[dict[str, Any], int]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        samples = list(executor.map(lambda _: perform_get(client, url), range(requests)))
    wall_seconds = time.perf_counter() - started
    consecutive = 0
    maximum_consecutive = 0
    for sample in samples:
        consecutive = 0 if sample.error is None else consecutive + 1
        maximum_consecutive = max(maximum_consecutive, consecutive)
    stage = {
        "kind": "http",
        "concurrency": concurrency,
        **_summarize_request_samples(samples, wall_seconds),
    }
    return stage, maximum_consecutive


def _summarize_request_samples(samples: list[RequestSample], wall_seconds: float) -> dict[str, Any]:
    from scripts.perf.core import summarize_samples

    summary = summarize_samples(samples)
    summary["requests_per_second"] = round(len(samples) / wall_seconds, 2) if wall_seconds else 0.0
    return summary


def run_sse_flow(
    client: HttpClient,
    target_url: str,
    token: str,
    model_id: str,
    manifest: CleanupManifest,
    max_tokens: int,
) -> dict[str, Any]:
    conversation_id = str(uuid.uuid4())
    manifest.add_conversation(conversation_id)
    payload = _chat_payload(manifest.run_id, conversation_id, model_id, max_tokens)
    started = time.perf_counter()
    parser = SSEParser()
    chunk_types: list[str] = []
    ttft_ms: float | None = None
    done = False
    try:
        with client.open_sse(join_url(target_url, "/api/chat/send"), payload, token) as response:
            for raw_line in response:
                event = parser.feed_line(raw_line.decode("utf-8"))
                if event is None:
                    continue
                if event.done:
                    done = True
                    break
                event_payload = event.payload or {}
                chunk_type = str(event_payload.get("chunk_type", ""))
                chunk_types.append(chunk_type)
                run_id, trace_id = extract_agent_trace_ids(event_payload)
                manifest.add_agent_trace(run_id, trace_id)
                if ttft_ms is None and chunk_type in {"reasoning", "answering"}:
                    ttft_ms = _elapsed_ms(started)
        error_frames = chunk_types.count("error")
        return _safe_sse_flow(started, ttft_ms, done, chunk_types, error_frames, None)
    except (RunnerError, ValueError, UnicodeDecodeError, urllib.error.URLError, TimeoutError, socket.timeout) as error:
        return _safe_sse_flow(started, ttft_ms, done, chunk_types, 0, type(error).__name__)


def _chat_payload(run_id: str, conversation_id: str, model_id: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "message": f"[PERF:{run_id}] 请用一句简短中文回答：1+1 等于多少？不要使用工具。",
        "conversation_id": conversation_id,
        "stream": True,
        "options": {"max_tokens": max_tokens},
        "file_ids": [],
    }


def _safe_sse_flow(
    started: float,
    ttft_ms: float | None,
    done: bool,
    chunk_types: list[str],
    error_frames: int,
    error: str | None,
) -> dict[str, Any]:
    return {
        "total_ms": _elapsed_ms(started),
        "ttft_ms": ttft_ms,
        "done": done,
        "event_count": len(chunk_types),
        "error_frames": error_frames,
        "error": error,
    }


def run_sse_stage(
    client: HttpClient,
    target_url: str,
    token: str,
    model_id: str,
    manifest: CleanupManifest,
    concurrency: int,
    max_tokens: int,
) -> dict[str, Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_sse_flow, client, target_url, token, model_id, manifest, max_tokens)
            for _ in range(concurrency)
        ]
        flows = [future.result() for future in futures]
    successful = sum(
        1
        for flow in flows
        if flow["done"] and flow["ttft_ms"] is not None and not flow["error"] and not flow["error_frames"]
    )
    return {
        "kind": "sse",
        "concurrency": concurrency,
        "flows": len(flows),
        "successful": successful,
        "failed": len(flows) - successful,
        "p50_ttft_ms": percentile([flow["ttft_ms"] for flow in flows if flow["ttft_ms"] is not None], 0.50),
        "p95_ttft_ms": percentile([flow["ttft_ms"] for flow in flows if flow["ttft_ms"] is not None], 0.95),
        "p95_total_ms": percentile([flow["total_ms"] for flow in flows], 0.95),
        "error_frames": sum(flow["error_frames"] for flow in flows),
    }


def cleanup_run(client: HttpClient, target_url: str, auth_url: str, token: str | None, manifest: CleanupManifest):
    errors: list[str] = []
    deleted = 0
    if token:
        for conversation_id in sorted(manifest.conversation_ids):
            try:
                client.request_json(
                    "DELETE", join_url(target_url, f"/api/chat/conversations/{conversation_id}"), token=token
                )
                deleted += 1
            except (RunnerError, ValueError, urllib.error.URLError, TimeoutError, socket.timeout):
                errors.append("conversation_delete_failed")
    revoked = 0
    for _, refresh_token in manifest.refresh_tokens():
        try:
            client.request_json(
                "POST", join_url(auth_url, "/auth/token/revoke"), payload={"refresh_token": refresh_token}
            )
            revoked += 1
        except (RunnerError, ValueError, urllib.error.URLError, TimeoutError, socket.timeout):
            errors.append("token_revoke_failed")
    return {"conversations_deleted": deleted, "tokens_revoked": revoked, "errors": errors}


def parse_concurrency(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("并发阶梯必须是逗号分隔的正整数")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fusion 生产 HTTP/SSE 阶梯压测")
    parser.add_argument("--mode", choices=("http", "sse", "all"), default="all")
    parser.add_argument("--target-url", default=DEFAULT_TARGET_URL)
    parser.add_argument("--auth-url", default=DEFAULT_AUTH_URL)
    parser.add_argument("--client-id", default=DEFAULT_CLIENT_ID)
    parser.add_argument("--model-id")
    parser.add_argument("--http-path", default="/api/models/")
    parser.add_argument("--http-concurrency", type=parse_concurrency, default=parse_concurrency("1,5,10,25,50"))
    parser.add_argument("--sse-concurrency", type=parse_concurrency, default=parse_concurrency("1,3,5"))
    parser.add_argument("--requests-per-stage", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--confirm-production", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if "seanfield.org" in args.target_url and not args.confirm_production:
        raise RunnerError("生产域名压测必须显式传入 --confirm-production")
    if args.mode in {"sse", "all"} and not args.model_id:
        raise RunnerError("SSE 压测必须传入 --model-id")
    if args.requests_per_stage < 1 or args.max_tokens < 1:
        raise RunnerError("requests-per-stage 与 max-tokens 必须是正整数")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    client = HttpClient(args.timeout_seconds)
    run_id, email, password = generate_identity()
    manifest = CleanupManifest(run_id=run_id, email=email)
    policy = StopPolicy(max_p95_ms=args.max_p95_ms)
    token: str | None = None
    stages: list[dict[str, Any]] = []
    stop_reasons: list[str] = []
    try:
        if args.mode in {"http", "all"}:
            stop_reasons = _execute_http_stages(args, client, policy, stages)
        if not stop_reasons and args.mode in {"sse", "all"}:
            token = authenticate(client, args.auth_url, args.client_id, manifest, password)
            stop_reasons = _execute_sse_stages(args, client, token, manifest, stages)
    finally:
        cleanup = cleanup_run(client, args.target_url, args.auth_url, token, manifest)
    return build_safe_result(
        manifest=manifest,
        stages=stages,
        cleanup=cleanup,
        stopped=bool(stop_reasons),
        stop_reasons=stop_reasons,
    )


def _execute_http_stages(args, client, policy, stages) -> list[str]:
    url = join_url(args.target_url, args.http_path)
    for concurrency in args.http_concurrency:
        stage, consecutive = run_http_stage(client, url, concurrency, args.requests_per_stage)
        stages.append(stage)
        reasons = policy.evaluate(stage, consecutive_failures=consecutive)
        if reasons:
            return [f"http:{reason}" for reason in reasons]
    return []


def _execute_sse_stages(args, client, token, manifest, stages) -> list[str]:
    for concurrency in args.sse_concurrency:
        stage = run_sse_stage(client, args.target_url, token, args.model_id, manifest, concurrency, args.max_tokens)
        stages.append(stage)
        if stage["failed"]:
            return ["sse:failed_flow"]
        if stage["error_frames"]:
            return ["sse:error_frame"]
    return []


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
