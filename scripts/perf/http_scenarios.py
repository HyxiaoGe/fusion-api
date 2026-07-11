"""Fusion L1 HTTP 场景定义与安全阶梯采样。

本模块只返回延迟、成功率和吞吐等聚合指标。令牌、账号、请求 JSON、URL
以及响应正文仅在进程内参与请求，不进入 repr 或结果。
"""

from __future__ import annotations

import concurrent.futures
import copy
import re
import socket
import time
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol
from urllib.parse import quote, urlencode

from scripts.perf.core import RequestSample, StopPolicy, summarize_samples

JsonMethod = Literal["GET", "POST"]
ResponseValidator = Callable[[dict[str, Any]], bool]
_SAFE_SCENARIO_NAME = re.compile(r"^[a-z0-9_-]{1,50}$")


class JsonResponseLike(Protocol):
    status: int
    data: dict[str, Any]


class JsonRequester(Protocol):
    def request_json(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> JsonResponseLike: ...


@dataclass(frozen=True)
class JsonScenario:
    """单个 JSON API 场景；敏感请求数据明确排除在 repr 和结果之外。"""

    name: str
    method: JsonMethod
    url: str = field(repr=False)
    payload: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    token: str | None = field(default=None, repr=False, compare=False)
    expected_statuses: tuple[int, ...] = (200,)
    response_validator: ResponseValidator | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not _SAFE_SCENARIO_NAME.fullmatch(self.name):
            raise ValueError("场景名称只能包含小写字母、数字、下划线和短横线")
        if self.method not in {"GET", "POST"}:
            raise ValueError("JSON 场景只支持 GET 或 POST")
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("场景 URL 必须是 HTTP(S)")
        if not self.expected_statuses:
            raise ValueError("expected_statuses 不能为空")
        if self.method == "GET" and self.payload is not None:
            raise ValueError("GET 场景不能携带 JSON payload")

    @property
    def authenticated(self) -> bool:
        return bool(self.token)


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _auth_login_response(data: dict[str, Any]) -> bool:
    return isinstance(data.get("access_token"), str) and bool(data["access_token"])


def _models_response(data: dict[str, Any]) -> bool:
    payload = data.get("data")
    return data.get("code") == "SUCCESS" and isinstance(payload, dict) and isinstance(payload.get("models"), list)


def _conversation_list_response(data: dict[str, Any]) -> bool:
    payload = data.get("data")
    return data.get("code") == "SUCCESS" and isinstance(payload, dict) and isinstance(payload.get("items"), list)


def _conversation_detail_response(data: dict[str, Any]) -> bool:
    payload = data.get("data")
    return data.get("code") == "SUCCESS" and isinstance(payload, dict) and bool(payload.get("id"))


def build_l1_scenarios(
    *,
    target_url: str,
    auth_url: str,
    email: str,
    password: str,
    client_id: str,
    access_token: str | None,
    conversation_id: str | None = None,
    page_size: int = 20,
) -> dict[str, JsonScenario]:
    """创建 L1 场景。

    `auth_login` 只登录调用方准备好的测试账号，绝不注册账号；注册和账号清理由
    上层 runner 在阶梯执行前后各做一次。获得 access token 后再构造带认证的会话
    场景，可避免每个请求重复注册。
    """

    if not 1 <= page_size <= 100:
        raise ValueError("page_size 必须在 1 到 100 之间")
    scenarios = {
        "auth_login": JsonScenario(
            name="auth_login",
            method="POST",
            url=_join_url(auth_url, "/auth/login"),
            payload={"email": email, "password": password, "client_id": client_id},
            response_validator=_auth_login_response,
        ),
        "models": JsonScenario(
            name="models",
            method="GET",
            url=_join_url(target_url, "/api/models/"),
            response_validator=_models_response,
        ),
    }
    if access_token:
        scenarios["conversation_list"] = JsonScenario(
            name="conversation_list",
            method="GET",
            url=f"{_join_url(target_url, '/api/chat/conversations')}?{urlencode({'page': 1, 'page_size': page_size})}",
            token=access_token,
            response_validator=_conversation_list_response,
        )
        if conversation_id:
            safe_conversation_id = quote(conversation_id, safe="")
            scenarios["conversation_detail"] = JsonScenario(
                name="conversation_detail",
                method="GET",
                url=_join_url(target_url, f"/api/chat/conversations/{safe_conversation_id}"),
                token=access_token,
                response_validator=_conversation_detail_response,
            )
    return scenarios


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _is_timeout(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, socket.timeout)):
        return True
    return isinstance(error, urllib.error.URLError) and isinstance(error.reason, (TimeoutError, socket.timeout))


def sample_json_request(client: JsonRequester, scenario: JsonScenario) -> RequestSample:
    """采样一次 JSON 请求，仅保留状态、耗时和安全错误类别。"""

    started = time.perf_counter()
    try:
        response = client.request_json(
            scenario.method,
            scenario.url,
            payload=copy.deepcopy(scenario.payload),
            token=scenario.token,
        )
        status = int(response.status)
        if status not in scenario.expected_statuses:
            return RequestSample(_elapsed_ms(started), status, error=f"http_{status}")
        if not isinstance(response.data, dict):
            return RequestSample(_elapsed_ms(started), status, error="invalid_response")
        if scenario.response_validator is not None:
            try:
                valid = scenario.response_validator(response.data)
            except Exception:
                valid = False
            if not valid:
                return RequestSample(_elapsed_ms(started), status, error="invalid_response")
        return RequestSample(_elapsed_ms(started), status)
    except Exception as error:  # noqa: BLE001 — 客户端实现可抛自定义网络异常，结果只保留安全类别
        timed_out = _is_timeout(error)
        return RequestSample(
            _elapsed_ms(started),
            None,
            error="timeout" if timed_out else "request_error",
            timed_out=timed_out,
        )


def _maximum_consecutive_failures(samples: list[RequestSample]) -> int:
    current = 0
    maximum = 0
    for sample in samples:
        current = 0 if sample.error is None else current + 1
        maximum = max(maximum, current)
    return maximum


def run_scenario_stage(
    client: JsonRequester,
    scenario: JsonScenario,
    *,
    concurrency: int,
    requests: int,
) -> tuple[dict[str, Any], int]:
    """执行单档并发，只返回安全聚合与最大连续失败数。"""

    if concurrency < 1 or requests < 1:
        raise ValueError("concurrency 与 requests 必须是正整数")
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        samples = list(executor.map(lambda _: sample_json_request(client, scenario), range(requests)))
    wall_seconds = time.perf_counter() - started
    summary = summarize_samples(samples)
    stage = {
        "scenario": scenario.name,
        "kind": "http",
        "method": scenario.method,
        "authenticated": scenario.authenticated,
        "concurrency": concurrency,
        **summary,
        "requests_per_second": round(len(samples) / wall_seconds, 2) if wall_seconds else 0.0,
    }
    return stage, _maximum_consecutive_failures(samples)


def run_scenario_ladder(
    client: JsonRequester,
    scenario: JsonScenario,
    *,
    concurrencies: list[int] | tuple[int, ...],
    requests_per_stage: int,
    stop_policy: StopPolicy | None = None,
) -> dict[str, Any]:
    """按并发阶梯执行一个场景，并在门禁触发后停止升压。"""

    levels = list(concurrencies)
    if not levels or any(level < 1 for level in levels):
        raise ValueError("concurrencies 必须包含正整数")
    if requests_per_stage < 1:
        raise ValueError("requests_per_stage 必须是正整数")
    policy = stop_policy or StopPolicy()
    stages: list[dict[str, Any]] = []
    stop_reasons: list[str] = []
    for concurrency in levels:
        stage, consecutive = run_scenario_stage(
            client,
            scenario,
            concurrency=concurrency,
            requests=requests_per_stage,
        )
        stages.append(stage)
        reasons = policy.evaluate(stage, consecutive_failures=consecutive)
        if reasons:
            stop_reasons = [f"{scenario.name}:{reason}" for reason in reasons]
            break
    return {
        "scenario": scenario.name,
        "kind": "http_ladder",
        "stages": stages,
        "stopped": bool(stop_reasons),
        "stop_reasons": stop_reasons,
    }
