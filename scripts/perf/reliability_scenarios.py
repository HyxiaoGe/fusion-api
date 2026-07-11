"""L3/L4 可靠性压测场景的纯编排模块。

本模块只接收可注入回调，不负责认证、HTTP 请求或响应正文处理。输出仅包含聚合指标和
受控错误码，避免把会话标识、消息内容、游标或异常消息写入压测结果。
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
_STREAM_EVENT_ID = re.compile(r"^(\d+)-(\d+)$")


def _safe_label(value: str | None, fallback: str = "unknown_error") -> str | None:
    if value is None:
        return None
    return value if _SAFE_LABEL.fullmatch(value) else fallback


def _exception_reason(stage: str, exc: BaseException) -> str:
    return f"{stage}_exception:{type(exc).__name__}"


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[rank])


@dataclass(frozen=True, slots=True)
class StreamReadObservation:
    """一次 SSE 读取的非敏感观测结果。"""

    event_ids: tuple[str, ...] = ()
    chunk_types: tuple[str, ...] = ()
    message_id: str | None = field(default=None, repr=False)
    done: bool = False
    disconnected: bool = False
    error_frames: int = 0
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.error_frames < 0:
            raise ValueError("error_frames 不能为负数")

    @property
    def last_event_id(self) -> str:
        """返回客户端最后完整消费的游标；没有事件时从 0 开始。"""

        return self.event_ids[-1] if self.event_ids else "0"


@dataclass(frozen=True, slots=True)
class StreamStatusObservation:
    """stream-status 的最小非正文观测结果。"""

    status: str
    message_id: str | None = field(default=None, repr=False)
    last_entry_id: str | None = field(default=None, repr=False)
    stream_mode: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class StopAck:
    cancelled: bool
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class RecoveryOutcome:
    case_id: str
    success: bool
    duration_ms: float
    initial_events: int
    recovered_events: int
    duplicate_events: int
    lost_events: int
    ordering_errors: int
    error_frames: int
    status: str | None
    reasons: tuple[str, ...]

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 3),
            "initial_events": self.initial_events,
            "recovered_events": self.recovered_events,
            "duplicate_events": self.duplicate_events,
            "lost_events": self.lost_events,
            "ordering_errors": self.ordering_errors,
            "error_frames": self.error_frames,
            "status": _safe_label(self.status, "unknown_status"),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class RecoveryBatchOutcome:
    cases: tuple[RecoveryOutcome, ...]
    total: int
    successful: int
    failed: int
    p95_duration_ms: float
    error_rate: float
    lost_events: int
    ordering_errors: int
    reason_counts: tuple[tuple[str, int], ...]

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "p95_duration_ms": round(self.p95_duration_ms, 3),
            "error_rate": self.error_rate,
            "lost_events": self.lost_events,
            "ordering_errors": self.ordering_errors,
            "reason_counts": dict(self.reason_counts),
            "cases": [case.to_safe_dict() for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class StopOutcome:
    case_id: str
    success: bool
    duration_ms: float
    stop_attempted: bool
    cancelled: bool
    persistence_verified: bool | None
    pre_status: str | None
    post_status: str | None
    error_frames: int
    reasons: tuple[str, ...]

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 3),
            "stop_attempted": self.stop_attempted,
            "cancelled": self.cancelled,
            "persistence_verified": self.persistence_verified,
            "pre_status": _safe_label(self.pre_status, "unknown_status"),
            "post_status": _safe_label(self.post_status, "unknown_status"),
            "error_frames": self.error_frames,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class SoakSample:
    latency_ms: float
    success: bool
    timed_out: bool = False
    error_code: str | None = None
    requests: int = 1
    failures: int | None = None
    timeouts: int | None = None

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("latency_ms 不能为负数")
        failures = int(not self.success) if self.failures is None else self.failures
        timeouts = int(self.timed_out) if self.timeouts is None else self.timeouts
        if self.requests < 1:
            raise ValueError("requests 必须为正数")
        if not 0 <= failures <= self.requests:
            raise ValueError("failures 必须位于 [0, requests]")
        if not 0 <= timeouts <= failures:
            raise ValueError("timeouts 必须位于 [0, failures]")

    @property
    def failure_count(self) -> int:
        return int(not self.success) if self.failures is None else self.failures

    @property
    def timeout_count(self) -> int:
        return int(self.timed_out) if self.timeouts is None else self.timeouts


@dataclass(frozen=True, slots=True)
class SoakPolicy:
    duration_seconds: float = 30 * 60
    cadence_seconds: float = 5
    window_seconds: float = 60
    min_samples: int = 20
    max_error_rate: float = 0.05
    max_timeout_rate: float = 0.05
    max_consecutive_failures: int = 5
    max_p95_ms: float | None = None

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds 必须大于 0")
        if self.cadence_seconds <= 0:
            raise ValueError("cadence_seconds 必须大于 0")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds 必须大于 0")
        if self.min_samples <= 0:
            raise ValueError("min_samples 必须大于 0")
        if not 0 <= self.max_error_rate <= 1:
            raise ValueError("max_error_rate 必须位于 [0, 1]")
        if not 0 <= self.max_timeout_rate <= 1:
            raise ValueError("max_timeout_rate 必须位于 [0, 1]")
        if self.max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures 必须大于 0")
        if self.max_p95_ms is not None and self.max_p95_ms <= 0:
            raise ValueError("max_p95_ms 必须大于 0")


@dataclass(frozen=True, slots=True)
class SoakWindowSummary:
    index: int
    start_seconds: float
    end_seconds: float
    samples: int
    successful: int
    failed: int
    timeouts: int
    p50_ms: float
    p95_ms: float
    max_ms: float
    error_rate: float
    timeout_rate: float
    peak_consecutive_failures: int
    error_codes: tuple[tuple[str, int], ...]

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "samples": self.samples,
            "successful": self.successful,
            "failed": self.failed,
            "timeouts": self.timeouts,
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "error_rate": self.error_rate,
            "timeout_rate": self.timeout_rate,
            "peak_consecutive_failures": self.peak_consecutive_failures,
            "error_codes": dict(self.error_codes),
        }


@dataclass(frozen=True, slots=True)
class SoakResult:
    elapsed_seconds: float
    executed_ticks: int
    skipped_ticks: int
    windows: tuple[SoakWindowSummary, ...]
    stopped: bool
    stop_reasons: tuple[str, ...]

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "executed_ticks": self.executed_ticks,
            "skipped_ticks": self.skipped_ticks,
            "windows": [window.to_safe_dict() for window in self.windows],
            "stopped": self.stopped,
            "stop_reasons": list(self.stop_reasons),
        }


def _unique_reasons(reasons: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_safe_label(reason) or "unknown_error" for reason in reasons))


def _parse_stream_event_id(event_id: str) -> tuple[int, int] | None:
    matched = _STREAM_EVENT_ID.fullmatch(event_id)
    if matched is None:
        return None
    return int(matched.group(1)), int(matched.group(2))


def _ordering_error_count(event_ids: Sequence[str]) -> int:
    parsed = [_parse_stream_event_id(event_id) for event_id in event_ids]
    invalid = sum(event_id is None for event_id in parsed)
    non_increasing = sum(
        current <= previous
        for previous, current in zip(parsed, parsed[1:])
        if previous is not None and current is not None
    )
    return invalid + non_increasing


def run_disconnect_reconnect(
    conversation_ref: str,
    *,
    initial_read: Callable[[str], StreamReadObservation],
    read_status: Callable[[str], StreamStatusObservation],
    reconnect_read: Callable[[str, str], StreamReadObservation],
    case_id: str = "recovery-1",
    monotonic: Callable[[], float] = time.monotonic,
) -> RecoveryOutcome:
    """编排一次主动断线和断点续读，结果中不保留会话、消息或游标标识。"""

    started = monotonic()
    reasons: list[str] = []
    initial: StreamReadObservation | None = None
    status: StreamStatusObservation | None = None
    recovered: StreamReadObservation | None = None

    try:
        initial = initial_read(conversation_ref)
    except Exception as exc:  # noqa: BLE001 - 场景边界需要把任意客户端异常转成安全结果
        reasons.append(_exception_reason("initial_read", exc))

    if initial is not None:
        if initial.done:
            reasons.append("initial_completed_before_disconnect")
        if not initial.disconnected:
            reasons.append("disconnect_not_observed")
        if initial.error_frames:
            reasons.append("error_frame")
        if initial.error_code:
            reasons.append(f"initial_error:{_safe_label(initial.error_code)}")

        try:
            status = read_status(conversation_ref)
        except Exception as exc:  # noqa: BLE001
            reasons.append(_exception_reason("status", exc))

    if status is not None:
        if status.status not in {"streaming", "completed"}:
            reasons.append(f"status_not_recoverable:{_safe_label(status.status, 'unknown_status')}")
        if status.error_code:
            reasons.append(f"status_error:{_safe_label(status.error_code)}")

    if initial is not None and status is not None:
        try:
            # 只能从客户端最后完整处理的事件续读，不能使用服务端 stream-status 的尾游标。
            recovered = reconnect_read(conversation_ref, initial.last_event_id)
        except Exception as exc:  # noqa: BLE001
            reasons.append(_exception_reason("reconnect", exc))

    message_ids = {
        observation.message_id
        for observation in (initial, status, recovered)
        if observation is not None and observation.message_id is not None
    }
    if len(message_ids) > 1:
        reasons.append("message_id_mismatch")

    duplicate_events = 0
    if initial is not None and recovered is not None:
        duplicate_events = len(set(initial.event_ids).intersection(recovered.event_ids))
        if duplicate_events:
            reasons.append("duplicate_event")

    observed_event_ids = (initial.event_ids if initial else ()) + (recovered.event_ids if recovered else ())
    ordering_errors = _ordering_error_count(observed_event_ids)
    lost_events = 0
    if status is not None and status.last_entry_id not in {None, "0"}:
        if _parse_stream_event_id(status.last_entry_id) is None:
            ordering_errors += 1
            reasons.append("server_tail_invalid")
        elif status.last_entry_id not in observed_event_ids:
            # Redis Stream ID 不连续，无法推算缺失总数；尾游标缺失至少证明丢失一条。
            lost_events = 1
            reasons.append("server_tail_not_recovered")
    if ordering_errors:
        reasons.append("event_id_not_strictly_increasing")

    if recovered is not None:
        if recovered.error_frames:
            reasons.append("error_frame")
        if recovered.error_code:
            reasons.append(f"reconnect_error:{_safe_label(recovered.error_code)}")
        if not recovered.done:
            reasons.append("reconnect_not_terminal")

    safe_reasons = _unique_reasons(reasons)
    return RecoveryOutcome(
        case_id=_safe_label(case_id, "recovery") or "recovery",
        success=not safe_reasons,
        duration_ms=max(0.0, (monotonic() - started) * 1000),
        initial_events=len(initial.event_ids) if initial else 0,
        recovered_events=len(recovered.event_ids) if recovered else 0,
        duplicate_events=duplicate_events,
        lost_events=lost_events,
        ordering_errors=ordering_errors,
        error_frames=(initial.error_frames if initial else 0) + (recovered.error_frames if recovered else 0),
        status=_safe_label(status.status, "unknown_status") if status else None,
        reasons=safe_reasons,
    )


def run_concurrent_recovery(
    conversation_refs: Sequence[str],
    *,
    initial_read: Callable[[str], StreamReadObservation],
    read_status: Callable[[str], StreamStatusObservation],
    reconnect_read: Callable[[str, str], StreamReadObservation],
    max_workers: int = 4,
    monotonic: Callable[[], float] = time.monotonic,
) -> RecoveryBatchOutcome:
    """在有界线程池中并发执行恢复场景。"""

    if max_workers <= 0:
        raise ValueError("max_workers 必须大于 0")
    if not conversation_refs:
        return RecoveryBatchOutcome((), 0, 0, 0, 0.0, 0.0, 0, 0, ())

    outcomes: list[RecoveryOutcome | None] = [None] * len(conversation_refs)
    workers = min(max_workers, len(conversation_refs))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_indexes = {
            executor.submit(
                run_disconnect_reconnect,
                ref,
                initial_read=initial_read,
                read_status=read_status,
                reconnect_read=reconnect_read,
                case_id=f"recovery-{index + 1}",
                monotonic=monotonic,
            ): index
            for index, ref in enumerate(conversation_refs)
        }
        for future in as_completed(future_indexes):
            index = future_indexes[future]
            try:
                outcomes[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - 保底隔离线程任务异常
                reason = _exception_reason("scenario", exc)
                outcomes[index] = RecoveryOutcome(
                    case_id=f"recovery-{index + 1}",
                    success=False,
                    duration_ms=0.0,
                    initial_events=0,
                    recovered_events=0,
                    duplicate_events=0,
                    lost_events=0,
                    ordering_errors=0,
                    error_frames=0,
                    status=None,
                    reasons=(reason,),
                )

    cases = tuple(outcome for outcome in outcomes if outcome is not None)
    successful = sum(case.success for case in cases)
    reason_counts = Counter(reason for case in cases for reason in case.reasons)
    return RecoveryBatchOutcome(
        cases=cases,
        total=len(cases),
        successful=successful,
        failed=len(cases) - successful,
        p95_duration_ms=_percentile([case.duration_ms for case in cases], 0.95),
        error_rate=(len(cases) - successful) / len(cases),
        lost_events=sum(case.lost_events for case in cases),
        ordering_errors=sum(case.ordering_errors for case in cases),
        reason_counts=tuple(sorted(reason_counts.items())),
    )


def run_stop_scenario(
    conversation_ref: str,
    *,
    initial_read: Callable[[str], StreamReadObservation],
    read_status: Callable[[str], StreamStatusObservation],
    stop_stream: Callable[[str, str], StopAck],
    read_status_after_stop: Callable[[str], StreamStatusObservation],
    verify_persisted: Callable[[str, str], bool] | None = None,
    case_id: str = "stop-1",
    monotonic: Callable[[], float] = time.monotonic,
) -> StopOutcome:
    """编排带 message_id 防护的停止场景，正文由注入的客户端自行管理。"""

    started = monotonic()
    reasons: list[str] = []
    initial: StreamReadObservation | None = None
    before: StreamStatusObservation | None = None
    after: StreamStatusObservation | None = None
    ack: StopAck | None = None
    stop_attempted = False
    persistence_verified: bool | None = None

    try:
        initial = initial_read(conversation_ref)
    except Exception as exc:  # noqa: BLE001
        reasons.append(_exception_reason("initial_read", exc))

    if initial is not None:
        if initial.done:
            reasons.append("stream_already_terminal")
        if initial.error_frames:
            reasons.append("error_frame")
        if initial.error_code:
            reasons.append(f"initial_error:{_safe_label(initial.error_code)}")
        try:
            before = read_status(conversation_ref)
        except Exception as exc:  # noqa: BLE001
            reasons.append(_exception_reason("status_before_stop", exc))

    guarded_message_id: str | None = None
    if initial is not None and before is not None:
        if before.status != "streaming":
            reasons.append("stream_not_active")
        if initial.message_id and before.message_id and initial.message_id != before.message_id:
            reasons.append("message_id_mismatch")
        elif initial.message_id or before.message_id:
            guarded_message_id = initial.message_id or before.message_id
        else:
            reasons.append("message_id_missing")
        if before.error_code:
            reasons.append(f"status_error:{_safe_label(before.error_code)}")

    can_stop = (
        initial is not None
        and not initial.done
        and before is not None
        and before.status == "streaming"
        and guarded_message_id is not None
        and "message_id_mismatch" not in reasons
    )
    if can_stop and guarded_message_id is not None:
        stop_attempted = True
        try:
            ack = stop_stream(conversation_ref, guarded_message_id)
        except Exception as exc:  # noqa: BLE001
            reasons.append(_exception_reason("stop", exc))

        if ack is not None:
            if not ack.cancelled:
                reasons.append("stop_not_acknowledged")
            if ack.error_code:
                reasons.append(f"stop_error:{_safe_label(ack.error_code)}")

        try:
            after = read_status_after_stop(conversation_ref)
        except Exception as exc:  # noqa: BLE001
            reasons.append(_exception_reason("status_after_stop", exc))

        if after is not None:
            if after.status != "cancelled":
                reasons.append("status_not_cancelled")
            if after.message_id and after.message_id != guarded_message_id:
                reasons.append("message_id_mismatch")
            if after.error_code:
                reasons.append(f"post_status_error:{_safe_label(after.error_code)}")

        if verify_persisted is not None:
            try:
                persistence_verified = bool(verify_persisted(conversation_ref, guarded_message_id))
            except Exception as exc:  # noqa: BLE001
                reasons.append(_exception_reason("persistence", exc))
            if persistence_verified is False:
                reasons.append("persistence_not_verified")

    if stop_attempted and ack is None and not any(reason.startswith("stop_exception:") for reason in reasons):
        reasons.append("stop_ack_missing")
    if (
        stop_attempted
        and after is None
        and not any(reason.startswith("status_after_stop_exception:") for reason in reasons)
    ):
        reasons.append("post_status_missing")

    safe_reasons = _unique_reasons(reasons)
    cancelled = bool(ack and ack.cancelled and after and after.status == "cancelled")
    persistence_ok = verify_persisted is None or persistence_verified is True
    return StopOutcome(
        case_id=_safe_label(case_id, "stop") or "stop",
        success=stop_attempted and cancelled and persistence_ok and not safe_reasons,
        duration_ms=max(0.0, (monotonic() - started) * 1000),
        stop_attempted=stop_attempted,
        cancelled=cancelled,
        persistence_verified=persistence_verified,
        pre_status=_safe_label(before.status, "unknown_status") if before else None,
        post_status=_safe_label(after.status, "unknown_status") if after else None,
        error_frames=initial.error_frames if initial else 0,
        reasons=safe_reasons,
    )


def _summarize_window(index: int, samples: Sequence[SoakSample], policy: SoakPolicy) -> SoakWindowSummary:
    latencies = [sample.latency_ms for sample in samples]
    count = sum(sample.requests for sample in samples)
    failed = sum(sample.failure_count for sample in samples)
    timeouts = sum(sample.timeout_count for sample in samples)
    streak = 0
    peak_streak = 0
    errors: Counter[str] = Counter()
    for sample in samples:
        if sample.success:
            streak = 0
        else:
            streak += 1
            peak_streak = max(peak_streak, streak)
            errors[_safe_label(sample.error_code) or "unspecified_error"] += sample.failure_count

    return SoakWindowSummary(
        index=index,
        start_seconds=index * policy.window_seconds,
        end_seconds=min((index + 1) * policy.window_seconds, policy.duration_seconds),
        samples=count,
        successful=count - failed,
        failed=failed,
        timeouts=timeouts,
        p50_ms=_percentile(latencies, 0.50),
        p95_ms=_percentile(latencies, 0.95),
        max_ms=max(latencies, default=0.0),
        error_rate=failed / count if count else 0.0,
        timeout_rate=timeouts / count if count else 0.0,
        peak_consecutive_failures=peak_streak,
        error_codes=tuple(sorted(errors.items())),
    )


def _policy_stop_reasons(
    window: SoakWindowSummary,
    policy: SoakPolicy,
    consecutive_failures: int,
) -> list[str]:
    reasons: list[str] = []
    if consecutive_failures >= policy.max_consecutive_failures:
        reasons.append("consecutive_failures")
    if window.samples >= policy.min_samples:
        if window.error_rate > policy.max_error_rate:
            reasons.append("error_rate")
        if window.timeout_rate > policy.max_timeout_rate:
            reasons.append("timeout_rate")
        if policy.max_p95_ms is not None and window.p95_ms > policy.max_p95_ms:
            reasons.append("p95_latency")
    return reasons


def run_soak(
    execute_tick: Callable[[int], SoakSample],
    *,
    policy: SoakPolicy | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_window: Callable[[SoakWindowSummary], None] | None = None,
    hard_stop: Callable[[SoakWindowSummary], Iterable[str]] | None = None,
) -> SoakResult:
    """按固定节拍运行 soak；错过的节拍会跳过，避免恢复后突发追赶。"""

    active_policy = policy or SoakPolicy()
    started = monotonic()
    deadline = started + active_policy.duration_seconds
    slot = 0
    executed = 0
    skipped = 0
    consecutive_failures = 0
    active_window_index: int | None = None
    active_samples: list[SoakSample] = []
    windows: list[SoakWindowSummary] = []
    stop_reasons: list[str] = []

    def finalize_active_window() -> None:
        nonlocal active_samples
        if active_window_index is None or not active_samples:
            return
        summary = _summarize_window(active_window_index, active_samples, active_policy)
        windows.append(summary)
        active_samples = []
        if on_window is not None:
            try:
                on_window(summary)
            except Exception as exc:  # noqa: BLE001
                stop_reasons.append(_exception_reason("window_callback", exc))

    while not stop_reasons:
        now = monotonic()
        if now >= deadline:
            break

        scheduled_at = started + slot * active_policy.cadence_seconds
        if scheduled_at >= deadline:
            break
        if now < scheduled_at:
            sleep(max(0.0, scheduled_at - now))
            now = monotonic()
            if now >= deadline:
                break
        elif now > scheduled_at:
            latest_slot = math.floor((now - started) / active_policy.cadence_seconds)
            if latest_slot > slot:
                viable_slot = min(
                    latest_slot, math.ceil(active_policy.duration_seconds / active_policy.cadence_seconds) - 1
                )
                if viable_slot > slot:
                    skipped += viable_slot - slot
                    slot = viable_slot
                    scheduled_at = started + slot * active_policy.cadence_seconds

        window_index = math.floor((scheduled_at - started) / active_policy.window_seconds)
        if active_window_index is None:
            active_window_index = window_index
        elif window_index != active_window_index:
            finalize_active_window()
            if stop_reasons:
                break
            active_window_index = window_index

        try:
            sample = execute_tick(slot)
            if not isinstance(sample, SoakSample):
                raise TypeError("execute_tick 必须返回 SoakSample")
        except Exception as exc:  # noqa: BLE001
            sample = SoakSample(latency_ms=0.0, success=False, error_code=type(exc).__name__)

        active_samples.append(sample)
        executed += 1
        if sample.success:
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        current = _summarize_window(active_window_index, active_samples, active_policy)
        stop_reasons.extend(_policy_stop_reasons(current, active_policy, consecutive_failures))
        if hard_stop is not None and not stop_reasons:
            try:
                external_reasons = hard_stop(current)
                if isinstance(external_reasons, str):
                    external_reasons = (external_reasons,)
                stop_reasons.extend(_safe_label(reason) or "external_stop" for reason in external_reasons)
            except Exception as exc:  # noqa: BLE001
                stop_reasons.append(_exception_reason("hard_stop", exc))

        slot += 1

    stopped = bool(stop_reasons)
    if not stopped:
        remaining = deadline - monotonic()
        if remaining > 0:
            sleep(max(0.0, remaining))

    finalize_active_window()
    safe_stop_reasons = _unique_reasons(stop_reasons)
    return SoakResult(
        elapsed_seconds=max(0.0, monotonic() - started),
        executed_ticks=executed,
        skipped_ticks=skipped,
        windows=tuple(windows),
        stopped=bool(safe_stop_reasons),
        stop_reasons=safe_stop_reasons,
    )
