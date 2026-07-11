"""SSE L2 输出质量指标；只保留计数与时序，不保留任何模型内容。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

OUTPUT_CHUNK_TYPES = frozenset({"reasoning", "answering"})


def estimate_approx_tokens(text: str) -> int:
    """按非 ASCII 可见字符 1 token、ASCII 可见字符 4:1 粗略估算。"""
    non_ascii, ascii_visible = _count_token_units(text)
    return non_ascii + math.ceil(ascii_visible / 4)


@dataclass
class SSEFlowMetrics:
    """单条 SSE flow 的无内容指标采集器。"""

    started_at_ms: float
    _last_observed_at_ms: float | None = field(default=None, init=False)
    _first_output_at_ms: float | None = field(default=None, init=False)
    _first_reasoning_at_ms: float | None = field(default=None, init=False)
    _first_answering_at_ms: float | None = field(default=None, init=False)
    _last_output_at_ms: float | None = field(default=None, init=False)
    _finished_at_ms: float | None = field(default=None, init=False)
    _chunk_intervals_ms: list[float] = field(default_factory=list, init=False)
    _output_chunks: int = field(default=0, init=False)
    _reasoning_chunks: int = field(default=0, init=False)
    _answering_chunks: int = field(default=0, init=False)
    _reasoning_visible_chars: int = field(default=0, init=False)
    _answering_visible_chars: int = field(default=0, init=False)
    _non_ascii_token_units: int = field(default=0, init=False)
    _ascii_token_units: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.started_at_ms = _finite_timestamp(self.started_at_ms)

    def observe_envelope(self, envelope: Mapping[str, Any], *, observed_at_ms: float) -> bool:
        """消费现有 ``{chunk_type, data: {delta}}`` envelope；有效输出返回 True。"""
        observed_at_ms = self._validate_observed_at(observed_at_ms)
        self._last_observed_at_ms = observed_at_ms
        chunk_type = envelope.get("chunk_type")
        data = envelope.get("data")
        if chunk_type not in OUTPUT_CHUNK_TYPES or not isinstance(data, Mapping):
            return False
        delta = data.get("delta")
        if not isinstance(delta, str) or not delta:
            return False
        self._record_output(chunk_type, delta, observed_at_ms)
        return True

    def build_summary(self, *, finished_at_ms: float | None = None) -> dict[str, int | float | None]:
        """生成严格无内容的 flow 汇总。"""
        end_at_ms = self._resolve_finished_at(finished_at_ms)
        output_window_ms = self._output_window_ms()
        approx_tokens = self._approx_tokens()
        tokens_per_second = None
        if output_window_ms > 0 and approx_tokens > 0:
            tokens_per_second = round(approx_tokens / (output_window_ms / 1000), 2)
        return {
            "duration_ms": _rounded(end_at_ms - self.started_at_ms),
            "first_output_ms": self._relative_ms(self._first_output_at_ms),
            "first_reasoning_ms": self._relative_ms(self._first_reasoning_at_ms),
            "first_answering_ms": self._relative_ms(self._first_answering_at_ms),
            "output_chunks": self._output_chunks,
            "reasoning_chunks": self._reasoning_chunks,
            "answering_chunks": self._answering_chunks,
            "visible_chars": self._reasoning_visible_chars + self._answering_visible_chars,
            "reasoning_visible_chars": self._reasoning_visible_chars,
            "answering_visible_chars": self._answering_visible_chars,
            "approx_tokens": approx_tokens,
            "chunk_interval_count": len(self._chunk_intervals_ms),
            "chunk_interval_p50_ms": _optional_percentile(self._chunk_intervals_ms, 0.50),
            "chunk_interval_p95_ms": _optional_percentile(self._chunk_intervals_ms, 0.95),
            "chunk_interval_max_ms": _optional_max(self._chunk_intervals_ms),
            "output_window_ms": output_window_ms,
            "tokens_per_second": tokens_per_second,
        }

    def _validate_observed_at(self, observed_at_ms: float) -> float:
        observed_at_ms = _finite_timestamp(observed_at_ms)
        if observed_at_ms < self.started_at_ms:
            raise ValueError("观察时间戳不能早于开始时间")
        if self._last_observed_at_ms is not None and observed_at_ms < self._last_observed_at_ms:
            raise ValueError("观察时间戳必须单调递增或相等")
        if self._finished_at_ms is not None and observed_at_ms > self._finished_at_ms:
            raise ValueError("flow 已完成，不能继续记录输出")
        return observed_at_ms

    def _record_output(self, chunk_type: str, delta: str, observed_at_ms: float) -> None:
        if self._first_output_at_ms is None:
            self._first_output_at_ms = observed_at_ms
        if self._last_output_at_ms is not None:
            self._chunk_intervals_ms.append(observed_at_ms - self._last_output_at_ms)
        self._last_output_at_ms = observed_at_ms
        self._output_chunks += 1
        visible_chars = _count_visible_chars(delta)
        non_ascii, ascii_visible = _count_token_units(delta)
        self._non_ascii_token_units += non_ascii
        self._ascii_token_units += ascii_visible
        if chunk_type == "reasoning":
            if self._first_reasoning_at_ms is None:
                self._first_reasoning_at_ms = observed_at_ms
            self._reasoning_chunks += 1
            self._reasoning_visible_chars += visible_chars
        else:
            if self._first_answering_at_ms is None:
                self._first_answering_at_ms = observed_at_ms
            self._answering_chunks += 1
            self._answering_visible_chars += visible_chars

    def _resolve_finished_at(self, finished_at_ms: float | None) -> float:
        if finished_at_ms is None:
            return self._finished_at_ms or self._last_observed_at_ms or self.started_at_ms
        finished_at_ms = _finite_timestamp(finished_at_ms)
        minimum = self._last_observed_at_ms or self.started_at_ms
        if finished_at_ms < minimum:
            raise ValueError("完成时间不能早于最后观察时间")
        self._finished_at_ms = finished_at_ms
        return finished_at_ms

    def _relative_ms(self, timestamp_ms: float | None) -> float | None:
        return None if timestamp_ms is None else _rounded(timestamp_ms - self.started_at_ms)

    def _output_window_ms(self) -> float:
        if self._first_output_at_ms is None or self._last_output_at_ms is None:
            return 0.0
        return _rounded(self._last_output_at_ms - self._first_output_at_ms)

    def _approx_tokens(self) -> int:
        return self._non_ascii_token_units + math.ceil(self._ascii_token_units / 4)


def summarize_sse_stage(flows: Iterable[SSEFlowMetrics]) -> dict[str, int | float | None]:
    """聚合多个采集器的精确时序样本，输出无内容 stage 汇总。"""
    flow_list = list(flows)
    summaries = [flow.build_summary() for flow in flow_list]
    first_output = _numeric_values(summaries, "first_output_ms")
    output_windows = _numeric_values(summaries, "output_window_ms", require_output=True)
    tokens_per_second = _numeric_values(summaries, "tokens_per_second")
    intervals = [interval for flow in flow_list for interval in flow._chunk_intervals_ms]
    return {
        "flows": len(flow_list),
        "flows_with_output": len(first_output),
        "output_chunks": _sum_int(summaries, "output_chunks"),
        "reasoning_chunks": _sum_int(summaries, "reasoning_chunks"),
        "answering_chunks": _sum_int(summaries, "answering_chunks"),
        "visible_chars": _sum_int(summaries, "visible_chars"),
        "reasoning_visible_chars": _sum_int(summaries, "reasoning_visible_chars"),
        "answering_visible_chars": _sum_int(summaries, "answering_visible_chars"),
        "approx_tokens": _sum_int(summaries, "approx_tokens"),
        "first_output_p50_ms": _optional_percentile(first_output, 0.50),
        "first_output_p95_ms": _optional_percentile(first_output, 0.95),
        "first_output_max_ms": _optional_max(first_output),
        "chunk_interval_p50_ms": _optional_percentile(intervals, 0.50),
        "chunk_interval_p95_ms": _optional_percentile(intervals, 0.95),
        "chunk_interval_max_ms": _optional_max(intervals),
        "output_window_p50_ms": _optional_percentile(output_windows, 0.50),
        "output_window_p95_ms": _optional_percentile(output_windows, 0.95),
        "output_window_max_ms": _optional_max(output_windows),
        "tokens_per_second_p50": _optional_percentile(tokens_per_second, 0.50),
        "tokens_per_second_p95": _optional_percentile(tokens_per_second, 0.95),
        "tokens_per_second_max": _optional_max(tokens_per_second),
    }


def _count_visible_chars(text: str) -> int:
    return sum(not char.isspace() for char in text)


def _count_token_units(text: str) -> tuple[int, int]:
    non_ascii = 0
    ascii_visible = 0
    for char in text:
        if char.isspace():
            continue
        if ord(char) > 127:
            non_ascii += 1
        else:
            ascii_visible += 1
    return non_ascii, ascii_visible


def _finite_timestamp(value: float) -> float:
    timestamp = float(value)
    if not math.isfinite(timestamp):
        raise ValueError("时间戳必须是有限数值")
    return timestamp


def _rounded(value: float) -> float:
    return round(value, 2)


def _optional_percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percent) - 1)
    return _rounded(ordered[index])


def _optional_max(values: list[float]) -> float | None:
    return None if not values else _rounded(max(values))


def _sum_int(summaries: list[dict[str, int | float | None]], key: str) -> int:
    return sum(int(summary[key] or 0) for summary in summaries)


def _numeric_values(
    summaries: list[dict[str, int | float | None]],
    key: str,
    *,
    require_output: bool = False,
) -> list[float]:
    return [
        float(summary[key])
        for summary in summaries
        if summary[key] is not None and (not require_output or int(summary["output_chunks"] or 0) > 0)
    ]
