import json
import unittest

from scripts.perf.sse_metrics import SSEFlowMetrics, estimate_approx_tokens, summarize_sse_stage


def _delta(chunk_type: str, text: str) -> dict:
    return {
        "chunk_type": chunk_type,
        "data": {
            "block_id": "block-secret",
            "delta": text,
            "run_id": "run-secret",
            "step_id": "step-secret",
        },
    }


def test_flow_summary_counts_visible_output_and_chunk_timing_without_content():
    metrics = SSEFlowMetrics(started_at_ms=1_000)
    metrics.observe_envelope(_delta("reasoning", "你好 "), observed_at_ms=1_100)
    metrics.observe_envelope(_delta("answering", "hello"), observed_at_ms=1_200)
    metrics.observe_envelope(_delta("answering", " world"), observed_at_ms=1_400)

    summary = metrics.build_summary(finished_at_ms=1_500)

    assert summary == {
        "duration_ms": 500.0,
        "first_output_ms": 100.0,
        "first_reasoning_ms": 100.0,
        "first_answering_ms": 200.0,
        "output_chunks": 3,
        "reasoning_chunks": 1,
        "answering_chunks": 2,
        "visible_chars": 12,
        "reasoning_visible_chars": 2,
        "answering_visible_chars": 10,
        "approx_tokens": 5,
        "chunk_interval_count": 2,
        "chunk_interval_p50_ms": 100.0,
        "chunk_interval_p95_ms": 200.0,
        "chunk_interval_max_ms": 200.0,
        "output_window_ms": 300.0,
        "tokens_per_second": 16.67,
    }
    serialized = json.dumps(summary, ensure_ascii=False)
    for forbidden in ("你好", "hello", "world", "delta", "block-secret", "run-secret", "step-secret"):
        assert forbidden not in serialized
        assert forbidden not in repr(metrics)


def test_non_output_envelopes_and_invalid_deltas_are_ignored():
    metrics = SSEFlowMetrics(started_at_ms=0)

    assert metrics.observe_envelope({"chunk_type": "preparing", "data": {}}, observed_at_ms=10) is False
    assert metrics.observe_envelope({"chunk_type": "answering", "data": {"delta": 123}}, observed_at_ms=20) is False
    assert metrics.observe_envelope({"chunk_type": "reasoning", "data": {"delta": ""}}, observed_at_ms=30) is False
    assert metrics.observe_envelope({"chunk_type": "answering", "data": {"delta": "   "}}, observed_at_ms=40) is True

    summary = metrics.build_summary(finished_at_ms=50)

    assert summary["output_chunks"] == 1
    assert summary["visible_chars"] == 0
    assert summary["approx_tokens"] == 0
    assert summary["first_output_ms"] == 40.0
    assert summary["tokens_per_second"] is None


def test_single_output_chunk_has_no_interval_or_measurable_tokens_per_second():
    metrics = SSEFlowMetrics(started_at_ms=100)
    metrics.observe_envelope(_delta("answering", "单包"), observed_at_ms=150)

    summary = metrics.build_summary(finished_at_ms=180)

    assert summary["first_output_ms"] == 50.0
    assert summary["output_window_ms"] == 0.0
    assert summary["chunk_interval_count"] == 0
    assert summary["chunk_interval_p50_ms"] is None
    assert summary["chunk_interval_p95_ms"] is None
    assert summary["chunk_interval_max_ms"] is None
    assert summary["tokens_per_second"] is None


def test_zero_timestamp_is_preserved_as_first_reasoning_packet():
    metrics = SSEFlowMetrics(started_at_ms=0)
    metrics.observe_envelope(_delta("reasoning", "甲"), observed_at_ms=0)
    metrics.observe_envelope(_delta("reasoning", "乙"), observed_at_ms=1)

    summary = metrics.build_summary(finished_at_ms=2)

    assert summary["first_output_ms"] == 0.0
    assert summary["first_reasoning_ms"] == 0.0


def test_approximate_tokens_use_cjk_characters_and_four_visible_latin_chars_per_token():
    assert estimate_approx_tokens("你好") == 2
    assert estimate_approx_tokens("hello world") == 3
    assert estimate_approx_tokens("你好 hello") == 4
    assert estimate_approx_tokens("   \n") == 0


def test_stage_summary_aggregates_exact_safe_flow_metrics():
    first = SSEFlowMetrics(started_at_ms=1_000)
    first.observe_envelope(_delta("reasoning", "你好 "), observed_at_ms=1_100)
    first.observe_envelope(_delta("answering", "hello"), observed_at_ms=1_200)
    first.observe_envelope(_delta("answering", " world"), observed_at_ms=1_400)
    first.build_summary(finished_at_ms=1_500)

    second = SSEFlowMetrics(started_at_ms=0)
    second.observe_envelope(_delta("answering", "甲"), observed_at_ms=200)
    second.observe_envelope(_delta("answering", "乙"), observed_at_ms=400)
    second.build_summary(finished_at_ms=450)

    stage = summarize_sse_stage([first, second])

    assert stage == {
        "flows": 2,
        "flows_with_output": 2,
        "output_chunks": 5,
        "reasoning_chunks": 1,
        "answering_chunks": 4,
        "visible_chars": 14,
        "reasoning_visible_chars": 2,
        "answering_visible_chars": 12,
        "approx_tokens": 7,
        "first_output_p50_ms": 100.0,
        "first_output_p95_ms": 200.0,
        "first_output_max_ms": 200.0,
        "chunk_interval_p50_ms": 200.0,
        "chunk_interval_p95_ms": 200.0,
        "chunk_interval_max_ms": 200.0,
        "output_window_p50_ms": 200.0,
        "output_window_p95_ms": 300.0,
        "output_window_max_ms": 300.0,
        "tokens_per_second_p50": 10.0,
        "tokens_per_second_p95": 16.67,
        "tokens_per_second_max": 16.67,
    }
    assert "hello" not in json.dumps(stage)


def test_empty_stage_is_explicit_and_contains_no_samples():
    assert summarize_sse_stage([]) == {
        "flows": 0,
        "flows_with_output": 0,
        "output_chunks": 0,
        "reasoning_chunks": 0,
        "answering_chunks": 0,
        "visible_chars": 0,
        "reasoning_visible_chars": 0,
        "answering_visible_chars": 0,
        "approx_tokens": 0,
        "first_output_p50_ms": None,
        "first_output_p95_ms": None,
        "first_output_max_ms": None,
        "chunk_interval_p50_ms": None,
        "chunk_interval_p95_ms": None,
        "chunk_interval_max_ms": None,
        "output_window_p50_ms": None,
        "output_window_p95_ms": None,
        "output_window_max_ms": None,
        "tokens_per_second_p50": None,
        "tokens_per_second_p95": None,
        "tokens_per_second_max": None,
    }


def test_timestamps_must_be_monotonic():
    metrics = SSEFlowMetrics(started_at_ms=100)

    with unittest.TestCase().assertRaisesRegex(ValueError, "时间戳不能早于"):
        metrics.observe_envelope(_delta("answering", "x"), observed_at_ms=99)

    metrics.observe_envelope(_delta("answering", "x"), observed_at_ms=120)
    with unittest.TestCase().assertRaisesRegex(ValueError, "时间戳必须单调"):
        metrics.observe_envelope(_delta("answering", "y"), observed_at_ms=119)
    with unittest.TestCase().assertRaisesRegex(ValueError, "完成时间不能早于"):
        metrics.build_summary(finished_at_ms=119)


def load_tests(_loader, _tests, _pattern):
    """让仓库的 unittest discover CI 执行本模块的函数式用例。"""
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
