import json
import unittest

from scripts.perf.reliability_scenarios import (
    SoakPolicy,
    SoakSample,
    StopAck,
    StreamReadObservation,
    StreamStatusObservation,
    run_concurrent_recovery,
    run_disconnect_reconnect,
    run_soak,
    run_stop_scenario,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds


class ReliabilityScenarioTests(unittest.TestCase):
    def test_disconnect_reconnect_uses_client_cursor_not_server_tail(self):
        reconnect_cursors: list[str] = []

        result = run_disconnect_reconnect(
            "private-conversation-id",
            initial_read=lambda _: StreamReadObservation(
                event_ids=("1-0", "2-0"),
                chunk_types=("preparing", "answering"),
                message_id="message-1",
                disconnected=True,
            ),
            read_status=lambda _: StreamStatusObservation(
                status="streaming",
                message_id="message-1",
                last_entry_id="99-0",
            ),
            reconnect_read=lambda _, cursor: (
                reconnect_cursors.append(cursor)
                or StreamReadObservation(
                    event_ids=("3-0", "99-0"),
                    chunk_types=("answering", "done"),
                    message_id="message-1",
                    done=True,
                )
            ),
        )

        self.assertTrue(result.success)
        self.assertEqual(reconnect_cursors, ["2-0"])
        self.assertEqual(result.initial_events, 2)
        self.assertEqual(result.recovered_events, 2)
        self.assertEqual(result.duplicate_events, 0)
        self.assertEqual(result.lost_events, 0)
        self.assertEqual(result.ordering_errors, 0)
        self.assertNotIn("private-conversation-id", repr(result))

    def test_disconnect_reconnect_detects_missing_server_tail_without_exposing_cursor(self):
        result = run_disconnect_reconnect(
            "private-conversation-id",
            initial_read=lambda _: StreamReadObservation(
                event_ids=("1-0", "2-0"),
                message_id="message-1",
                disconnected=True,
            ),
            read_status=lambda _: StreamStatusObservation(
                status="streaming",
                message_id="message-1",
                last_entry_id="99-0",
            ),
            reconnect_read=lambda _conversation_id, _cursor: StreamReadObservation(
                event_ids=("3-0", "4-0"),
                message_id="message-1",
                done=True,
            ),
        )

        serialized = json.dumps(result.to_safe_dict())
        self.assertFalse(result.success)
        self.assertEqual(result.lost_events, 1)
        self.assertIn("server_tail_not_recovered", result.reasons)
        self.assertNotIn("99-0", serialized)

    def test_disconnect_reconnect_requires_strictly_increasing_event_ids(self):
        result = run_disconnect_reconnect(
            "conversation-1",
            initial_read=lambda _: StreamReadObservation(
                event_ids=("1-0", "3-0"),
                message_id="message-1",
                disconnected=True,
            ),
            read_status=lambda _: StreamStatusObservation(
                status="streaming",
                message_id="message-1",
                last_entry_id="4-0",
            ),
            reconnect_read=lambda _conversation_id, _cursor: StreamReadObservation(
                event_ids=("2-0", "4-0"),
                message_id="message-1",
                done=True,
            ),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.ordering_errors, 1)
        self.assertIn("event_id_not_strictly_increasing", result.reasons)

    def test_completed_status_without_server_tail_remains_compatible(self):
        result = run_disconnect_reconnect(
            "conversation-1",
            initial_read=lambda _: StreamReadObservation(
                event_ids=("1-0",),
                message_id="message-1",
                disconnected=True,
            ),
            read_status=lambda _: StreamStatusObservation(status="completed"),
            reconnect_read=lambda _conversation_id, _cursor: StreamReadObservation(
                event_ids=("2-0",),
                message_id="message-1",
                done=True,
            ),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.lost_events, 0)
        self.assertEqual(result.ordering_errors, 0)

    def test_disconnect_reconnect_rejects_duplicates_error_frames_and_message_mismatch(self):
        result = run_disconnect_reconnect(
            "conversation-1",
            initial_read=lambda _: StreamReadObservation(
                event_ids=("1-0", "2-0"),
                chunk_types=("answering",),
                message_id="message-1",
                disconnected=True,
            ),
            read_status=lambda _: StreamStatusObservation(status="streaming", message_id="message-2"),
            reconnect_read=lambda _conversation_id, _cursor: StreamReadObservation(
                event_ids=("2-0", "3-0"),
                chunk_types=("error",),
                message_id="message-2",
                done=False,
                error_frames=1,
            ),
        )

        self.assertFalse(result.success)
        self.assertIn("message_id_mismatch", result.reasons)
        self.assertIn("duplicate_event", result.reasons)
        self.assertIn("error_frame", result.reasons)
        self.assertIn("reconnect_not_terminal", result.reasons)

    def test_concurrent_recovery_is_bounded_and_never_exposes_refs_or_exception_messages(self):
        private_refs = ["private-conversation-a", "private-conversation-b"]

        def initial_read(conversation_id: str) -> StreamReadObservation:
            if conversation_id.endswith("b"):
                raise RuntimeError("Bearer should-never-appear")
            return StreamReadObservation(
                event_ids=("1-0",),
                chunk_types=("answering",),
                message_id="message-1",
                disconnected=True,
            )

        batch = run_concurrent_recovery(
            private_refs,
            initial_read=initial_read,
            read_status=lambda _: StreamStatusObservation(status="streaming", message_id="message-1"),
            reconnect_read=lambda _conversation_id, _cursor: StreamReadObservation(
                event_ids=("2-0",),
                chunk_types=("done",),
                message_id="message-1",
                done=True,
            ),
            max_workers=2,
        )

        serialized = json.dumps(batch.to_safe_dict())
        self.assertEqual(batch.total, 2)
        self.assertEqual(batch.successful, 1)
        self.assertEqual(batch.failed, 1)
        self.assertEqual(batch.lost_events, 0)
        self.assertEqual(batch.ordering_errors, 0)
        self.assertNotIn("private-conversation", serialized)
        self.assertNotIn("Bearer", serialized)
        self.assertIn("initial_read_exception:RuntimeError", serialized)

    def test_concurrent_recovery_aggregates_lost_and_out_of_order_events(self):
        batch = run_concurrent_recovery(
            ["lost-case", "ordering-case"],
            initial_read=lambda conversation_id: StreamReadObservation(
                event_ids=("1-0", "3-0") if conversation_id == "ordering-case" else ("1-0", "2-0"),
                message_id="message-1",
                disconnected=True,
            ),
            read_status=lambda conversation_id: StreamStatusObservation(
                status="streaming",
                message_id="message-1",
                last_entry_id="4-0" if conversation_id == "ordering-case" else "99-0",
            ),
            reconnect_read=lambda conversation_id, _cursor: StreamReadObservation(
                event_ids=("2-0", "4-0") if conversation_id == "ordering-case" else ("3-0", "4-0"),
                message_id="message-1",
                done=True,
            ),
            max_workers=2,
        )

        serialized = json.dumps(batch.to_safe_dict())
        self.assertEqual(batch.lost_events, 1)
        self.assertEqual(batch.ordering_errors, 1)
        self.assertNotIn("99-0", serialized)

    def test_stop_scenario_requires_cancelled_status_and_persistence_verification(self):
        stop_calls: list[tuple[str, str]] = []
        result = run_stop_scenario(
            "private-conversation",
            initial_read=lambda _: StreamReadObservation(
                event_ids=("1-0",),
                chunk_types=("answering",),
                message_id="message-1",
                disconnected=True,
            ),
            read_status=lambda _: StreamStatusObservation(status="streaming", message_id="message-1"),
            stop_stream=lambda conversation_id, message_id: (
                stop_calls.append((conversation_id, message_id)) or StopAck(cancelled=True)
            ),
            read_status_after_stop=lambda _: StreamStatusObservation(status="cancelled", message_id="message-1"),
            verify_persisted=lambda _conversation_id, _message_id: True,
        )

        self.assertTrue(result.success)
        self.assertTrue(result.stop_attempted)
        self.assertTrue(result.cancelled)
        self.assertTrue(result.persistence_verified)
        self.assertEqual(stop_calls, [("private-conversation", "message-1")])
        self.assertNotIn("private-conversation", repr(result))

    def test_stop_scenario_does_not_issue_unsafe_stop_for_wrong_stream(self):
        stop_calls = 0

        def stop_stream(_conversation_id: str, _message_id: str) -> StopAck:
            nonlocal stop_calls
            stop_calls += 1
            return StopAck(cancelled=True)

        result = run_stop_scenario(
            "conversation-1",
            initial_read=lambda _: StreamReadObservation(
                event_ids=("1-0",),
                chunk_types=("answering",),
                message_id="old-message",
                disconnected=True,
            ),
            read_status=lambda _: StreamStatusObservation(status="streaming", message_id="new-message"),
            stop_stream=stop_stream,
            read_status_after_stop=lambda _: StreamStatusObservation(status="cancelled"),
        )

        self.assertFalse(result.success)
        self.assertEqual(stop_calls, 0)
        self.assertFalse(result.stop_attempted)
        self.assertIn("message_id_mismatch", result.reasons)

    def test_default_soak_is_thirty_minutes_and_fixed_cadence_windows_are_stable(self):
        self.assertEqual(SoakPolicy().duration_seconds, 30 * 60)
        clock = FakeClock()
        scheduled_slots: list[int] = []
        windows = []

        result = run_soak(
            lambda slot: scheduled_slots.append(slot) or SoakSample(latency_ms=100 + slot, success=True),
            policy=SoakPolicy(
                duration_seconds=5,
                cadence_seconds=1,
                window_seconds=2,
                min_samples=2,
            ),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            on_window=windows.append,
        )

        self.assertFalse(result.stopped)
        self.assertEqual(scheduled_slots, [0, 1, 2, 3, 4])
        self.assertEqual(result.executed_ticks, 5)
        self.assertEqual(result.skipped_ticks, 0)
        self.assertEqual([window.samples for window in result.windows], [2, 2, 1])
        self.assertEqual(windows, list(result.windows))
        self.assertAlmostEqual(result.elapsed_seconds, 5.0)
        self.assertTrue(all(call >= 0 for call in clock.sleep_calls))

    def test_soak_skips_missed_beats_instead_of_catch_up_burst(self):
        clock = FakeClock()
        executed: list[int] = []

        def execute(slot: int) -> SoakSample:
            executed.append(slot)
            clock.now += 2.4
            return SoakSample(latency_ms=2400, success=True)

        result = run_soak(
            execute,
            policy=SoakPolicy(duration_seconds=6, cadence_seconds=1, window_seconds=3),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

        self.assertEqual(executed, [0, 2, 4])
        self.assertEqual(result.skipped_ticks, 2)
        self.assertEqual(result.executed_ticks, 3)

    def test_soak_hard_stops_on_consecutive_failures_without_leaking_exception_message(self):
        clock = FakeClock()

        def execute(slot: int) -> SoakSample:
            if slot == 0:
                return SoakSample(latency_ms=20, success=False, error_code="upstream_error")
            raise ValueError("access_token=must-not-leak")

        result = run_soak(
            execute,
            policy=SoakPolicy(
                duration_seconds=30,
                cadence_seconds=1,
                window_seconds=10,
                min_samples=20,
                max_consecutive_failures=2,
            ),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

        serialized = json.dumps(result.to_safe_dict())
        self.assertTrue(result.stopped)
        self.assertEqual(result.executed_ticks, 2)
        self.assertIn("consecutive_failures", result.stop_reasons)
        self.assertNotIn("must-not-leak", serialized)
        self.assertIn("ValueError", serialized)

    def test_soak_accepts_external_hard_stop_callback(self):
        clock = FakeClock()

        result = run_soak(
            lambda _slot: SoakSample(latency_ms=50, success=True),
            policy=SoakPolicy(duration_seconds=10, cadence_seconds=1, window_seconds=5),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            hard_stop=lambda window: ["resource:api_cpu"] if window.samples >= 3 else [],
        )

        self.assertTrue(result.stopped)
        self.assertEqual(result.executed_ticks, 3)
        self.assertEqual(result.stop_reasons, ("resource:api_cpu",))

    def test_soak_batch_samples_preserve_flow_level_error_and_timeout_rates(self):
        clock = FakeClock()

        result = run_soak(
            lambda _slot: SoakSample(
                latency_ms=500,
                success=False,
                requests=4,
                failures=1,
                timeouts=1,
                error_code="timeout",
            ),
            policy=SoakPolicy(
                duration_seconds=1,
                cadence_seconds=1,
                window_seconds=1,
                min_samples=10,
                max_consecutive_failures=5,
            ),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

        window = result.windows[0]
        self.assertEqual(window.samples, 4)
        self.assertEqual(window.successful, 3)
        self.assertEqual(window.failed, 1)
        self.assertEqual(window.timeouts, 1)
        self.assertEqual(window.error_rate, 0.25)
        self.assertEqual(window.timeout_rate, 0.25)


if __name__ == "__main__":
    unittest.main()
