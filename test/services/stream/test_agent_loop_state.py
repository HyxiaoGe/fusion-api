import unittest
from types import SimpleNamespace

from app.schemas.chat import ContextUsage, TextBlock, Usage
from app.services.stream.agent_loop_state import AgentLoopState


class AgentLoopStateTests(unittest.TestCase):
    def test_active_elapsed_excludes_context_wait(self):
        state = AgentLoopState(context_wait_seconds=45.0)

        self.assertEqual(state.active_elapsed_seconds(now=320.0, run_start=0.0), 275.0)

    def test_initial_state_matches_runner_defaults(self):
        state = AgentLoopState()

        self.assertEqual(state.content_blocks, [])
        self.assertEqual(state.accumulated_usage, Usage(input_tokens=0, output_tokens=0))
        self.assertEqual(state.step, 0)
        self.assertEqual(state.total_tool_calls, 0)
        self.assertEqual(state.current_step_id, None)
        self.assertEqual(state.finish_reason, "stop")
        self.assertEqual(state.limit_reason, None)
        self.assertEqual(state.unknown_terminated, False)
        self.assertEqual(state.terminal_emitted, False)

    def test_step_and_tool_call_mutations_are_explicit(self):
        state = AgentLoopState()

        self.assertEqual(state.next_step_number(), 1)
        state.mark_current_step("step-1")
        state.record_executed_tool_calls(2)
        state.clear_current_step()

        self.assertEqual(state.step, 1)
        self.assertEqual(state.total_tool_calls, 2)
        self.assertEqual(state.current_step_id, None)
        self.assertEqual(state.run_stats("run-1").total_steps, 1)
        self.assertEqual(state.run_stats("run-1").total_tool_calls, 2)

    def test_two_consecutive_no_progress_search_results_request_summary_across_rounds(self):
        state = AgentLoopState()

        state.record_no_progress_search_results((True,))
        self.assertFalse(state.should_summarize_no_progress_search())

        state.record_no_progress_search_results((True,))
        self.assertTrue(state.should_summarize_no_progress_search())

    def test_one_no_progress_search_result_does_not_request_summary(self):
        state = AgentLoopState()

        state.record_no_progress_search_results((True,))

        self.assertFalse(state.should_summarize_no_progress_search())

    def test_progress_result_resets_no_progress_search_streak(self):
        state = AgentLoopState()

        state.record_no_progress_search_results((True, False, True))

        self.assertEqual(state.consecutive_no_progress_search_results, 1)
        self.assertFalse(state.should_summarize_no_progress_search())

    def test_standard_search_record_does_not_enable_deep_research_gate(self):
        state = AgentLoopState(
            plan_coordinator=SimpleNamespace(execution_items_terminal=lambda: True),
        )
        state.configure_research_mode(network_required=False)

        state.record_research_content_blocks(
            [
                {
                    "type": "url_read",
                    "id": "url-read-failed",
                    "status": "failed",
                    "source_refs": [],
                }
            ]
        )

        self.assertFalse(state.research_network_required)
        self.assertTrue(state.ready_for_plan_synthesis())

    def test_deep_research_waits_for_two_successful_reads_before_plan_synthesis(self):
        state = AgentLoopState(
            plan_coordinator=SimpleNamespace(execution_items_terminal=lambda: True),
        )
        state.configure_research_mode(network_required=True)
        state.research_workset.successful_searches = 1
        state.research_workset.successful_read_urls = {"https://example.com/one"}

        self.assertFalse(state.ready_for_plan_synthesis())

        state.research_workset.successful_read_urls.add("https://example.com/two")

        self.assertTrue(state.ready_for_plan_synthesis())

    def test_usage_content_and_terminal_mutations_are_explicit(self):
        state = AgentLoopState()
        block = TextBlock(type="text", id="blk-1", text="answer")

        state.content_blocks.append(block)
        state.update_usage(Usage(input_tokens=3, output_tokens=5))
        state.mark_unknown_terminated()
        state.mark_terminal_emitted()

        self.assertEqual(state.content_blocks, [block])
        self.assertEqual(state.final_usage(), Usage(input_tokens=3, output_tokens=5))
        self.assertEqual(state.unknown_terminated, True)
        self.assertEqual(state.terminal_emitted, True)

    def test_final_usage_omits_zero_input_usage(self):
        state = AgentLoopState()

        self.assertIsNone(state.final_usage())

    def test_final_usage_keeps_accumulated_tokens_and_last_round_context(self):
        state = AgentLoopState(accumulated_usage=Usage(input_tokens=100, output_tokens=20))
        first = ContextUsage(status="no_op", window_tokens=1000, actual_prompt_tokens=40)
        last = ContextUsage(status="trimmed", window_tokens=1000, actual_prompt_tokens=70, removed_turns=1)

        state.update_context(first)
        state.update_context(last)

        final = state.final_usage()
        self.assertEqual(final.input_tokens, 100)
        self.assertEqual(final.output_tokens, 20)
        self.assertEqual(final.context, last)


if __name__ == "__main__":
    unittest.main()
