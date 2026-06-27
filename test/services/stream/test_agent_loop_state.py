import unittest

from app.schemas.chat import TextBlock, Usage
from app.services.stream.agent_loop_state import AgentLoopState


class AgentLoopStateTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
