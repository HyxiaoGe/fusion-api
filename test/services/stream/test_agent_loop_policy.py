from app.services.stream.agent_loop_policy import (
    AgentLoopLimits,
    AgentRunTerminalState,
    check_agent_loop_limit,
    map_run_terminal_state,
)


def test_check_agent_loop_limit_returns_none_before_limits():
    limits = AgentLoopLimits(max_steps=3, max_tool_calls=5, total_timeout_s=10)

    assert check_agent_loop_limit(elapsed_seconds=10, step=2, total_tool_calls=4, limits=limits) is None


def test_check_agent_loop_limit_prefers_timeout_over_other_limits():
    limits = AgentLoopLimits(max_steps=3, max_tool_calls=5, total_timeout_s=10)

    reason = check_agent_loop_limit(
        elapsed_seconds=11,
        step=3,
        total_tool_calls=5,
        limits=limits,
    )

    assert reason == "timeout"


def test_check_agent_loop_limit_returns_max_steps_reason():
    limits = AgentLoopLimits(max_steps=3, max_tool_calls=5, total_timeout_s=10)

    reason = check_agent_loop_limit(
        elapsed_seconds=9,
        step=3,
        total_tool_calls=4,
        limits=limits,
    )

    assert reason == "max_steps"


def test_check_agent_loop_limit_returns_max_tool_calls_reason():
    limits = AgentLoopLimits(max_steps=3, max_tool_calls=5, total_timeout_s=10)

    reason = check_agent_loop_limit(
        elapsed_seconds=9,
        step=2,
        total_tool_calls=5,
        limits=limits,
    )

    assert reason == "max_tool_calls"


def test_map_run_terminal_state_maps_unknown_to_incomplete():
    terminal_state = map_run_terminal_state(unknown_terminated=True, limit_reason=None)

    assert terminal_state == AgentRunTerminalState(
        run_finish_reason="incomplete",
        session_status="incomplete",
    )


def test_map_run_terminal_state_maps_limit_to_limit_reached():
    terminal_state = map_run_terminal_state(unknown_terminated=False, limit_reason="max_steps")

    assert terminal_state == AgentRunTerminalState(
        run_finish_reason="limit_reached",
        session_status="limit_reached",
    )


def test_map_run_terminal_state_maps_normal_to_stop_completed():
    terminal_state = map_run_terminal_state(unknown_terminated=False, limit_reason=None)

    assert terminal_state == AgentRunTerminalState(
        run_finish_reason="stop",
        session_status="completed",
    )
