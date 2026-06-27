# 2026-06-28 Agent Loop Run Completion 拆分计划

## 背景

`runner.generate_to_redis()` 在 request prep、runtime/state、driver 拆出后，仍直接处理五类终态收尾：

- completed
- superseded
- user_cancelled
- failed
- finally fallback

这些路径都混合了 assistant message 落库、agent run 终态事件、session cache、SSE finalize 和兜底写状态。它们不属于 loop 状态机本身，但对协议正确性很关键。

## 目标

把终态收尾移动到 `app/services/stream/agent_loop_run_completion.py`：

- runner 保留 try/except/finally 的外层编排
- completion helper 负责各终态路径的落库、run finalizer、SSE finalize 顺序
- 通过依赖注入传入 `persist_message`、`finalize_stream` 和 run finalizer，保留现有测试 patch 路径

## 测试点

新增 `test/services/stream/test_agent_loop_run_completion.py` 覆盖：

1. completed：persist -> complete run -> finalize success
2. superseded：persist -> interrupt -> finalize error
3. cancelled：有内容才 persist，interrupt 失败要吞掉并继续 finalize
4. failed：有内容才 persist，fail event 失败要吞掉并继续 finalize
5. fallback：已发终态时不重复写 fallback

## 非目标

- 不改变 `run_finalizer.py` 的事件/session 写入顺序。
- 不改变 `run_agent_loop()` 或工具执行行为。
- 不启动本地 Fusion 服务。
