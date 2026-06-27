# Agent run 生命周期边界拆分计划

## 背景

`runner.py` 的终态收尾已经交给 `run_finalizer.py`，但启动阶段仍分散在 runner 里：先直接写 `agent_sessions` started 行，后面再直接 emit `run_started`。这让 run 生命周期的顺序约束仍由 runner 隐式维护。

## 目标

1. 在 lifecycle 边界中新增 `start_agent_run`，明确 started session 与 `run_started` event 的顺序。
2. 让 runner 只调用生命周期函数，不再直接组合 session start 和 run start event。
3. 保持现有 agent loop 行为、终态路径和 Redis Stream 事件契约不变。

## 步骤

1. 先补 `start_agent_run` 单元测试，断言先写 session started，再 emit `run_started`。
2. 实现 `start_agent_run`，扩展现有 emitter/session_cache Protocol。
3. 在 runner 中接入 `start_agent_run`，移除分散的 started 写入和 `run_started` emit。
4. 跑聚焦 agent loop/stream/lifecycle 测试、全量 unittest、ruff、架构检查与 diff 检查。
5. 提交、推送并通过正常 CI/CD 合入 dev。
