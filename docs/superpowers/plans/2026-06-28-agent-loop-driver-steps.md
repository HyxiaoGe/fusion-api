# 2026-06-28 Agent Loop Driver Steps 拆分计划

## 背景

`runner.py` 已抽出 runtime/state/request prep 后，`app/services/stream/agent_loop_driver.py` 仍由 `run_agent_loop()` 同时承载：

- 循环顶部 limit 检查与 `run_limit_reached` 事件
- step lifecycle 启动
- 单轮 LLM 调用
- stop / cancelled / tool_calls / unknown 退化分支
- 触顶后的 summary step 调用

这些职责都属于 driver 层，但堆在一个 164 行函数中，后续改触顶策略或工具回合时容易误伤其他分支。

## 目标

把 `run_agent_loop()` 收敛为主状态机骨架，具体分支交给小函数处理。保持外部协议不变：

- `run_agent_loop()` 签名与 `AgentLoopOutcome` 不变
- `runner.py` 不参与本轮拆分
- Redis event、session cache、message append、usage 统计、tool call 计数语义不变

## 实施步骤

1. 在 `test/services/stream/test_agent_loop_driver.py` 补齐 timeout、tool-call、unknown 退化分支单测。
2. 抽出 driver 内部 helper：
   - limit 检查与触顶事件
   - step 启动
   - round 调用与 state 更新
   - stop / cancelled / tool_calls / unknown 分支处理
   - limit summary 调用
3. 跑聚焦测试、架构检查、ruff、全量 unittest。
4. 推送后监控 GitHub Actions 和 dev health。

## 非目标

- 不调整 loop limit 策略。
- 不调整 tool executor、limit summary、llm stream 的内部实现。
- 不启动本地 Fusion 服务。
