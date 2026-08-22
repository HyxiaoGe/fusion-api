# Trajectory P0 异步队列修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 将 Trajectory P0 从 emitter 锁内同步等待 PostgreSQL，切换为每个 run 的有界异步队列，使事件接纳路径满足 p95 ≤ 5ms、p99 ≤ 15ms，同时保持既有脱敏、顺序、降级与 durable terminal intent 正确性。

**Architecture:** 保留已审查的 `TrajectoryRecorder` 作为同步、耐久的落库核心；新增组合式 `QueuedTrajectoryRecorder` 作为生产入口。每个 run 由一个实例和一个单消费者队列保证顺序，`record_chunk()` 只做有界非阻塞接纳，`finalize()` 关闭接纳、执行有界 flush，再调用同步核心完成既有终态握手。队满、flush 超时或 worker 异常统一进入核心的 degraded latch，绝不阻塞 emitter，也不把失败 run 标为 complete。

**Tech Stack:** Python 3.12、asyncio、SQLAlchemy、pytest/unittest、Ruff

**Spec:** `docs/TRAJECTORY_DESIGN.md`

## Global Constraints

- 每个 run 的队列容量必须恰好为 `1000`。
- `AgentEventCompositeWriter` 调用的生产 `record_chunk()` 路径不得等待数据库写入。
- 每个 run 只允许一个后台消费者，必须按接纳顺序严格写入 sequence；跨 run 继续由核心线程池并发。
- 队满必须 fail-open：标记 `degraded/admission_full`、丢弃新事件、立即返回，不得阻塞 emitter 锁。
- `finalize()` 必须先关闭新接纳，再用不超过 `10.0` 秒的 flush barrier 等待已接纳事件定论，最后调用同步核心 `finalize(expected_last_sequence)`。
- flush 超时必须标记 `degraded/recorder_timeout`；取消必须标记 `degraded/recorder_cancelled` 并重新抛出 `CancelledError`；worker 异常必须标记 `degraded/write_failed`。
- 任何失败或丢事件路径都不得声明 `complete`，同步核心现有 durable terminal intent、CAS、late future 与 terminal ack 状态机不得重写。
- 包装器必须透传 `run_id`、`conversation_id`、`message_id`、`degraded_reason`、`degraded_latch()`、`pending_terminal_reconciliation`。
- 不新增数据库迁移，不改变事件协议、payload allowlist、SSE 或 UI。
- 测试必须先 RED 后 GREEN，并保留命令、失败原因与通过输出。
- 不启动本地 Fusion 服务；验证使用测试、CI、已授权的 dev 部署与已打开的登录态 Chrome 标签。

### Ruling 1：flush 超时取 10.0 秒

容量 1000 × dev 同步单写 p50 约 7.4ms，正常最坏 flush 约 7.4 秒；10 秒给出有界余量且不会无限拖住终态。如果估计错误，代价是极端积压 run 被保守标为 degraded，而不是错误 complete。

### Ruling 2：包装同步核心，不改写核心状态机

`TrajectoryRecorder` 已覆盖 durable terminal intent、迟到事务、取消、permit 守恒与 CAS 竞态；异步化只改变接纳/flush 边界。如果该边界判断错误，代价是需要把队列生命周期进一步并入核心，但当前最小改动避免重新验证 1500 行终态逻辑。

## Task 1: 实现每个 run 的有界异步接纳与生产装配

**Files:**

- Create: `app/services/agent/queued_trajectory_recorder.py`
- Create: `test/services/agent/test_queued_trajectory_recorder.py`
- Modify: `app/services/stream/agent_loop_execution.py`
- Modify: `test/services/stream/test_agent_loop_execution.py`
- Test: `test/services/stream/test_agent_loop_lifecycle.py`

**Step 1: 写失败测试并确认 RED**

新增真实行为测试：

1. 同步核心 `record_chunk()` 被慢写阻塞时，包装器的 `record_chunk()` 与 `AgentEventCompositeWriter.append_chunk()` 仍快速返回，证明 emitter 路径不等待数据库。
2. `finalize()` 在所有已接纳事件按 sequence 顺序写完后才调用核心 finalize。
3. 容量为 1 的测试实例在 worker 忙时队满，后续接纳立即返回并令核心 latch 为 `admission_full`。
4. flush 超时令核心 latch 为 `recorder_timeout`，取消令 latch 为 `recorder_cancelled` 并重新抛出；测试结束没有泄漏后台任务。
5. production assembly 创建 `QueuedTrajectoryRecorder(TrajectoryRecorder(...))`，writer、completion context 与 execution 共用同一包装器。

先运行：

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest \
  test/services/agent/test_queued_trajectory_recorder.py \
  test/services/stream/test_agent_loop_execution.py -q
```

确认失败是包装器或装配尚不存在，而不是 fixture/导入错误。

**Step 2: 最小实现包装器**

包装器要求：

- 默认常量 `TRAJECTORY_QUEUE_SIZE = 1000`、`TRAJECTORY_FLUSH_TIMEOUT_SECONDS = 10.0`。
- 构造参数接受同步核心，以及仅供测试覆盖的 `queue_size`、`flush_timeout_seconds`。
- `record_chunk()` 过滤 conversation/chunk type，关闭或核心已 degraded 时直接返回；用 `put_nowait` 非阻塞接纳，并惰性创建单个消费者 task。
- 消费者顺序 `await inner.record_chunk(...)`；发现核心 degraded 后继续排空但不再写入，确保 finalize 可收口。
- 队满调用核心既有降级接口，原因 `admission_full`。
- 消费者异常调用核心既有降级接口，原因 `write_failed`，并保存异常状态供 barrier 感知。
- `finalize()` 先原子关闭接纳，再等待消费者处理完队列；用 `asyncio.wait_for(asyncio.shield(...), 10.0)` 做 flush barrier。超时取消消费者并标 `recorder_timeout`；当前任务被取消时标 `recorder_cancelled`、清理消费者后重新抛出；无论成功或降级，最后调用核心 finalize，只有外部取消不吞异常。
- 提供只读 `inner` 以便装配测试确认组合关系；其余公共字段/属性按 Global Constraints 透传。

**Step 3: 改生产装配**

`_build_execution_parts()` 先构造同步 `TrajectoryRecorder`，再包装为 `QueuedTrajectoryRecorder`；更新相关 dataclass 类型标注与装配断言。不要修改 `AgentEventCompositeWriter` 的 sink 顺序。

**Step 4: 运行聚焦测试并确认 GREEN**

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest \
  test/services/agent/test_queued_trajectory_recorder.py \
  test/services/stream/test_agent_loop_execution.py \
  test/services/stream/test_agent_loop_lifecycle.py -q
```

**Step 5: 回归同步核心**

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest \
  test/services/agent/test_trajectory_recorder.py \
  test/services/agent/test_trajectory_recorder_hardening.py \
  test/services/agent/test_trajectory_terminal_protocol.py -q
```

**Step 6: 自审并提交**

确认无未捕获 task 异常、无取消吞噬、无协议/迁移/UI 改动。提交信息：

```text
fix: 轨迹账本改为有界异步接纳
```

提交必须包含 `Co-Authored-By: Codex <noreply@openai.com>`。

## Task 2: 将性能基准切到生产包装器并固化切换结论

**Files:**

- Modify: `scripts/trajectory_p0_baseline.py`
- Modify: `test/services/agent/test_trajectory_performance.py`
- Modify: `docs/TRAJECTORY_DESIGN.md`

**Step 1: 写失败测试并确认 RED**

将性能 runner/test 的 `trajectory_stub` 改为生产 `QueuedTrajectoryRecorder` 路径，并新增断言：

- 250 次 CompositeWriter 接纳 p95 ≤ 5ms、p99 ≤ 15ms，即使同步核心每次写等待 20ms。
- runner 每轮结束显式 finalize，确保后台 task 不泄漏。
- 失败 sink 不影响 Redis/progress 已可见顺序。

运行：

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/agent/test_trajectory_performance.py -q
```

确认当前 runner 仍使用同步核心，慢写路径导致门限失败。

**Step 2: 修改 runner 与测试**

runner 必须使用生产包装器，且每轮使用唯一 run_id，采样后调用 `finalize(samples_per_path - 1)`。测试可以用可控慢核心隔离数据库，但测量对象必须是 `AgentEventCompositeWriter → QueuedTrajectoryRecorder.record_chunk()` 的生产接纳边界。

**Step 3: 更新设计文档**

只更新与本次已触发切换直接相关的文字：

- §2.1/§5.1/§5.4 将“后续升级/v1 不实现”改为 P0 当前实现。
- 明确包装同步核心、容量 1000、单 run 单消费者、10 秒 flush barrier、队满/超时/取消/worker 异常语义。
- §9.1 记录 dev 同步基线：200 事件，p50 7.428ms、p95 10.781ms、p99 11.124ms，触发切换；不得把本地 stub 当真实 DB 基线。
- P0 范围表注明异步接纳已启用。

不要扩写 P1/P3。

**Step 4: 验证并提交**

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/agent/test_trajectory_performance.py -q
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff check \
  app/services/agent/queued_trajectory_recorder.py \
  app/services/stream/agent_loop_execution.py \
  scripts/trajectory_p0_baseline.py \
  test/services/agent/test_queued_trajectory_recorder.py \
  test/services/agent/test_trajectory_performance.py \
  test/services/stream/test_agent_loop_execution.py
```

提交信息：

```text
docs: 固化轨迹异步队列性能门槛
```

提交必须包含 `Co-Authored-By: Codex <noreply@openai.com>`。

## Branch-wide verification

所有任务与逐任务审查通过后，由控制 Agent 运行：

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/ -q
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff check .
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff format --check .
git diff --check origin/master...HEAD
```

然后进行 whole-branch 独立审查；审查通过后 push、创建 PR、监控 CI、合并到 `master`、等待 dev 发布成功，并在 dev 重跑真实 PostgreSQL 200 事件包装器基准与至少一条真实模型工具链路。dev 验收必须确认：

- 生产镜像 SHA 等于新 merge commit，health 与 Alembic head 正常。
- 包装器接纳 p95 ≤ 5ms、p99 ≤ 15ms；finalize 后 `complete`、event_count=200、expected_last_sequence=199、sequence=0..199。
- 相对基线使用本次同环境同步核心 p95 10.781ms 对比，不伪造缺失的历史端到端多轮基线。
- 真实工具 run 刷新后回答仍在，trajectory 为 complete、无 sequence 空洞；浏览器控制台无 error。
