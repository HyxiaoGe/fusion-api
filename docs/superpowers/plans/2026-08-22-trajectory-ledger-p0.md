# Fusion Trajectory P0 实施计划

> Required sub-skill: `superpowers:subagent-driven-development`
> Spec: `docs/TRAJECTORY_DESIGN.md` v0.11
> Base: `origin/master@b478045`
> Branch: `feat/trajectory-ledger-p0`

**目标：** 在不改变现有聊天 SSE、进度快照和 UI 的前提下，为 Fusion Agent 增加可持久化、可判定完整性的脱敏事件账本，并补齐 LLM round、知识库检索、工具 attempt 与 run attempt 协议。

**本轮范围：** 仅 P0。不得实现轨迹快照 API、投影器、MVP Adapter、fusion-ui 侧栏，也不得替换现有“过程”视图。

**总体架构：** `AgentEventEmitter` 仍是控制面事件唯一发送方；`AgentEventCompositeWriter` 保持 Redis required/fail-closed，依次旁路 progress snapshot 与新增 `TrajectoryRecorder`。Recorder 使用独立短 Session、只追加 `agent_events` 并原子维护 `run_trajectory_meta`，通过有界线程池和内存 degraded latch 隔离主链路。编排层在全部同步辅助事件之后统一 seal + finalize。

## Global Constraints

- `docs/TRAJECTORY_DESIGN.md` 是规格权威；计划与实现冲突时以规格为准并在 SDD ledger 记录 Ruling。
- 所有事件携带 `schema_version=1`；原 `protocol_version` 字段保持兼容。
- Redis 写入语义保持不变：权威 `plan_snapshot` 失败继续 fail-closed；Progress 与 Trajectory 是 auxiliary sink，不得把异常反向传播到主链路。
- sequence 必须在 payload 校验完成后、第一次可取消 `await` 前预留；complete run 连续 `0..N-1`，degraded run 允许空洞但不得重复或复用。
- Recorder 固定使用 `ThreadPoolExecutor(max_workers=4)` + `threading.BoundedSemaphore(4)` 非阻塞准入；主链路等待 `0.25s`；超时/取消使用 `asyncio.shield`，底层 worker 负责释放 permit。
- DB 超时固定为 `statement_timeout=200ms`、`lock_timeout=100ms`、`connect_timeout=1s`；连接超时不是主隔离手段。
- 事件 INSERT 与 meta 计数必须同事务；Recorder 不得持有或跨线程使用请求 Session。
- degraded latch 优先于 COUNT/MIN/MAX；迟到成功不得恢复 complete；`CancelledError` 必须置 latch 后重新抛出。
- `app/ai` 禁止新增对 `app/services` 的依赖；生命周期事件由 Service 编排层发出或通过无框架依赖的回调/观测结果注入。
- payload 必须经过按事件类型的单点 allowlist；prompt、完整输入输出、工具 schema、精确位置和上游私密错误文本不得入账。
- P0 只做 FK CASCADE，不做独立 TTL。
- 不启动本地 Fusion 服务；实现阶段使用单测/集成测试，发布后再按仓库流程做真实登录回归。
- 每个任务先写失败测试，再实现最小代码，随后跑目标测试与相关回归；提交信息中文且包含 `Co-Authored-By`。

## Task 1：冻结实施依据，升级事件协议与 Emitter 序列语义

**Files**

- Modify: `.gitignore`
- Add/track: `docs/TRAJECTORY_DESIGN.md`
- Add/track: `docs/superpowers/plans/2026-08-22-trajectory-ledger-p0.md`
- Modify: `app/services/agent/events.py`
- Modify: `app/services/agent/emitter.py`
- Modify: `test/services/agent/test_events.py`
- Modify: `test/services/agent/test_emitter.py`

**Steps**

1. 在 `test_events.py` 先加入失败断言：现有和新增事件均输出 `schema_version=1`；新增 union 能判别 `llm_round_*`、`retrieval_*`、`tool_attempt_*`；边界字段（1-based index、非负 token/duration、允许状态）拒绝非法值。
2. 在 `test_emitter.py` 先加入失败测试：
   - writer 在 await 中取消后，下一事件使用更大 sequence；
   - payload 校验失败不消耗 sequence；
   - `seal_and_get_last_sequence()` 与 emit 共用锁，返回 `-1` 或最后已预留序号；
   - seal 后任何 emit 抛 `RuntimeError`；
   - 新生命周期 helper 的父级字段和安全字段正确。
3. 在 `events.py` 给 `AgentEventBase.schema_version` 设置固定默认值 `1`，加入 v0.10 §3.3 的 11 个生命周期模型并扩展 `AnyAgentEvent`。
4. 在 `emitter.py` 将 sequence 递增移动到 payload 序列化/体积校验之后、writer await 之前；增加 `_sealed` 与 `seal_and_get_last_sequence()`；新增 LLM/retrieval/tool attempt helper，所有字符串摘要先使用既有安全截断函数或明确长度限制。
5. 确认 21 个旧事件的 JSON 形状只增 `schema_version`，`protocol_version` 不删除。
6. Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/agent/test_events.py test/services/agent/test_emitter.py -q`
7. Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff check app/services/agent/events.py app/services/agent/emitter.py test/services/agent/test_events.py test/services/agent/test_emitter.py`
8. Commit: `feat: 升级轨迹事件协议与序列封口`

## Task 2：增加账本模型、迁移、run attempt 原子分配

**Files**

- Modify: `app/db/models.py`
- Add: `alembic/versions/e8f5a1c4d2b7_add_agent_trajectory_ledger.py`
- Modify: `app/services/agent/session_cache.py`
- Modify: `app/schemas/chat.py`
- Modify: `app/services/stream/agent_loop_execution.py`
- Modify: `app/services/stream/agent_loop_wiring.py`
- Modify: `app/services/stream/runner.py`
- Modify: `app/services/chat_service.py`
- Add: `test/test_agent_trajectory_migration.py`
- Modify: `test/services/agent/test_session_cache.py`
- Modify: `test/test_chat_request_message_ids.py`
- Modify: `test/services/agent/test_continuation.py`

**Steps**

1. 先写迁移 SQL 形状测试，断言 revision 以当前 head `d7e4a9c2f1b6` 为 `down_revision`，并覆盖：
   - `agent_sessions.previous_run_id` 自引用 FK `ON DELETE SET NULL`；
   - `turn_message_id/attempt_index` 可空及 `(turn_message_id, attempt_index)` 部分唯一索引；
   - 回填先令历史 `turn_message_id = message_id`，再使用 `row_number() over (partition by turn_message_id order by created_at, id)`，`message_id IS NULL` 不回填；
   - 历史 `previous_run_id` 不推断；
   - 新建 `agent_events`、`run_trajectory_meta`，conversation/run FK 均 `ON DELETE CASCADE`；
   - 新建单例 `trajectory_ledger_settings` 并由迁移写入环境实际 `ledger_enabled_at`（TIMESTAMPTZ），downgrade 精确删除该表和新增对象。
2. 在 `models.py` 增加 `AgentEvent`、`RunTrajectoryMeta`、`TrajectoryLedgerSettings`，字段、唯一约束和索引严格匹配 v0.11 §4；给 `AgentSession` 增加 `turn_message_id`、`previous_run_id`、`attempt_index` 和关系/索引；保留既有 `message_id` 的 assistant 锚点语义。
3. 先在 `test_session_cache.py` 写失败测试：新 run 必须在一个事务内锁定 conversation 行、按稳定 `turn_message_id` 分配 `max(attempt_index)+1`、解析兼容期 previous run、唯一冲突可有界重试；已有 run 的幂等更新不得重置 turn/lineage/index。
4. 将 `write_session_started()` 扩展为原子 attempt 分配入口，新增 `turn_message_id` 与 optional `previous_run_id`；initial 为 1，retry/regenerate/continue 递增；请求缺 previous 时只在同一加锁事务中按 turn 解析最新 run并记录兼容指标。
5. 给 `ChatRequest` 增加 optional `previous_run_id`，并将现有稳定 `user_message_id` 作为 `turn_message_id` 沿 `chat.py → chat_service.py → runner.py → agent_loop_wiring.py → execution/lifecycle start` 透传；continue 从被续写 assistant 的前序 user message解析同一 turn，并复用现有 `previous_run_id`。不得让旧客户端因缺字段失败。
6. 加并发测试：两个独立 Session 对同一 `turn_message_id` 分配 attempt，不产生重复；另加“首次生成失败后重试换 assistant id 仍属于同一 turn”的回归。若 SQLite 不能表达 `FOR UPDATE`，使用 SQL/调用契约测试并在 PostgreSQL migration test 中校验唯一约束。
7. Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/test_agent_trajectory_migration.py test/services/agent/test_session_cache.py test/test_chat_request_message_ids.py test/services/agent/test_continuation.py -q`
8. Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff check app/db/models.py app/services/agent/session_cache.py app/schemas/chat.py app/services/stream/agent_loop_execution.py app/services/stream/agent_loop_wiring.py app/services/stream/runner.py app/services/chat_service.py test/test_agent_trajectory_migration.py test/services/agent/test_session_cache.py test/test_chat_request_message_ids.py test/services/agent/test_continuation.py`
9. Commit: `feat: 增加轨迹账本模型与运行尝试层级`

## Task 3：实现 TrajectoryRecorder、allowlist、latch 与完整性 barrier

**Files**

- Add: `app/services/agent/trajectory_payload.py`
- Add: `app/services/agent/trajectory_recorder.py`
- Modify: `app/services/stream/tool_executor.py`
- Modify: `app/services/stream/agent_loop_execution.py`
- Add: `test/services/agent/test_trajectory_payload.py`
- Add: `test/services/agent/test_trajectory_recorder.py`
- Modify: `test/test_tool_executor.py`
- Modify: `test/services/stream/test_agent_loop_execution.py`

**Steps**

1. 在 `test_trajectory_payload.py` 先为每个事件类型建立 allowlist 契约测试；加入恶意附加字段、prompt、完整 content、工具 schema、URL query、错误密钥文本，断言落库 DTO 只保留允许字段并执行上限截断。
2. 在 `test_trajectory_recorder.py` 先写失败测试覆盖：
   - 独立 Session，event INSERT + meta 创建/计数同事务；commit 前故障整体回滚；重复 `(run_id, sequence)` 不重复计数；
   - `BoundedSemaphore(4)` 满载立即返回并 latch `admission_full`；submit 失败释放 permit；worker 正常/异常/迟到均释放；
   - `wait_for(shield(future), 0.25)` 真超时，迟到异常被消费；worker 未启动和运行中取消均 latch `recorder_cancelled`、re-raise、permit 最终守恒；
   - latch 首次原因稳定，后续事件跳过 DB，迟到成功不清 latch；
   - finalize 先读 latch，再验证 COUNT/MIN/MAX；complete 写 expected/finalized，mismatch 写 degraded；最后事件迟到后仍 degraded。
3. 在 `trajectory_payload.py` 建立按 `event_type` 的唯一 allowlist 入口；未知事件 fail-open 到 degraded，但不得原样入库。
4. 在 `trajectory_recorder.py` 实现进程级 executor/semaphore、独立 Session factory、同步 worker、late future consumer、per-run latch、`record_chunk()` 与 `finalize(expected_last_sequence)`。DB 连接/事务设置使用规格固定超时。
5. 修改 `AgentEventCompositeWriter`：顺序必须是 `await Redis` → progress recorder → `await trajectory_recorder.record_chunk()`；Redis 异常原样抛出，两个 auxiliary sink 自己 fail-open。
6. 在 `_build_execution_parts()` 创建每 run Recorder 并注入 writer；同时把 Recorder 放入 execution/completion context，供统一 finalize 使用。
7. Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/agent/test_trajectory_payload.py test/services/agent/test_trajectory_recorder.py test/test_tool_executor.py::AgentEventCompositeWriterTests test/services/stream/test_agent_loop_execution.py -q`
8. Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff check app/services/agent/trajectory_payload.py app/services/agent/trajectory_recorder.py app/services/stream/tool_executor.py app/services/stream/agent_loop_execution.py test/services/agent/test_trajectory_payload.py test/services/agent/test_trajectory_recorder.py test/test_tool_executor.py test/services/stream/test_agent_loop_execution.py`
9. Commit: `feat: 实现有界轨迹账本记录器`

## Task 4：把 seal + finalize 接入全部 run 终态

**Files**

- Modify: `app/services/stream/agent_loop_execution.py`
- Modify: `app/services/stream/agent_loop_lifecycle.py`
- Modify: `app/services/stream/agent_loop_run_completion.py`
- Modify: `app/services/stream/agent_loop_wiring.py`
- Modify: `app/services/stream/run_finalizer.py`
- Modify: `test/services/stream/test_agent_loop_lifecycle.py`
- Modify: `test/services/stream/test_agent_loop_run_completion.py`
- Modify: `test/services/stream/test_run_finalizer.py`

**Steps**

1. 先写终态矩阵失败测试：completed、limit_reached、failed、interrupted、superseded、user cancellation、shutdown 均只 seal/finalize 一次；`run_completed` 自身不 finalize；`suggested_questions_pending` 在 seal 前；seal 后无事件。
2. 增加统一 `commit_trajectory_barrier()`：在 lifecycle 的最外层 `finally` 中执行 `seal_and_get_last_sequence()` + `trajectory_recorder.finalize(last_seq)`，以有界 `asyncio.wait_for(asyncio.shield(...))` 保护；重复进入幂等。
3. 保持现有业务终态/Redis stream 终态优先级：barrier 失败只使轨迹 degraded，不替换原始 `CancelledError`、`StreamOwnershipLostError` 或业务异常。
4. 对开始前失败、session 行不存在、辅助问题领取失败分别测试并明确：能 seal 的一律 seal；无法建 meta 的由 latch/stale 路径判 degraded。
5. Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/stream/test_agent_loop_lifecycle.py test/services/stream/test_agent_loop_run_completion.py test/services/stream/test_run_finalizer.py test/services/agent/test_emitter.py test/services/agent/test_trajectory_recorder.py -q`
6. Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff check app/services/stream/agent_loop_execution.py app/services/stream/agent_loop_lifecycle.py app/services/stream/agent_loop_run_completion.py app/services/stream/agent_loop_wiring.py app/services/stream/run_finalizer.py test/services/stream/test_agent_loop_lifecycle.py test/services/stream/test_agent_loop_run_completion.py test/services/stream/test_run_finalizer.py`
7. Commit: `feat: 接入轨迹完整性提交屏障`

## Task 5：接入 LLM round、知识库 retrieval、工具 attempt 生命周期

**Files**

- Modify: `app/ai/llm_round_observability.py`
- Modify: `app/schemas/chat.py`
- Modify: `app/services/stream/agent_round.py`
- Modify: `app/services/stream/llm_stream.py`
- Modify: `app/services/stream/agent_loop_lifecycle.py`
- Modify: `app/services/stream/tool_executor.py`
- Modify as required: `app/services/stream/limit_summary.py`
- Modify: `test/services/stream/test_llm_round_observability.py`
- Modify: `test/services/stream/test_agent_round.py`
- Modify: `test/services/stream/test_llm_stream.py`
- Modify: `test/services/stream/test_agent_loop_lifecycle.py`
- Modify: `test/test_tool_executor.py`

**Steps**

1. LLM：先写失败测试覆盖 started → first delta → completed/failed/cancelled；first delta 识别 reasoning/content/tool_call；空流不发 first delta 且 completed.ttft_ms 为 null；reasoning/content 可见 Redis chunk 必须先于 first delta event，tool_call-only 无此前置要求。
2. 将 `LLMRoundObservation` 保持为纯测量组件，输出结构化 started/first delta/finish 数据或接受无 `services` 依赖的 async callback；`agent_round.py`/`limit_summary.py` 的 Service 层负责调用 emitter。
3. 扩展 `Usage` 或内部 usage extraction，兼容供应商字段提取 `cache_read_tokens/cache_write_tokens`；缺失/非法时为 null，不报错，不破坏现有 Usage 序列化。
4. Retrieval：在 `_prepare_knowledge_grounding()` 周围发 started/completed/failed/cancelled，query 只存 ≤120 字安全摘要；真实取消必须发 cancelled 后 re-raise。
5. Tool attempt：把 attempt 生命周期放进每一次真正 handler 执行，而非整个 tool_call；同 tool_call 1-based 递增，timeout/cancelled/failed/success 正确结束。不要改变现有 backoff 次数与 ToolResult 语义。
6. 增加架构测试：`app/ai` 无任何新增 `app.services` import。
7. Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/stream/test_llm_round_observability.py test/services/stream/test_agent_round.py test/services/stream/test_llm_stream.py test/services/stream/test_agent_loop_lifecycle.py test/test_tool_executor.py -q`
8. Run: `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff check app/ai/llm_round_observability.py app/schemas/chat.py app/services/stream/agent_round.py app/services/stream/llm_stream.py app/services/stream/agent_loop_lifecycle.py app/services/stream/tool_executor.py app/services/stream/limit_summary.py test/services/stream/test_llm_round_observability.py test/services/stream/test_agent_round.py test/services/stream/test_llm_stream.py test/services/stream/test_agent_loop_lifecycle.py test/test_tool_executor.py`
9. Commit: `feat: 补齐轨迹生命周期事件`

## Task 6：实现 stale 收敛、legacy 判定、性能与全量回归门禁

**Files**

- Add: `app/services/agent/trajectory_reconciliation.py`
- Modify: `app/services/scheduler_service.py`
- Add: `test/services/agent/test_trajectory_reconciliation.py`
- Add: `test/services/agent/test_trajectory_performance.py`
- Add: `test/test_scheduler_service.py`
- Add: `reports/trajectory-p0-baseline.md`

**Steps**

1. 先写 reconciliation 失败测试：仅扫描 stale `recording`；run 已终态且 expected 存在并通过三断言才 complete；expected 缺失、校验不符或新 run 缺 meta 均 degraded；无 meta 时只以数据库持久化 `ledger_enabled_at` 区分 legacy 与 meta_missing。
2. 实现幂等协调函数并挂入现有 scheduler 体系；不得实现 TTL、删除事件或新增 P1 API。
3. 建立无需真实网络的性能门禁：测 Redis writer stub + progress + recorder fast path 的 p50/p95/p99、并发连接/permit 上限、Recorder timeout 时首个可见 SSE 仍先到；将测试方法和本机基线写入 `reports/trajectory-p0-baseline.md`。
4. 运行 P0 完整目标集：
   `/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/agent test/services/stream/test_agent_loop_execution.py test/services/stream/test_agent_loop_lifecycle.py test/services/stream/test_agent_loop_run_completion.py test/services/stream/test_agent_round.py test/services/stream/test_llm_stream.py test/services/stream/test_llm_round_observability.py test/services/stream/test_run_finalizer.py test/test_tool_executor.py test/test_chat_request_message_ids.py test/services/agent/test_continuation.py test/test_agent_trajectory_migration.py test/test_scheduler_service.py -q`
5. 运行全量：`/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/ -q`
6. 运行静态检查：`/Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff check app test`
7. 检查迁移单头：`/Users/sean/code/fusion/fusion-api/.venv/bin/python -m alembic heads`，必须仅有新 revision。
8. Commit: `test: 完善轨迹账本回归与性能门禁`

## Task 7：对抗审查、修复、推送与发布后验收

**Files**

- 仅修改审查发现直接涉及的 P0 文件。
- 更新：`reports/trajectory-p0-baseline.md`

**Steps**

1. 按 SDD 流程生成从 `b478045..HEAD` 的全分支 review package，由高能力 Reviewer 对照 v0.10 §11 做 spec + quality 审查。
2. 所有 Critical/Important findings 进入一次统一修复波次并做 scoped re-review；不得把 P1–P3 建议扩入本分支。
3. 重跑受影响测试、全量 pytest、ruff、Alembic heads；记录精确命令和结果。
4. 精确暂存并核对 `git diff --cached`，提交审查修复（如有）；确认主 checkout 与 fusion-ui 无改动。
5. Push: `git push -u origin feat/trajectory-ledger-p0`；创建 PR，等待并核对 GitHub Actions。推送/PR 不是合并、部署或验收。
6. 若仓库自动部署到授权 dev 环境，核对 CI 对应 commit、运行镜像/健康；不手动触发生产部署。
7. 使用已有登录 Chrome 标签，以新建真实对话覆盖：普通 LLM 流、tool retry、知识库检索、取消各一条；从数据库/日志核对事件序列、meta 状态和安全 payload。P0 无 UI，因此页面只验聊天/现有“过程”无回归。
8. 将真实模型事件类型、延迟对比、degraded/complete 结果写入 `reports/trajectory-p0-baseline.md`，不得记录密钥、prompt 或完整用户内容。
