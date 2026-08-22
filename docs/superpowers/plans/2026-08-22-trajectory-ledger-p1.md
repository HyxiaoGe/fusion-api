# Trajectory P1 历史投影与快照 API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Fusion API 中提供可鉴权、可截断、可审计的历史轨迹读侧，将 P0 脱敏事件账本查询时 O(n) 投影为 records/spans，并暴露普通用户运行列表、普通用户快照和管理员诊断快照。

**Architecture:** 新增独立 `TrajectoryRepository → TrajectoryQueryService → API` 读侧，不复用聊天写流服务。`TrajectoryProjector` 保持无数据库依赖的纯函数，输入有序事件前缀与 run 终态，输出稳定 record/span DTO；完整性状态继续复用 P0 的 `RunTrajectoryMeta`/ledger watermark 契约。普通用户和管理员使用不同端点与不同 DTO，管理员端额外返回按 run 可靠关联并经过现有审计脱敏器处理的 `ToolCallLog` 摘要，同时记录访问审计。

**Tech Stack:** Python 3.11、FastAPI、Pydantic v2、SQLAlchemy、PostgreSQL/SQLite 测试、pytest/unittest、Ruff

**Spec:** `docs/TRAJECTORY_DESIGN.md`

## Global Constraints

- P1 只提供历史快照，不新增 SSE、轮询或其他实时语义，不修改 P3 前端。
- 普通端点固定为 `GET /api/conversations/{conversation_id}/runs` 与 `GET /api/conversations/{conversation_id}/runs/{run_id}/trajectory`；不复用 `/api/chat` 写流前缀。
- 管理员端点固定为 `GET /api/admin/audit/conversations/{conversation_id}/runs/{run_id}/trajectory`，必须使用 `get_conversation_auditor`、`X-Admin-Audit-Reason` 与现有访问审计。
- 普通端点必须在数据库查询中同时约束 `conversation_id + current_user.id`，并校验 run 属于该 conversation；会话不存在、越权、run 不属于会话统一返回 404。
- 普通 DTO 与管理员 DTO 必须是不同 Pydantic 类型；普通端点不得根据角色扩展字段。
- `MAX_TRAJECTORY_EVENTS_PER_RUN` 默认 `5000`；查询必须 `LIMIT max+1`，只投影前 max 条并返回 `truncated=true`。
- `MAX_TRAJECTORY_RUNS_PER_CONVERSATION` 默认 `500`；查询必须 `LIMIT max+1`，只返回最近 max 个 run 并返回 `truncated=true`。
- 投影器必须是 O(n) 纯函数，不查询数据库、不写入 span 表、不修改事件 payload。
- `complete`、`degraded`、`legacy` 与 `truncated` 是独立维度；`complete + pending terminal intent` 对外必须降级，缺 meta 必须按 ledger watermark 判定，不能直接视为 legacy。
- 截断前缀中的未闭合 span 必须标记 `terminal_source=inferred`、`status=unknown`、`inferred_reason=truncated_prefix`，不得使用完整 run 的终态伪装其完成状态。
- 非截断终态 run 的 orphan：成功 run → `unknown/incomplete`，失败 run → `failed`，中断 run → `cancelled`；均标记 `terminal_source=inferred` 与设计规定的 `inferred_reason`。
- 真实终态事件关闭的 span 标记 `terminal_source=recorded`；仍在运行且未截断的 open span 允许 `status=running`、`terminal_source=null`。
- `tool_call_completed`、`step_completed` 即使缺少 started，也必须利用自带 `duration_ms` 形成 recorded span；其他生命周期终态缺 start 只保留 record，不伪造 start。
- annotation 不建 span；record 通过 `span_id` 关联到 run、step、tool、llm、retrieval 或 tool-attempt span，无法精确关联时附着 run span。
- 普通事件 record 只返回 P0 已 allowlist 落库的 payload 副本；不得从消息、ToolCallLog 或工具注册表补充敏感内容。
- 管理员诊断 v1 只额外返回 `ToolCallLog` 中可可靠按 `trace_id=run_id` 关联的脱敏摘要列表；不得声称该列表与 ledger `tool_call_id` 精确一一对应。
- 不启动本地 Fusion 服务；测试先 RED 后 GREEN，最终运行全量 pytest、ruff check、ruff format --check 与 diff check。

### Ruling 1：普通轨迹资源使用 `/api` 根路径

设计文档持续使用 `/api/conversations/...`，轨迹是独立只读资源而非 `/api/chat` 的写流动作，因此新 router 挂载到 `/api`。如果判断错误，代价是 P3 客户端需要调整 base path；不会造成数据迁移或双写。

### Ruling 2：每个会话默认最多返回最近 500 个 run

设计只给出常量名而未给默认值；500 足以覆盖调试历史，同时阻止无界列表查询。若容量估计偏小，后续可通过环境变量上调，不改变响应契约。

### Ruling 3：管理员工具诊断按 run 汇总，不伪造 tool_call 精确关联

`ToolCallLog` 只有 `trace_id + step_number`，没有 ledger `tool_call_id` 外键。P1 返回独立 `tool_calls[]` 诊断列表并明确 `association="run"`；若未来补齐稳定关联键，再在新 schema_version 中增加 span 级关联。

---

### Task 1: 定义 DTO 并实现纯函数 TrajectoryProjector

**Files:**

- Create: `app/schemas/trajectory.py`
- Create: `app/services/agent/trajectory_projector.py`
- Create: `test/services/agent/test_trajectory_projector.py`

**Interfaces:**

- Consumes: 按 `sequence ASC` 排序的 `TrajectoryEventRecord`、`run_status: str`、`run_ended_at: datetime | None`、`truncated: bool`。
- Produces: `TrajectoryProjection(records: list[TrajectoryRecord], spans: list[TrajectorySpan])`；Task 2 将其嵌入 `TrajectorySnapshot`。
- `TrajectoryRecord` 字段：`sequence,event_type,schema_version,timestamp,step_id,tool_call_id,parent_step_id,trace_id,span_id,payload`。
- `TrajectorySpan` 字段：`span_id,kind,name,parent_span_id,start_sequence,end_sequence,started_at,ended_at,duration_ms,status,terminal_source,inferred_reason,ttft_ms,record_sequences`。

- [ ] **Step 1: 写投影失败测试并确认 RED**

  在 `test_trajectory_projector.py` 用手写 literal 事件覆盖：

  1. run/step/tool/llm/retrieval/tool-attempt 正常配对，层级、状态、duration、TTFT 与 record 关联正确；
  2. `tool_call_completed`、`step_completed` 缺 started 时仍依据 duration 生成 recorded span；
  3. plan/evidence/context/suggested annotation 不创建 span，并附着最精确父 span或 run；
  4. 成功、失败、中断终态分别收口 orphan；成功 orphan 必须为 `status=unknown`、`inferred_reason=run_completed_without_close`；
  5. 截断前缀一律使用 `unknown/truncated_prefix`，不能借完整 run 状态关闭；
  6. 缺失 `schema_version` 的历史 record 返回 `schema_version=0`，不抛错；
  7. running run 的 open span 保留 `running + terminal_source=null`；
  8. 事件输入对象在投影后保持不变。

  Run:

  ```bash
  /Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/agent/test_trajectory_projector.py -q
  ```

  Expected: FAIL，原因是 `trajectory_projector`/DTO 尚不存在。

- [ ] **Step 2: 最小实现 DTO 与单遍状态机**

  `project_trajectory()` 只遍历 records 一次；用 `dict[span_id, mutable builder]` 保存 open/closed span，用输出顺序列表保持首次出现顺序。固定 span id：`run:{run_id}`、`step:{step_id}`、`tool:{tool_call_id}`、`llm:{llm_round_id}`、`retrieval:{retrieval_id}`、`tool_attempt:{tool_attempt_id}`。终态处理优先使用事件 payload 的 `duration_ms/status/ttft_ms`，缺失 duration 才用时间差；负时间差归零。

- [ ] **Step 3: 实现 orphan、截断与 annotation 关联**

  终态 run 的推导关闭时间使用 `run_ended_at`，为空时使用最后一条已加载事件时间；截断时始终使用前缀末事件时间并标 `truncated_prefix`。annotation 关联顺序为 `tool_call_id → step_id/parent_step_id → run`；所有关联 sequence 进入目标 span 的 `record_sequences`。

- [ ] **Step 4: 运行聚焦测试并确认 GREEN**

  ```bash
  /Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/services/agent/test_trajectory_projector.py -q
  /Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff check app/schemas/trajectory.py app/services/agent/trajectory_projector.py test/services/agent/test_trajectory_projector.py
  ```

- [ ] **Step 5: 自审并中文提交**

  提交信息：`feat: 实现轨迹历史投影器`，提交必须包含 `Co-Authored-By: Codex <noreply@openai.com>`。

### Task 2: 实现受限查询服务与普通用户 API

**Files:**

- Create: `app/db/trajectory_repository.py`
- Create: `app/services/trajectory_query_service.py`
- Create: `app/api/trajectory.py`
- Create: `test/services/test_trajectory_query_service.py`
- Create: `test/test_trajectory_api.py`
- Modify: `app/api/deps.py`
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Modify: `main.py`
- Modify: `app/services/agent/trajectory_reconciliation.py`
- Modify: `test/services/agent/test_trajectory_reconciliation.py`

**Interfaces:**

- `TrajectoryRepository.list_runs(conversation_id, user_id, limit) -> list[tuple[AgentSession, RunTrajectoryMeta | None]] | None`：会话不存在/越权返回 `None`，成功按最新 run 查询 `limit` 条。
- `TrajectoryRepository.get_run(conversation_id, run_id, user_id) -> tuple[AgentSession, RunTrajectoryMeta | None] | None`：归属约束在查询中完成。
- `TrajectoryRepository.list_events(conversation_id, run_id, limit) -> list[AgentEvent]`：`sequence ASC`，调用方传 `max+1`。
- `TrajectoryQueryService.list_runs(...) -> TrajectoryRunListResponse`。
- `TrajectoryQueryService.get_user_snapshot(...) -> TrajectorySnapshot`。
- `resolve_trajectory_status_from_rows(run, meta, watermark) -> TrajectoryStatusAssessment`：供列表批量状态判定与现有单 run helper 共用。

- [ ] **Step 1: 写状态判定与查询服务失败测试并确认 RED**

  覆盖：

  1. `complete + pending` 对外降级；meta 状态正常复用；无 meta 时 ledger 之前为 legacy、之后为 degraded/meta_missing；
  2. 用户只能读取自己的 conversation，越权和 run 跨 conversation 均得到 service 层 not-found；
  3. events 使用 `max+1`，只投影 max 条且 `truncated=true`；恰好 max 条不截断；
  4. runs 使用 `max+1`，取最近 max 个 group，再按 group 最近时间倒序、组内 `attempt_index` 升序；null attempt 保持 null，不伪装成 1；
  5. snapshot 的 `run` 使用 AgentSession 权威计数/duration，`completeness` 同时报告 status、degraded_reason、meta event_count/expected、loaded_event_count 与 sequence 边界；
  6. legacy run 无 events 时仍返回可解释空快照，而不是 404。

  Run:

  ```bash
  /Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest \
    test/services/agent/test_trajectory_reconciliation.py \
    test/services/test_trajectory_query_service.py -q
  ```

  Expected: FAIL，原因是批量状态 helper、repository/service 尚不存在。

- [ ] **Step 2: 实现 repository 与批量完整性 helper**

  查询只选择 P1 必需行；列表用一次 outer join 加一次 ledger watermark 查询，禁止每个 run 单独 `session.get()`。`resolve_run_trajectory_status()` 改为调用新的纯 helper，保持 P0 既有语义与测试。

- [ ] **Step 3: 实现 query service、上限配置与快照组装**

  在 `Settings` 增加：

  ```python
  MAX_TRAJECTORY_EVENTS_PER_RUN: int = int(os.getenv("MAX_TRAJECTORY_EVENTS_PER_RUN", "5000"))
  MAX_TRAJECTORY_RUNS_PER_CONVERSATION: int = int(os.getenv("MAX_TRAJECTORY_RUNS_PER_CONVERSATION", "500"))
  ```

  service 构造时拒绝非正数配置；事件 DTO 必须从数据库列与 `dict(event.payload)` 新建，缺版本兼容为 0。不得返回 terminal intent 字段。

- [ ] **Step 4: 写普通 API 失败测试并确认 RED**

  使用 TestClient + SQLite 真模型覆盖两个端点、success envelope/request_id、认证依赖、越权 404、run 跨 conversation 404、truncated 响应；断言普通响应不存在 `input_params/output_data/terminal_intent`。

  ```bash
  /Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/test_trajectory_api.py -q
  ```

- [ ] **Step 5: 挂载 router 与依赖并确认 GREEN**

  `app/api/trajectory.py` router 内路径从 `/conversations` 开始；`main.py` 使用 `prefix="/api"`。两个端点均注入 `TrajectoryQueryService` 与 `get_current_user`，service 返回空值时统一 `ApiException.not_found("会话或轨迹不存在，或无权访问")`。

  ```bash
  /Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest \
    test/services/agent/test_trajectory_reconciliation.py \
    test/services/test_trajectory_query_service.py \
    test/test_trajectory_api.py -q
  /Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff check \
    app/db/trajectory_repository.py app/services/trajectory_query_service.py \
    app/api/trajectory.py app/api/deps.py app/core/config.py main.py \
    test/services/test_trajectory_query_service.py test/test_trajectory_api.py
  ```

- [ ] **Step 6: 自审并中文提交**

  提交信息：`feat: 提供轨迹运行列表与快照接口`，提交必须包含 `Co-Authored-By: Codex <noreply@openai.com>`。

### Task 3: 实现管理员诊断 DTO 与访问审计

**Files:**

- Create: `app/schemas/admin_trajectory.py`
- Modify: `app/db/trajectory_repository.py`
- Modify: `app/services/trajectory_query_service.py`
- Modify: `app/services/admin_audit_service.py`
- Modify: `app/api/admin_audit.py`
- Modify: `test/test_admin_audit_api.py`
- Modify: `test/test_admin_audit_service.py`
- Modify: `docs/TRAJECTORY_DESIGN.md`

**Interfaces:**

- `TrajectoryRepository.list_tool_diagnostics(run_id) -> list[ToolCallLog]`：严格按 `trace_id=run_id`，`step_number/created_at/id` 排序。
- `TrajectoryQueryService.get_admin_snapshot(conversation_id, run_id) -> AdminTrajectorySnapshot | None`：无 user 归属过滤，但仍校验 run 属于 conversation。
- `AdminTrajectorySnapshot` 包含普通 `snapshot` 与 `tool_calls`；每个工具项固定含 `association="run"`、`id,message_id,step_number,tool_name,status,duration_ms,model_id,provider,arguments,result_preview,error,redacted_fields,created_at`。
- `AdminAuditService.record_trajectory_view(...)`：写 `action="admin.audit.trajectory.view"`、`resource_type="conversation_run_trajectory"`、`resource_id=run_id`、`target_user_id=conversation.user_id`。

- [ ] **Step 1: 写管理员服务/API 失败测试并确认 RED**

  覆盖：

  1. 非审计员 403；管理员响应 `Cache-Control: private, no-store`；
  2. run 不属于 conversation 返回 404；
  3. 读取成功后新增 `admin.audit.trajectory.view`，reason 被记录但敏感值不复制进审计 metadata；
  4. `ToolCallLog` 仅按 run 返回，使用既有 `_tool_item` 脱敏规则，原始 secret/error 不泄漏；
  5. admin DTO 与普通 DTO 结构不同，但 base snapshot 内容一致；工具项明确 `association=run`，没有伪造 `tool_call_id`；
  6. 审计写入失败返回现有 503 语义，不返回未审计的敏感诊断。

  ```bash
  /Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest \
    test/test_admin_audit_service.py test/test_admin_audit_api.py -q
  ```

  Expected: FAIL，原因是管理员轨迹接口、DTO 与审计 action 尚不存在。

- [ ] **Step 2: 实现管理员 DTO、run 级诊断与审计**

  复用 `AdminAuditService._tool_item()` 生成允许字段；不得直接序列化 `ToolCallLog.input_params/output_data/error_message`。管理员 endpoint 先通过 query service 构造 DTO，再调用公开审计方法；审计失败时整体请求失败。

- [ ] **Step 3: 更新设计文档的 P1 实施契约**

  将文首状态更新为 P0 已 dev 验收、P1 实施中；在 §7 固化普通 `/api` 路径、两个默认上限、普通/admin DTO 形状、管理员 run 级工具关联限制与 `truncated_prefix` 语义。不得提前宣称 P1 已部署或 P3 已开始。

- [ ] **Step 4: 聚焦回归并确认 GREEN**

  ```bash
  /Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest \
    test/services/agent/test_trajectory_projector.py \
    test/services/test_trajectory_query_service.py \
    test/test_trajectory_api.py \
    test/test_admin_audit_service.py \
    test/test_admin_audit_api.py -q
  /Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff check \
    app/schemas/admin_trajectory.py app/db/trajectory_repository.py \
    app/services/trajectory_query_service.py app/services/admin_audit_service.py \
    app/api/admin_audit.py test/test_admin_audit_service.py test/test_admin_audit_api.py
  ```

- [ ] **Step 5: 自审并中文提交**

  提交信息：`feat: 增加管理员轨迹诊断与审计`，提交必须包含 `Co-Authored-By: Codex <noreply@openai.com>`。

## Branch-wide verification

所有任务与逐任务审查通过后，由控制 Agent 运行：

```bash
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m pytest test/ -q
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff check .
/Users/sean/code/fusion/fusion-api/.venv/bin/python -m ruff format --check .
git diff --check origin/master...HEAD
```

随后进行 whole-branch 独立审查。未经额外发布授权，不合并、不部署；若推送/CI 授权沿当前仓库规则有效，则只推进到特性分支 PR 与 CI，不把 CI 成功表述为 dev 验收。
