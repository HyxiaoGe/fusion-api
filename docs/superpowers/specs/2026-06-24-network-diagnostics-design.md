# 会话级联网诊断设计

## 背景

Fusion 的联网回答已经有两类信息：

- 用户可见资料层：assistant message content 中的 `search` / `url_read` block，前端已经统一到“回答依据”区域和侧栏。
- 运行过程层：Redis Stream 里的 `agent_event`，以及 PostgreSQL 中的 `agent_sessions`、`agent_steps`、`tool_call_logs`。

当前缺口是“会话级回看”。实时回答期间，前端可以从 `agent_event` 还原工具过程；但刷新页面、切换会话或管理员排查历史回答时，前端只稳定拿到 message content，不能稳定回答：

- 这次回答到底用了哪些联网工具。
- 每个工具调用耗时多少。
- 哪些工具失败或降级，原因是什么。
- 管理员如何从某条回答追到 run、step、tool call 和 trace。

本设计把这些信息收敛成一个只读 diagnostics 聚合层。第一阶段不新增诊断表，不把诊断数据写入 message content，而是基于现有 agent 表按消息聚合。

## 目标

- 用户在历史联网回答中能看到简明诊断摘要，例如 `搜索 3 次 · 读取 2 个网页 · 1 个降级 · 用时 4.2s`。
- 用户能看到失败、降级、跳过、策略拒绝、超时等原因的脱敏文案。
- 管理员能看到单个工具调用级别的明细：工具名、目标、状态、耗时、错误、run_id、trace_id、step_number、tool_call_log_id。
- 刷新页面、切换会话、重新打开历史对话后，诊断信息仍可查询。
- 实时回答阶段继续复用现有 `agent_event` 和 Agent timeline，不回退现有体验。
- 回答依据侧栏可以承载“联网诊断”分区，但正文区域保持低干扰。

## 非目标

- 不新增新的诊断表或大规模迁移历史数据。
- 不把完整工具参数、网页正文、搜索结果正文暴露给普通用户。
- 不改 LLM 工具调用策略、不改搜索服务和 reader-service 的行为。
- 不替换现有 Agent timeline；实时过程和历史诊断是两个数据源。
- 不做独立管理员后台页面；第一阶段只提供 API 和聊天页入口。
- 不启动本地 Fusion dev server 验证。

## 推荐方案

采用“后端 diagnostics API + 前端回答依据侧栏诊断分区”的方案。

后端新增只读聚合服务，从 `agent_sessions`、`agent_steps`、`tool_call_logs` 拼出某条 assistant message 的联网诊断。普通用户只能查看自己会话内消息的脱敏摘要；管理员在同一个接口里获得扩展字段。

前端在“回答依据”侧栏中增加低权重“联网诊断”分区。它不是日志面板：默认展示摘要和异常原因，管理员再展开工具明细。

## 备选方案

### 方案 A：只增强前端实时 timeline

复用 `agent_event` 和 Redux `currentRun`，直接把 `duration_ms` 显示出来。

优点：前端改动小。

缺点：刷新、切换会话、管理员回看时数据会丢；只能覆盖“正在回答”的过程态，不满足会话级诊断。

### 方案 B：新增 message-level diagnostics content block

生成结束时把工具诊断作为新的 message content block 写入 assistant message。

优点：历史回放简单，前端随消息一起拿到诊断。

缺点：message content 会承担运行日志职责；管理员查询、权限区分、后续扩展都不如表聚合清晰。第一阶段不采用。

### 方案 C：基于现有 agent 表新增 diagnostics 聚合 API

从已存在的 agent 表读取和聚合诊断，前端按 message 拉取。

优点：符合当前数据归属；支持用户摘要和管理员明细；不污染正文内容。

缺点：需要修正 `tool_call_logs.message_id` 关联，并新增读模型/API/前端接入。

推荐采用方案 C。

## 后端设计

### 数据源

诊断聚合读取现有三张表：

- `agent_sessions`
  - `id`：run_id / trace_id。
  - `conversation_id`。
  - `message_id`。
  - `status`。
  - `total_steps`。
  - `total_tool_calls`。
  - `total_duration_ms`。
  - `limit_reason`。
  - `error_message`。

- `agent_steps`
  - `trace_id`。
  - `step_number`。
  - `status`。
  - `tool_calls_count`。
  - `tool_names`。
  - `duration_ms`。

- `tool_call_logs`
  - `conversation_id`。
  - `message_id`。
  - `tool_name`。
  - `status`。
  - `duration_ms`。
  - `input_params`。
  - `output_data`。
  - `error_message`。
  - `trace_id`。
  - `step_number`。
  - `created_at`。

### 必修修正：message_id 关联

当前 `execute_tools_parallel(... message_id=assistant_message_id)` 已经传入消息 ID，但 `BaseToolHandler.log()` 的签名没有 `message_id` 参数。实现前必须修正：

- `BaseToolHandler.log()` 增加 `message_id: str | None = None`。
- 调用 `log_tool_call()` 时传入 `message_id=message_id`。
- `test/test_tool_executor.py::ToolExecutorMessageIdTests` 应覆盖该行为。

没有这一步，诊断接口只能按 conversation/run 猜测工具归属，无法可靠按某条 assistant message 查询。

### API

新增用户可用接口：

```http
GET /api/chat/conversations/{conversation_id}/messages/{message_id}/diagnostics
```

权限规则：

- 必须登录。
- `conversation_id` 必须属于当前用户。
- `message_id` 必须属于该会话，且消息角色为 `assistant`。
- 普通用户返回脱敏模型。
- `current_user.is_superuser == true` 时，在同一响应中附加管理员字段。

第一阶段不新增独立 admin endpoint。这样前端只接一个接口，后端通过当前用户权限决定返回粒度。

### 响应模型

建议新增 `app/schemas/network_diagnostics.py`：

```python
class NetworkDiagnosticsSummary(BaseModel):
    total_duration_ms: int | None = None
    total_steps: int = 0
    total_tool_calls: int = 0
    search_calls: int = 0
    url_read_calls: int = 0
    success_count: int = 0
    failed_count: int = 0
    degraded_count: int = 0
    interrupted_count: int = 0
    limit_reason: str | None = None
    run_status: str | None = None


class NetworkDiagnosticsToolItem(BaseModel):
    tool_call_log_id: str
    tool_name: str
    status: Literal["success", "failed", "degraded", "interrupted"]
    duration_ms: int | None = None
    target: str = ""
    result_count: int | None = None
    reason: str | None = None
    started_at: datetime | None = None
    admin: dict[str, Any] | None = None


class NetworkDiagnosticsResponse(BaseModel):
    conversation_id: str
    message_id: str
    run_id: str | None = None
    visibility: Literal["user", "admin"]
    summary: NetworkDiagnosticsSummary
    tools: list[NetworkDiagnosticsToolItem]
    is_empty: bool = False
```

字段口径：

- `target`
  - `web_search`：来自 `input_params.query`。
  - `url_read`：来自 `input_params.url`，展示层可只显示域名。
  - 其他工具：使用工具名或安全摘要。
- `result_count`
  - 优先读取 `output_data.result_count`。
  - 搜索成功但无 count 时可从 `output_data.sources` 长度派生。
  - URL 读取不强行写 1，避免和“读取页面数量”语义冲突。
- `reason`
  - 优先 `error_message`。
  - `degraded` 缺失时返回 `部分内容不可用，已降级处理`。
  - `failed` 缺失时返回 `未取得可用内容`。
  - `interrupted` 缺失时返回 `工具调用已中断`。

### 管理员字段

普通用户不能看到原始 `input_params`、`output_data`、trace 细节。管理员可在 `admin` 字段看到：

```python
{
    "trace_id": "...",
    "step_number": 1,
    "input_params": {"query": "..."},
    "error_message": "...",
    "created_at": "...",
}
```

管理员字段仍不能包含网页正文、搜索正文、prompt、LLM 回复正文、凭据、内部 URL 的敏感参数。

### 空状态

如果消息没有 `agent_sessions` 或 `tool_call_logs`：

- 返回 200。
- `is_empty=true`。
- `summary` 所有计数为 0。
- `tools=[]`。

前端据此不显示诊断分区。这样兼容旧消息、普通非联网消息和非流式回答。

### 错误处理

- 会话不存在或不属于当前用户：404。
- 消息不存在、不属于该会话或不是 assistant：404。
- 聚合读取异常：返回 500，并写后端错误日志。
- 某条 tool log 数据字段不完整：跳过危险字段，保留该工具的状态和错误摘要，不让整条诊断失败。

## 前端设计

### 数据获取

新增 API client：

- `src/lib/api/chatDiagnostics.ts`
- 函数：`getMessageNetworkDiagnostics(conversationId, messageId)`

聊天页展示某条 assistant message 时不立即批量拉取所有历史诊断。第一阶段采用懒加载：

- 用户打开“回答依据”侧栏时，若该消息可能存在联网信息，则请求 diagnostics。
- 如果 `answerEvidenceSidebar` 已经有来源或异常项，优先显示来源，再显示诊断分区。
- 如果没有来源但 diagnostics 返回失败/降级工具，也允许显示轻量入口。

懒加载避免每次切换会话额外拉取大量 diagnostics，也符合最近前端优化方向：减少空白时间，真实内容优先显示。

### UI 接入

在现有 `AnswerEvidenceSidebar` 中增加分区：

1. 顶部仍显示回答依据摘要。
2. 已使用来源和异常来源保持现有逻辑。
3. 下面新增 `联网诊断` 分区：
   - 普通用户默认展示一行摘要。
   - 有异常时展示异常原因列表。
   - 管理员显示“展开明细”，每个 tool call 一行。

建议文案：

- `联网诊断 · 搜索 3 次 · 读取 2 个网页 · 用时 4.2s`
- `1 个工具降级`
- `网页读取失败：reader-service 暂时未返回内容，已跳过网页读取`

### 实时与历史数据边界

- 正在流式回答时，Agent timeline 继续读取 Redux `currentRun`。
- 历史诊断分区不从 `currentRun` 派生，而是从 diagnostics API 派生。
- 如果刚完成回答且 Redux 仍有 `currentRun`，侧栏也应优先以 API 为准；API 还没写入完成时展示加载态或稍后重试。

### 前端类型

新增 `src/types/networkDiagnostics.ts`：

```ts
export interface NetworkDiagnosticsSummary {
  total_duration_ms: number | null;
  total_steps: number;
  total_tool_calls: number;
  search_calls: number;
  url_read_calls: number;
  success_count: number;
  failed_count: number;
  degraded_count: number;
  interrupted_count: number;
  limit_reason?: string | null;
  run_status?: string | null;
}

export interface NetworkDiagnosticsToolItem {
  tool_call_log_id: string;
  tool_name: string;
  status: "success" | "failed" | "degraded" | "interrupted";
  duration_ms: number | null;
  target: string;
  result_count?: number | null;
  reason?: string | null;
  started_at?: string | null;
  admin?: Record<string, unknown> | null;
}

export interface NetworkDiagnosticsResponse {
  conversation_id: string;
  message_id: string;
  run_id: string | null;
  visibility: "user" | "admin";
  summary: NetworkDiagnosticsSummary;
  tools: NetworkDiagnosticsToolItem[];
  is_empty: boolean;
}
```

## 测试策略

### 后端

新增或更新测试：

- `test/test_tool_executor.py`
  - 验证 `execute_tools_parallel()` 将 `message_id` 传给 `handler.log()`。
  - 验证 `BaseToolHandler.log()` 最终传给 `log_tool_call(message_id=...)`。

- `test/test_network_diagnostics.py`
  - 普通用户只能查询自己的会话消息。
  - 非 assistant message 返回 404。
  - 无诊断数据返回 `is_empty=true`。
  - 有 search/url_read 日志时正确聚合计数、耗时、结果数。
  - failed/degraded reason 有错误原文或兜底文案。
  - 管理员响应包含 `admin` 字段，普通用户不包含。

### 前端

新增或更新测试：

- `src/lib/api/chatDiagnostics.test.ts`
  - 请求路径和响应解析正确。
  - 404/500 返回可展示错误状态。

- `src/components/chat/networkDiagnosticsModel.test.ts`
  - summary 文案正确。
  - failed/degraded/interrupted 工具进入异常列表。
  - admin 字段只在 visibility 为 `admin` 时进入明细模型。

- `src/components/chat/AnswerEvidenceSidebar.test.tsx`
  - 有 diagnostics 时显示“联网诊断”分区。
  - 无来源但有工具失败时仍能展示诊断入口。
  - 管理员明细可展开，普通用户不展示原始参数。

## 验收标准

- 某条历史联网回答刷新后仍能看到联网诊断摘要。
- 诊断摘要能说明用了哪些工具、总耗时、成功/失败/降级数量。
- 失败和降级原因对普通用户可理解且不泄露原始参数或正文。
- 管理员能看到每个工具调用的 run、step、tool_call_log_id、耗时和错误。
- 普通非联网回答不显示空诊断区域。
- 实时 Agent timeline 不受影响。
- 旧回答没有诊断数据时不报错。
- 后端目标测试通过，前端相关组件和模型测试通过。

## 实施顺序

1. 后端修正 `BaseToolHandler.log()` 的 `message_id` 关联。
2. 后端新增 diagnostics schema、聚合 service、API 和测试。
3. 前端新增 diagnostics 类型、API client、纯模型和测试。
4. 前端把 diagnostics 接入 `AnswerEvidenceSidebar`，默认懒加载。
5. 回归现有联网回答来源侧栏、Agent timeline、历史会话刷新。

## 风险与约束

- `tool_call_logs.message_id` 修复只能保证新回答可按消息精确查询。旧日志如果没有 message_id，第一阶段不做历史补偿。
- 如果工具日志异步写入晚于 assistant message 完成，侧栏首次打开可能拿到空诊断。前端需要允许刷新或短暂 loading；后端接口保持幂等。
- 普通用户可见字段必须保持脱敏，不返回 `input_params` 和 `output_data` 原始对象。
- diagnostics API 是只读路径，不应影响发送消息、SSE、Redis Stream、工具执行主链路。
