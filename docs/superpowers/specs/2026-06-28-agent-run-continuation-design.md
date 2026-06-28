# Agent Run 触顶后继续执行设计

## 背景

Fusion 的 agent loop 已经拆成稳定的基础层：后端能发出 `run_started`、`step_started`、`tool_call_*`、`run_limit_reached`、`run_completed` 等事件，前端也能用 Agent timeline 展示运行过程和终态 banner。

当前缺口是触顶后的用户体验。用户遇到 `max_steps`、`max_tool_calls` 或 `timeout` 时，系统会给出强制总结，但这不是用户真正想要的终点。现在的“重试运行/重新提问”入口要么是占位，要么接近重新发送问题，不能表达“基于已有过程继续完成”。

本设计把 `limit_reached` 后的“继续查”做成真正的 continuation run：在同一条 assistant 消息上追加新的运行结果，用新的 `run_id` 继续写事件和日志，避免污染对话历史，也避免把续跑伪装成普通新问题。

## 目标

- 用户在回答触顶后可以点击“继续查”，让同一条 assistant 消息继续补充结果。
- 首版只支持 `limit_reached`，覆盖 `max_steps`、`max_tool_calls`、`timeout` 三种原因。
- 每次继续执行追加一份原始运行预算，用户不需要先理解预算配置。
- 继续执行不能改写已有回答，只能向原 assistant 消息追加新的 content blocks。
- 继续执行拥有新的 `run_id`，但归属同一个 `assistant_message_id`。
- 原回答内容即使 continuation 失败也必须保留。
- 实时 SSE、断线重连、历史消息刷新后都能看到追加后的完整 assistant 消息。

## 非目标

- 不做普通失败重试。
- 不做单个 tool call 的定点重试。
- 不做预算弹窗、深度继续、多档执行模式。
- 不合并多个 run 的内部事件序列号；每个 run 仍保持自己的 sequence。
- 不改普通 `/api/chat/send` 的用户消息创建语义。
- 不启动本地 Fusion 服务验证；实现阶段按代码检查、测试、CI/CD、远端 dev 和真实 Chrome 回归完成。

## 推荐方案

采用“后端 continuation API + 同 assistant 消息追加 content blocks + 前端 continuation CTA”的方案。

前端只在已触顶的 assistant 消息上显示“继续查”。点击后调用新的 continuation API。后端校验该消息属于当前用户、角色为 assistant、最近一次 run 是 `limit_reached`，然后用同一个 `assistant_message_id` 初始化 Redis Stream 和后台任务。后台任务复用 agent loop 基础层，但注入 continuation 上下文，让模型明确“基于已有回答和工具结果继续补充，不要重写前文”。

## 备选方案

### 方案 A：前端自动发送“请继续”

优点是实现最快，可以复用现有 `/send`。

缺点是会新增 user message，污染会话历史；模型也只能从普通对话上下文猜测续跑意图，不能复用原 assistant 消息和 run 归属。这个方案不采用。

### 方案 B：新 assistant 消息承载继续结果

优点是后端落库简单。

缺点是用户会看到两条 assistant 回答，续跑结果和原回答割裂。用户已经选择同一条消息追加，因此这个方案不采用。

### 方案 C：真正 continuation run

优点是产品语义正确，后端 run 归属清晰，前端体验自然。

缺点是需要新增 API、权限校验、消息追加和前端 run 归属处理。

推荐采用方案 C。

## 用户体验

### CTA 展示条件

“继续查”只在满足以下条件时展示：

- 消息角色为 assistant。
- 该消息存在 agent run 信息。
- run 终态为 `limit_reached`。
- `limitReachedReason` 是 `max_steps`、`max_tool_calls` 或 `timeout`。
- 当前会话没有其他流正在运行。
- 前端已拿到可用于 continuation 的 `conversationId` 和 `assistantMessageId`。

不在 `failed`、`interrupted`、`incomplete` 首版展示“继续查”。这些状态后续用失败重试设计处理。

### CTA 文案

- `max_steps`：`继续查`
- `max_tool_calls`：`继续查`
- `timeout`：`继续查`

辅助文案保持现有触顶解释，但从“可能未完整覆盖问题”调整为可行动的表达，例如：

- `已达最大步数，本次回答已先总结。可以继续执行来补充遗漏。`
- `工具预算用完，本次回答已基于现有信息总结。可以继续查更多资料。`
- `运行超时，本次回答可能不完整。可以继续执行剩余部分。`

### 继续中的状态

点击后，同一条 assistant 消息下展示新的运行中 timeline。已有正文保留在上方，新 delta 继续追加到消息末尾。

如果 continuation 成功，触顶 banner 消失或降级为历史摘要，不再阻挡阅读。新的 run 若再次触顶，可以再次显示“继续查”。

如果 continuation 失败，保留已有内容，并在同一条消息下显示失败 banner。失败 banner 不删除上一次已完成内容。

## 后端 API

新增接口：

```http
POST /api/chat/conversations/{conversation_id}/messages/{message_id}/continue
```

请求体：

```json
{
  "previous_run_id": "可选，前端已知时传入",
  "stream": true
}
```

首版不开放自定义预算。后端从上一轮 `run_started.config` 或当前默认配置恢复一份预算。

响应：

- 与 `/api/chat/send` 的流式响应一致，返回 `text/event-stream`。
- 首个有效 agent 事件仍是 `run_started`。
- `run_started.message_id` 必须等于路径中的 `message_id`。
- `run_started.run_id` 是新的 run。

权限和校验：

- 必须登录。
- `conversation_id` 必须属于当前用户。
- `message_id` 必须属于该会话。
- 消息角色必须是 `assistant`。
- 该消息最近一次 agent session 必须是 `limit_reached`。
- 如果传了 `previous_run_id`，它必须属于该 `message_id`，且状态为 `limit_reached`。
- 如果该会话已有 streaming meta，返回 409，避免同一会话并发写同一条消息。

错误响应建议：

- 404：会话或消息不存在。
- 400：消息不是 assistant，或最近一次 run 不可 continuation。
- 409：该会话当前已有运行中 stream。

## 后端执行模型

### continuation request

新增一个清晰的请求模型，例如：

```python
class ContinueAgentRunRequest(BaseModel):
    previous_run_id: str | None = None
    stream: bool = True
```

ChatService 增加 continuation 方法，职责是：

- 查会话和消息权限。
- 查最近一次 agent session。
- 计算 continuation 预算。
- 初始化 Redis Stream，`message_id` 使用原 assistant 消息 ID。
- 启动后台任务。
- 返回 `stream_redis_as_sse(conversation_id, message_id)`。

### 消息追加

现有 `persist_message()` 已经支持按 `assistant_message_id` 更新已有消息。continuation 需要在 agent loop 初始状态中放入原 assistant 消息已有 content blocks，再让后续 round 继续 append。

实现上应新增一个读取和归一化 helper：

- 输入：已有 assistant message。
- 输出：可继续追加的 `content_blocks`。
- 要求：保持已有 block 顺序和 id，不重建旧 block，不删除旧 usage。

最终落库仍通过 `persist_message()` 更新同一条 message content。

### continuation prompt

后端需要向 LLM 注入系统级 continuation 约束，不能只依赖用户自然语言。建议加入一条 system message：

```text
你正在继续上一轮因运行上限而停止的回答。请基于已有对话、已有回答和已有工具结果继续补充，不要重写或总结已完成的部分。若需要更多资料，可以继续调用可用工具。输出应自然衔接在上一段回答之后。
```

还需要把已有 assistant 内容放回 LLM messages 中，让模型知道前文已经写了什么。普通 message builder 已能把 conversation messages 转成 LLM messages；continuation 的关键是不要新增 user message，也不要把 continuation 指令落入用户消息历史。

### 预算策略

首版采用“一次继续 = 一份原始预算”：

- `max_steps`：沿用上一轮 config。
- `max_tool_calls`：沿用上一轮 config。
- `timeout_s`：沿用上一轮 config。

如果上一轮 config 缺失，则使用当前默认 agent loop limits。

continuation run 的计数从 0 开始，但前端展示时应能看出这是同一 assistant 消息的后续运行。

### session 和日志

continuation 必须新建 agent session：

- `run_id` 新生成。
- `message_id` 仍为原 assistant message ID。
- `conversation_id` 不变。
- `status` 按新 run 自己的结果写入。

tool call 日志继续写同一个 `message_id`，以便后续诊断能按消息聚合多次运行。

## 前端设计

### API client

新增：

```ts
continueAgentRunStream({
  conversationId,
  messageId,
  previousRunId,
}, callbacks, signal)
```

它复用 `parseSseEnvelopeStream`，但请求路径是 continuation API。

### hook

`useSendMessage` 不应继续膨胀。建议新增 `useContinueAgentRun`：

- 接收 `conversationId`、`assistantMessageId`、`previousRunId`。
- 调用 continuation API。
- 复用现有 stream callbacks 分发 `initRun`、`pushStep`、`pushToolCall`、`appendTextDelta`、`finalizeRun`。
- `onReady` 不创建新 assistant message，只绑定已有 message。
- `onDone` 把 streamSlice 中的新旧完整 blocks 写回原消息。

如果发现 callback 逻辑和 `useSendMessage` 重复过多，实现阶段应先抽共享 callback builder，而不是复制一份大函数。

### 状态归属

当前 `AgentRunTimeline` 通过 `run.messageId` 或 `serverMessageId` 匹配 assistant message。continuation 继续沿用这一规则：

- `run.messageId = assistantMessageId`
- `run.serverMessageId = assistantMessageId`

这样新的 run 会自然挂在原消息下面。

### 历史多 run 展示

首版可以只展示当前或最近一次 run 的 timeline，不强制做历史多 run 列表。原因是已有完整内容会落在同一条消息里，用户最需要看到的是当前 continuation 是否还在跑、是否再次触顶。

如果后续要展示“第 1 次运行 / 继续 1 次 / 继续 2 次”，再扩展历史 run 查询接口。

## 数据流

1. 用户看到 assistant 消息触顶 banner。
2. 用户点击“继续查”。
3. 前端调用 continuation API。
4. 后端校验会话、消息和上一轮 run 状态。
5. 后端用原 `assistant_message_id` 初始化 Redis Stream。
6. 后端后台任务加载已有 assistant content blocks，注入 continuation prompt，启动新的 agent loop。
7. 前端消费 SSE，新的 run 事件挂到同一条 assistant 消息。
8. 新的 reasoning/content/tool blocks 追加到原消息末尾。
9. run 完成后，后端更新同一条 assistant message；前端刷新 hydration 后仍看到完整内容。

## 错误处理

- continuation API 返回 400：前端 toast 或 banner 提示“这条回答不能继续执行”。
- 返回 409：提示“当前会话已有回答正在生成，请结束后再继续”。
- SSE 中途失败：保留已有内容，结束当前 stream 状态，显示失败 banner。
- 用户停止 continuation：调用现有 stop stream，run 标记为 `interrupted`，已有追加内容保留。
- continuation 再次触顶：同一条消息继续显示“继续查”。

## 测试计划

### 后端

- API 权限测试：非本人会话、非 assistant 消息、无 limit run、错误 previous_run_id。
- continuation 初始化测试：`init_stream` 使用原 assistant message ID。
- agent session 测试：新 run 写入同一 message_id。
- persistence 测试：已有 content blocks 保留，新 blocks 追加，不重建旧 block id。
- prompt 测试：continuation system instruction 注入，但不落库为 user message。
- 触顶续跑集成测试：第一次 `limit_reached`，第二次 continuation `stop`，同一 message content 增长。

### 前端

- `RunBanner`：`limit_reached` 三种 reason 显示“继续查”。
- `continueAgentRunStream`：请求路径和 SSE callback 解析正确。
- hook 测试：continuation 不创建新 assistant message，delta 追加到原 message。
- 页面测试：当前会话 streaming 时禁用“继续查”。
- 错误测试：400/409/SSE error 显示可读提示，不删除已有内容。

### 回归

- 普通 `/chat/send` 不受影响。
- 断线重连仍能恢复运行中 continuation。
- 历史会话刷新后 assistant 消息包含 continuation 追加内容。
- 真实 Chrome 回归使用已部署 `https://fusion.seanfield.org` 和用户登录态，覆盖一条触顶后继续查路径。

## 验收标准

- 用户能在触顶回答上点击“继续查”。
- 继续执行不会新增 user message，也不会新增 assistant message。
- 同一条 assistant 消息内容变长，旧内容保持不变。
- 后端为 continuation 创建新的 run_id，且 message_id 等于原 assistant message。
- continuation 成功、失败、停止、再次触顶都有明确 UI 状态。
- CI/CD 通过后，dev 环境健康，真实 Chrome 回归通过。
