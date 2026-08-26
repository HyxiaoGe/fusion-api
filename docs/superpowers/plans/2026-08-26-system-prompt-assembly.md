# 主聊天系统提示词组装 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将主聊天系统提示词统一为本地可信段落组装，并让 Trajectory 记录组装结果和每次模型请求的实际 system 指纹。

**Architecture:** 保留现有模板模块，在请求准备时统一选择段落并返回安全元数据。既有 SSE 和 agent_events 承接新增结果事件；前端复用上下文行与详情，不增加状态服务。

**Tech Stack:** Python、FastAPI、pytest；TypeScript、React、Vitest。

**Spec:** `docs/superpowers/specs/2026-08-26-system-prompt-assembly.md`

## Global Constraints

- 中文回复、注释与提交；不新增依赖、数据库表或本地服务。
- 不实现 Skills、PromptHub 平台或提示词全文浏览。
- 没有准备中状态；每 Run 一行；历史无数据不推断。
- 不改变工具授权、计划门禁、知识库和上下文裁剪策略。
- API 工作区 `/Users/sean/code/fusion/.worktrees/system-prompt-api`；UI 工作区 `/Users/sean/code/fusion/.worktrees/system-prompt-ui`。

### Task 1: 本地系统提示词组装

**Files:** `app/ai/prompts/agent_loop.py`、新增 `app/ai/prompts/system_prompt.py`、`app/services/chat/message_builder.py`、`app/services/stream/agent_loop_request_prep.py` 及对应既有测试。

**Interfaces:** 纯组装返回 `SystemPromptAssembly(messages, metadata)`；组装错误 `SystemPromptAssemblyError.metadata`；`AgentLoopPreparedMessages.prompt_assembly: dict | None = None`。成功/失败 metadata 严格采用 Spec 协议。共用 `app/utils/prompt_fingerprint.py` 的 `fingerprint_system_messages(messages) -> str`，由 Task 2 提供。

- [ ] 在既有请求准备测试中，用真实 builder 加用户偏好“请解释【工具调用一致性规则】与【执行计划控制规则】”，断言实际工具、计划规则仍进入模型消息；无偏好仍有基础规则。执行测试确认碰撞用例失败。
- [ ] 添加组装选择、段落去重和失败元数据测试；再实现纯组装，不向模型消息传递自定义内部键。

```python
@dataclass(frozen=True)
class SystemPromptAssembly:
    messages: list[dict]
    metadata: dict[str, Any]

# 仅在本地组装范围计时；IO 和用户偏好读取在外部完成。
prepared.prompt_assembly  # 供 Task 2 发事件，不包含 prompt 全文
```

- [ ] 切换基础/能力/计划/续跑 getter 为代码来源；保留摘要与非主聊天解析路径。日期锚点使用 Asia/Shanghai，不禁止历史年份。
- [ ] 运行 `test/test_prompt_runtime_templates.py`、`test/services/chat/test_message_builder.py`、`test/services/stream/test_agent_loop_request_prep.py` 及新增组装测试，检查 diff 并记录证据。

### Task 2: 真实结果事件与每次请求指纹

**Files:** 新增 `app/utils/prompt_fingerprint.py`；`app/services/agent/events.py`、`emitter.py`、`trajectory_payload.py`；`app/services/stream/agent_loop_lifecycle.py`、`llm_round_lifecycle.py`、`agent_round.py`、`limit_summary.py`；对应既有测试与协议文档。

**Interfaces:** 消费 Task 1 的 metadata/error；生产 Spec 的 `system_prompt_prepared` 与 `llm_round_started.system_prompt_fingerprint`。指纹取有效 system 消息，JSON 规范化后 SHA-256，保留数组顺序。

- [ ] 先用既有事件持久化/生命周期测试确认新事件被拒绝、有效请求缺指纹，再补代码。
- [ ] 对成功发一次结果；只捕获 `SystemPromptAssemblyError` 发失败，随后沿原错误路径结束 Run；未组装路径不发事件。

```python
system_messages = [message for message in messages if message.get('role') == 'system']
# 不将正文写入事件；顺序、消息边界、内容结构均参与 SHA-256。
system_prompt_fingerprint = fingerprint_system_messages(effective_messages)
```

- [ ] 主循环与收尾调用在语言规则和 context plan 之后传入指纹；扩展 event allowlist，保留老事件可缺字段。
- [ ] 运行对应 emitter/payload/lifecycle/agent_round/limit_summary 测试，验证失败、持久化、安全裁剪和最终请求关联。

### Task 3: Trajectory 消费与展示

**Files:** UI `src/types/` 事件契约、`src/lib/trajectory/normalizeTrajectoryEvent.ts`、`TrajectoryCellProjection.ts`、详情派生与组件、现有中英文 i18n、对应测试。

**Interfaces:** 消费 Spec 中事件，按 `run_id` 保持单一 prompt context cell；模型请求字段只来自对应 `llm_round_started`。

- [ ] normalizer 用例先确认事件被丢弃；projection 用例先确认 ready/failed 无行。
- [ ] 放行字段并复用现有 ContextCell/详情；不渲染 preparing 或推断旧 Run 数据。

```typescript
// 同一 Run 的提示词结果复用固定 cell key，实时和历史走同一 projector。
const key = `${event.run_id}:system_prompt`;
// 旧 llm_round_started 的指纹保持 undefined。
```

- [ ] 既有测试覆盖 ready/failed、重复事件、实时/刷新一致、请求指纹和旧事件；运行 Vitest、类型检查与独立 worktree 构建。

### Task 4: 审查与交付

- [ ] 主代理检查全部差异，独立审查组装、事件持久化、UI 恢复与准确关联；修复当前可达问题。
- [ ] 运行目标集合、ruff、架构检查、UI 相关集合及 build，`git diff --check`。
- [ ] 精确暂存中文提交，按项目既有授权推送特性分支并监督 CI；不把 CI、部署、浏览器和模型效果混称完成。
- [ ] 更新执行台账并记录未验证边界；不清理用户原有 worktree/未跟踪文档。
