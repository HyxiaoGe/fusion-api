# 主聊天 Prompt Runtime v2

## 目标

在不引入独立路由模型和完整 Skills runtime 的前提下，修正主聊天初始系统提示词过度携带产品域规则的问题，并准确区分 Run 初始 canonical prompt 与每个 LLM Round 的 effective system messages。

## 已确认的设计判断

- 自然语言语义判断由主 LLM 完成；服务端规则只负责权限、模型能力、显式模式和少量高置信优化，规则未命中不能等同于用户没有对应意图。
- 第一轮继续向主 LLM 提供经过权限与模型能力过滤的工具 schema，避免“我现在在北京，我想去上海”这类自然表达因正则未命中而失去路线工具。
- 工具可见不再自动意味着产品域完整 system prompt 常驻。工具选择、参数和调用前约束优先由工具 description/schema 与后端校验承担。
- 产品工具结果的事实边界跟随 ToolMessage 结果进入上下文。高德结果已经携带按工具区分的 usage contract 与最终回答约束；FlyAI 结果必须携带完整航班/高铁事实边界。
- 固定核心与固定运行规则排在动态日期和用户偏好之前，为相同能力配置保留尽可能长的稳定前缀。
- Run 初始快照只代表 canonical prompt；每轮实际 system 内容继续以 `llm_round_started.system_prompt_fingerprint` 关联，不把初始正文冒充为每轮完整请求。

## 本期范围

### API

1. `system_prompt.py` 的组装顺序调整为：
   - `app_identity`
   - 本 Run 的可信固定段落（工具、计划、研究、无能力边界等）
   - `current_date`
   - 非空 `user_preferences`
2. 初始 canonical prompt 不再按“工具已公告”注入 `amap_fact_boundary` 和 `flyai_travel_fact_boundary`。
3. 高德与 FlyAI 工具 schema 继续提供给主 LLM；既有权限、模型能力、Deep Research 调度及高置信工具收窄保持不变。
4. 原 `AMAP_FACT_BOUNDARY_SYSTEM_PROMPT` 不再保留为不可达的整段常量。路线、天气与组合行程的调用前规则下沉到对应工具 description/schema；高德结果继续携带按工具区分的 usage contract。
5. FlyAI 工具 description 承接组合行程日期门禁，`format_llm_context()` 在实际取得结果后加入完整事实边界与接驳后续规则。该内容属于不可信工具结果的本地可信包装，不提升外部数据权限。
6. 更新模板版本；Run 快照仍使用现有独立表和鉴权详情接口，不新增数据库表。

### UI

1. 把系统提示词节点明确命名为“Run 初始系统提示词”。
2. 正文详情说明它是 Run 创建时保存的 canonical prompt，不代表后续每轮追加的语言、修复、研究或总结规则。
3. 模型请求详情把现有 fingerprint 标为“当轮实际系统消息指纹”。
4. 复制、刷新、历史快照、权限和失败状态沿用现有行为。

## 明确不做

- 不实现 Skills 目录、`describe_skill`、`load_skill`、权限提升或 continuation 恢复。
- 不增加 LLM Router、Embedding Router 或新的路由服务。
- 不删除或收窄低置信请求可见的已授权工具 schema。
- 不新增每轮完整请求正文存储、内容寻址表或数据库迁移。
- 不修改现有 Agent Run、Context Manager、工具预算和 SSE 重连语义。
- 不提交、推送、合并或发布。

## 验收合同

### Prompt 组装

- 无工具、无偏好：`app_identity` 在前，`current_date` 在后。
- 有固定能力段落和用户偏好：所有固定段落位于 `current_date` 和 `user_preferences` 之前。
- 默认公告 Web、高德、FlyAI、Plan 工具时，Run 初始 section IDs 不包含 `amap_fact_boundary`、`flyai_travel_fact_boundary`。
- 用户偏好仍然追加且不能覆盖基础规则。
- 模板版本和 Run 初始 fingerprint 随正文变化更新。

### 语义召回边界

- 对“我现在在北京，我想去上海，你可以帮我吗”，即使现有正则没有命中显式路线策略，`route_compare` 仍保留在主 LLM 的工具 schema 中。
- 明确路线任务的现有高置信收窄继续只公告路线工具。
- Deep Research 的阶段工具限制保持不变。

### 产品事实边界

- 高德成功结果继续在 ToolMessage 本地包装中包含对应地点、路线或天气 usage contract。
- FlyAI 成功结果在 ToolMessage 本地包装中包含航班/高铁事实边界；失败结果不得编造班次、时间或价格。
- 组合行程的唯一日期门禁、目的地天气选择和单班次接驳规则分别保留在 FlyAI/高德工具描述及实际班次结果包装中，不依赖 Run 初始大 Prompt。
- 产品工具同批次、参数预检、接驳路线延后和确定性收口行为不改变。

### Trajectory

- Run 初始节点和详情不再暗示其正文等于每轮实际请求。
- LLM Round 详情继续展示对应 `system_prompt_fingerprint`，文案明确为当轮实际 system 消息指纹。
- 旧 Run、快照损坏、持久化降级、复制失败及权限隔离行为不回归。

## 验证边界

- 单元测试、Ruff、ESLint、Vitest 和构建只能证明代码与静态契约。
- 不启动本地 Fusion 服务。
- 未发布 dev 前不能声称真实模型对自然语言路线、组合行程或产品事实边界已经验收。
- 后续真实模型验收必须至少覆盖问候、自然表达路线、天气、航班、高铁、组合行程、失败工具和多轮追问；两条对话不能代表通过。
