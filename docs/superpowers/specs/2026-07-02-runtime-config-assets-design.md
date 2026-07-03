# Runtime Config Assets Design

## 背景

Fusion 目前已经把模型目录收敛到 LiteLLM Proxy，`fusion-api` 不再维护本地模型表。这条边界需要保留：模型是否存在、底层 provider、基础 metadata 仍以 LiteLLM `/model/info` 为准。

但产品/Agent 策略和可运营 Prompt 资产仍散落在代码里：

- 模型能力展示和推荐解释由 `fusion-ui` 自行计算，前后端容易漂移。
- Agent 搜索预算、深读上限、来源排序权重和域名分层写死在多个 Python 模块里。
- Prompt 文案分布在 `agent_loop.py`、`templates.py`、`tools.py`、`limit_summary.py`、`continuation.py` 等位置。

这次迁移目标不是把所有常量都落库，而是增加一个可回退、可版本化的运行时配置层，把可运营策略和文案资产从业务代码中抽出来。

## 设计目标

1. LiteLLM 仍然是模型目录事实源，Fusion 不恢复 `model_sources/providers`。
2. Fusion 新增运行时配置表，统一承载策略 payload 和 Prompt payload。
3. 所有配置都有代码默认值，DB 读取失败时不能影响聊天链路。
4. 后端负责派生模型展示字段，前端只渲染后端给出的展示结构。
5. Agent 策略读取同一份配置 profile，避免搜索预算、深读策略和 ranker 权重互相漂移。
6. Prompt 资产先通过本地 DB adapter 接入，后续迁移 PromptHub 时只替换 adapter，不改调用点。
7. `AgentSession.config` 记录本次实际使用的策略版本，便于回溯真实对话。

## 非目标

- 不提供完整后台管理 UI。
- 不让前端直接读 DB 配置。
- 不把 secrets、服务 URL、Redis/DB/storage/auth 配置落库。
- 不把 SSE 协议字段、事件名、安全脱敏和 URL policy 改成可配置。
- 不在自动测试里真实调用付费 LLM。

## 数据模型

新增 `runtime_config_entries`：

| 字段 | 说明 |
|---|---|
| `id` | UUID 字符串主键 |
| `namespace` | 配置域，例如 `agent_strategy`、`model_presentation`、`prompt_template` |
| `key` | 配置键，例如 `default`、`generate_title` |
| `version` | 人类可读版本，例如 `2026-07-02.v1` |
| `payload` | JSONB 配置内容 |
| `is_active` | 是否当前启用 |
| `description` | 简短说明 |
| `created_at` / `updated_at` | 审计时间 |

约束：

- `namespace + key + version` 唯一。
- 读取时只取 `is_active=true` 的最新一条。
- 如果 DB 无记录、payload 非法或 DB 异常，使用代码默认值。

## 配置域

### `model_presentation/default`

后端根据模型 capabilities、health、context window 和配置权重生成：

```json
{
  "score": 100,
  "level": "recommended",
  "headline": "推荐：实时资料、图片和长任务",
  "reasons": ["可处理普通文本任务", "可联网搜索并读取关键来源"],
  "warnings": [],
  "labels": [{"key": "network", "text": "可联网", "tone": "success"}],
  "tooltip": "DeepSeek V4 Flash\n推荐：实时资料、图片和长任务\n..."
}
```

前端仍保留本地 fallback，避免旧 API 或异常响应导致选择器不可用。

### `agent_strategy/default`

统一承载：

- search budget：不同 intent 的搜索条数、注入上下文条数、follow-up 预算。
- intent keywords：保守意图推断关键词。
- network budget：最大搜索轮次、最大读取次数、repair 搜索预算、弱结果阈值。
- read planner：quick/freshness/deep 的推荐深读数量和需核验 reason。
- source ranker：权威媒体、低优先级、视频/论坛域名列表，打分权重和优先级阈值。
- tool context：搜索上下文最多注入来源数、单域名限制、url_read 内容截断长度。

安全策略保留代码：URL policy、tracking 参数清洗、内部错误脱敏不落库。

### `prompt_template/<name>`

每个 prompt 单独一条：

- `generate_title`
- `generate_suggested_questions`
- `file_analysis`
- `file_content_enhancement`
- `app_identity`
- `tool_usage_contract`
- `no_tool_network_boundary`
- `no_vision_file_boundary`
- `limit_summary`
- `continuation_system`
- `url_read_tool_description`

PromptHub 接入时，`PromptManager` 和 agent prompt getters 改为调用 PromptHub adapter，代码调用点不再变化。

## 数据流

1. Alembic 创建 `runtime_config_entries` 并 seed 默认配置。
2. `app.core.runtime_config` 读取 DB active payload，和代码默认值做 deep merge；`app.services.runtime_config_service` 仅保留兼容 re-export。
3. 服务/API 层读取 `agent_strategy/default` 后，把 model runtime overrides 注入 `litellm_catalog` 的纯 normalize 函数。
4. `/api/models/` 构造 card 时追加 `capabilityPresentation`。
5. `fusion-ui` `convertApiModelToModelInfo()` 保存 `capabilityPresentation`，模型选择器优先渲染它。
6. Agent 搜索、深读、ranker、tool handler 从 `agent_strategy` 读取配置。
7. PromptManager 和 agent prompt getters 从 `prompt_template` 读取模板，失败时使用默认文案。
8. Agent run start 时把策略版本写入 `AgentSession.config.runtime_config_versions`。

## 回退和失败处理

- DB 查询失败：记录 warning，返回默认配置。
- 单条 payload 非 dict 或字段类型不符合预期：忽略该覆盖字段，保留默认值。
- 前端缺少 `capabilityPresentation`：继续使用本地 fallback 逻辑。
- Prompt 模板缺失变量：保持当前 `ValueError`，这是调用方参数错误，不静默吞掉。

## 测试矩阵

### 后端单元测试

- `runtime_config_service`：DB 命中、无记录 fallback、payload deep merge、异常 fallback。
- `model_presentation`：全能力模型、无联网模型、健康异常模型、长上下文标签。
- `litellm_catalog`：DB override 禁用某模型 agent tools；metadata 显式值仍优先。
- `search_budget`：配置覆盖 search count、followup count、intent keyword 生效。
- `network_budget`：配置覆盖 max search/read、planned search limit、repair count 生效。
- `search_read_planner`：配置覆盖 quick/freshness/deep read limit 生效。
- `source_candidate_ranker`：配置覆盖 authority/low priority domain、score weight、priority threshold 生效。
- `PromptManager`：DB 模板覆盖默认模板，缺失时 fallback。
- Alembic smoke：迁移文件包含表和默认 seed key。

### 前端单元/组件测试

- `modelConfig`：保留 API 返回的 `capabilityPresentation`。
- `modelCapabilityPresentation`：优先使用后端展示字段；缺失时 fallback。
- `ModelSelectorPanel`：能力分、标签、tooltip 来自后端字段，且不内联展示解释。

### 真实回归

部署到 dev 后使用已有登录态 Chrome 复用现有标签，不新开 Chrome/标签：

- `/chat/new` 模型选择器可打开，能力标签和 tooltip 正常。
- 创建一条无需联网的简单对话，不出现固定搜索计划。
- 创建一条需要最新事实的对话，能自主搜索、深读少量高价值来源。
- 刷新对话页后回答依据和执行过程不丢失。
- 记录 URL、输入、预期、实际、console error、刷新结果和结论。
