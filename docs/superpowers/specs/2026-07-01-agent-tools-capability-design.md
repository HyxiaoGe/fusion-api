# Agent Tools Capability Design

## 背景

多模型真实回归暴露了两个不同问题：

- 部分模型不具备或不稳定支持 Fusion agent 工具路径，例如 `deepseek-reasoner` 没有 function calling，`qwen-vl-max` 虽有 function calling 元数据但在 autonomous search 场景只口头说需要搜索。
- `functionCalling` 只能说明模型支持通用函数调用，不等价于“适合作为 Fusion 默认联网 agent”。

因此需要单独的能力位，让模型目录、后端运行时和测验脚本都能区分“通用函数调用能力”和“Fusion agent 工具能力”。

## 决策

新增派生能力 `capabilities.agentTools`：

- `agentTools=true`：后端可向模型下发 Fusion agent 工具，如 `web_search` 和 `url_read`。
- `agentTools=false`：后端不下发 agent 工具，模型按普通文本/多模态模型运行。
- 若 LiteLLM metadata 显式提供 `agentTools`，优先尊重该值。
- 若 metadata 未提供，默认 `agentTools=functionCalling`，并对真实回归确认的不稳定模型做短期 denylist。

当前短期 denylist：

- `qwen-vl-max`：有 function calling，但 autonomous search 回归未触发工具。

`deepseek-reasoner` 当前 metadata 已是 `functionCalling=false`，因此自然得到 `agentTools=false`。

## 运行时边界

`agent_loop_request_prep.build_agent_loop_call_config()` 只有在 `functionCalling && agentTools` 同时为真时才下发工具和 `tool_choice=auto`。

这会同步影响：

- 初始 `run_started.tools`
- 工具一致性 system prompt 注入
- URL 预处理失败后的 `url_read` fallback
- eval runner 对 expected tool 场景的判定

## 不包含

- 不把该能力直接写入 LiteLLM 远端 metadata；当前先在 fusion-api 归一化层短期实现。
- 不调整 prompt。
- 不隐藏模型，也不改变健康状态。

## 验收

- `/api/models` 返回 `capabilities.agentTools`。
- `qwen-vl-max` 默认 `agentTools=false`。
- `agentTools=false` 时不向 LLM 下发 `web_search/url_read`。
- 多模型 eval 中，非 agent 模型不再被 autonomous search 工具期望误判。
