# Reasoning Tag Filter Design

## 背景

多模型真实回归发现 `MiniMax-M2.7` 会把 `<think>...</think>` 直接写入用户可见正文。该内容来自 LLM streaming 的 `delta.content`，不是标准 `reasoning_content` 字段，因此现有 reasoning/text 分流无法拦截。

## 决策

过滤边界放在 `app/services/stream/llm_stream.py` 的 LLM streaming 消费层：

- 在 `delta.content` 写入 Redis `answering` chunk 之前过滤。
- 同一份过滤后的文本进入 `state.content_buf`，因此最终落库的 `TextBlock` 也不会包含 `<think>`。
- 保留标准 `reasoning_content` 到 `reasoning` chunk 的既有路径。

## 流式约束

`<think>` 可能被拆成多个 chunk，例如 `<thi`、`nk>...`、`</thi`、`nk>`。过滤器必须在开标签尚未闭合前暂存可疑前缀，不能先把 `<thi` 推给前端再尝试修正。

## 不包含

- 不把错误正文里的 `<think>` 转换为可见思考卡片。
- 不调整模型目录能力标记。
- 不改变工具调用策略或 PromptHub 配置。

## 验收

- 完整 `<think>内部</think>正文` 只输出 `正文`。
- 跨 chunk 的 `<think>...</think>` 不向 `answering` 泄漏任何标签片段。
- `content_buf` 和 SSE `answering` 使用同一份过滤后文本。
- 后端 stream 测试、全量测试和架构检查通过。
