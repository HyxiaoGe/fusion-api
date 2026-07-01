# Reasoning Tag Filter Implementation Plan

**Goal:** 在服务端过滤被模型错误写入正文通道的 `<think>...</think>`，避免实时输出和历史消息出现 reasoning 标签。

## Files

- Modify: `app/services/stream/llm_stream.py`
  - 增加状态化正文过滤逻辑。
  - 在 content delta 写入 Redis 和 `content_buf` 前过滤。
- Modify: `test/services/stream/test_llm_stream.py`
  - 覆盖完整标签和跨 chunk 标签两类泄漏。
- Create: `docs/superpowers/specs/2026-07-01-reasoning-tag-filter-design.md`
  - 记录过滤边界和非目标。
- Create: `docs/superpowers/plans/2026-07-01-reasoning-tag-filter.md`
  - 记录执行计划和验收。

## Test Matrix

| Case | 输入 | 预期 |
| --- | --- | --- |
| RTAG-01 | `<think>内部思考</think>可见正文` | SSE `answering` 和 `content_buf` 只保留 `可见正文` |
| RTAG-02 | `<think>` 与 `</think>` 跨多个 chunk | 标签和内部思考不被提前推送 |
| RTAG-03 | 标准 `reasoning_content` | 原 reasoning chunk 逻辑不变 |
| RTAG-04 | 普通文本流 | 原 lock check、usage、tool call 行为不变 |

## Tasks

- [x] 写 RTAG-01 / RTAG-02 失败测试。
- [x] 实现正文通道 reasoning tag 过滤。
- [x] 跑目标测试确认通过。
- [x] 跑 stream 测试、lint、全量后端测试和架构检查。
- [ ] commit + push，监控 CI/CD 和 dev 部署。
- [ ] 部署后使用 `MiniMax-M2.7` 真实 stream 对话回归。
