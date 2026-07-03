---
name: fusion-next-step
description: Use when the user asks Fusion “下一步”, “接下来做什么”, “还有什么优化”, “还能怎么加强”, roadmap direction, or whether to continue product/infrastructure work.
---

# Fusion 下一步建议

这个 skill 的目标是防止重复建议已经完成的工作。不要依赖 Codex memory 或印象来回答下一步。

## 必须步骤

1. 先读当前仓库的 `docs/EXECUTION_LEDGER.md`。
2. 如果存在兄弟仓，读取：
   - `/Users/sean/code/fusion/fusion-api/docs/EXECUTION_LEDGER.md`
   - `/Users/sean/code/fusion/fusion-ui/docs/EXECUTION_LEDGER.md`
3. 运行并阅读最近提交：
   - `git -C /Users/sean/code/fusion/fusion-api log --oneline -40`
   - `git -C /Users/sean/code/fusion/fusion-ui log --oneline -40`
4. 用 `rg` 搜索用户提到的关键词，范围至少包括：
   - `docs/superpowers`
   - `docs/MODEL_ACCEPTANCE_RUNBOOK.md`
   - 相关源码目录
5. 回答必须包含：
   - 已完成事实：只列和当前问题相关的事实。
   - 不应重复建议：明确指出哪些方向已经做过。
   - 可选下一步：只给当前证据支持的新方向；没有就说没有。

## 禁止事项

- 禁止直接凭 memory 或上下文印象回答。
- 禁止把执行台账里已经完成的方向包装成新建议。
- 禁止在没有查 `git log` 和文档时建议“多模型测验”“Search Planner”“配置落库”“CI 门禁”等历史方向。
- 禁止为了显得有计划而硬凑下一步；没有高置信方向时直接说当前不建议继续开基础设施优化坑。

## 输出格式

```markdown
我先查了执行记录。结论：

- 已完成：...
- 不应重复：...
- 当前可考虑：...

我的建议：...
```
