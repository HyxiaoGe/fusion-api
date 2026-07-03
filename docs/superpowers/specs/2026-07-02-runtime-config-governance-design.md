# Runtime Config 治理闭环 v1 设计

## 背景

`runtime_config_entries` 已经承载 Agent 策略、模型能力展示和 Prompt 模板，但上一阶段只完成“落库 + 读取 + 默认回退”。如果运营配置写坏，主链路会直接消费坏 payload；排查时也缺少统一入口确认某次线上行为使用了哪版配置。

## 目标

v1 做后端治理闭环：

1. 运行时读取时跳过无效 active 版本，自动回退到上一条有效 active 版本；如果全无效则回退代码默认值。
2. 提供轻量 schema 校验，覆盖 `agent_strategy/default`、`model_presentation/default`、`prompt_template/*`。
3. 提供 admin 只读诊断接口，列出当前有效版本、跳过的坏版本、所有配置条目的校验状态。
4. 提供 admin 无写入校验接口，支持在写入前验证 payload。
5. 提供 admin active 状态切换接口，便于禁用坏版本并触发缓存清理。

## 非目标

- 不做完整配置编辑 UI。
- 不接 PromptHub。
- 不引入 JSON Schema 依赖。
- 不改变聊天、SSE、模型列表的公开协议。

## 接口

### `GET /api/admin/runtime-config`

返回：

- `effective`: 每个已知配置项当前生效来源、版本、payload、跳过版本和校验告警。
- `entries`: 数据库中所有 runtime config 条目及其校验状态。

仅管理员可访问。

### `POST /api/admin/runtime-config/validate`

请求：

```json
{
  "namespace": "prompt_template",
  "key": "generate_title",
  "payload": {"template": "标题 prompt"}
}
```

返回 `valid` 和 `issues`，不写数据库。

### `PATCH /api/admin/runtime-config/{entry_id}/status`

请求：

```json
{"is_active": false}
```

更新 `is_active`，清理 runtime config 缓存，返回该条目的最新校验状态。

## 读取策略

`get_runtime_config_payload()` 从最新 active 版本开始最多检查 10 条候选：

1. 先把 DB payload deep merge 到代码默认值。
2. 如果代码默认值本身符合该配置域 schema，则对合并结果做强校验。
3. 无效候选记录到 `meta.skipped_versions` 和 `meta.validation_warnings` 后继续看下一条。
4. 找到第一条有效 DB 候选则返回它。
5. 如果没有有效 DB 候选，返回代码默认值。

这个设计保留了读取器的通用性：测试或未来未知 namespace 使用最小默认值时，不会被已知域强校验误伤。

## 测试

- 单测覆盖坏版本跳过、全坏回默认、schema 问题输出。
- service 测试覆盖治理快照和 active 状态切换。
- API 测试覆盖管理员鉴权、快照、validate 和 status patch。
