# LiteLLM 可持续治理设计

## 目标

把 LiteLLM 从“人工换镜像、人工补模型”改造成一条有备份门禁、隔离验证、成本元数据同步和候选模型准入的可持续链路，同时保证新模型不会在尚未验收时直接进入 Fusion 选择器。

## 当前事实

- dev 当前运行 `litellm/litellm:v1.88.0`，服务和数据库 readiness 正常。
- LiteLLM 的模型价格表没有定时刷新。
- LiteLLM 官方模型发现只覆盖部分 provider，Moonshot 不在支持列表内。
- Fusion 的 `/api/models` 直接消费 LiteLLM `/model/info` 中的 DB model；因此候选模型不能先写入 DB 再验收。
- 已有目录审计、allowlist 对账和全模型验收脚本可以复用。
- LiteLLM 数据库本机备份存在，但二级 restic 备份当前失败；升级前必须先恢复备份门禁。

## 非目标

- 不自动追踪 `latest`、RC 或 dev 镜像。
- 不让 Watchtower 无门禁更新 LiteLLM。
- 不把价格表刷新误当成模型注册或 Fusion 发布。
- 不在候选阶段调用 `/model/new`、修改 virtual key allowlist 或暴露到前端。
- 不在本阶段自动轮换密钥或修改敏感文件权限。

## 总体流程

```text
版本通知
  -> 备份与恢复门禁
  -> 固定版本和 digest
  -> 隔离数据库迁移
  -> Fusion 核心链路验收
  -> 受控滚动

厂商 /v1/models 或 provider adapter
  -> 只读候选快照
  -> 与 LiteLLM /model/info 对账
  -> 候选隔离报告
  -> 官方成本表与受审 override 只读富化
  -> LiteLLM wildcard 预准入能力与费用验收
  -> 人工或策略批准
  -> LiteLLM DB model
  -> Fusion virtual key allowlist
  -> Fusion 选择器
  -> Fusion 全模型产品验收
```

## 版本升级策略

- 目标版本固定为经过验证的稳定版；第一目标为 `v1.93.0`。
- 镜像同时固定 tag 和 digest，更新由显式变更触发。
- 升级前必须满足：
  - LiteLLM PostgreSQL 备份成功且压缩包校验通过；
  - 二级备份成功；
  - 数据库副本恢复成功；
  - 当前模型、virtual key、allowlist 和配置脱敏快照已导出；
  - readiness healthcheck 可用于判断容器是否就绪。
- 隔离验证至少覆盖数据库迁移、`/model/info`、virtual key 鉴权、SSE、reasoning、tool calling、vision 和 usage/cost。
- 若 LiteLLM 自身启用了 MCP OAuth，升级到 `v1.93.0` 前额外核对 `oauth2_flow` 兼容性。

## 成本表同步

- 在目标版本验证通过后，通过官方管理端点配置每 6 小时刷新。
- 监控调度状态、最近执行时间、下次执行时间，以及端点明确返回的 source/fallback 原因。
- 当前官方 status 端点没有独立的“远端数据拉取成功”字段；调度健康不能替代价格抽样对账，禁止把 `last_run` 表述成已确认成功。
- 刷新内容只用于价格、上下文窗口和能力元数据。
- 候选管道保存同源官方成本表的 SHA-256、ETag、抓取时间和模型数，避免把“调度存在”误当成内容正确。
- 上游没有价格的模型必须保持 `unknown` 或显式自定义价格，禁止猜测价格。

## 候选模型隔离

候选发现是纯读过程，输入包括：

- provider adapter 返回的模型 ID；
- LiteLLM `/model/info` 当前注册模型；
- 可选的上一次候选快照。

输出按以下状态分类：

- `new`：厂商存在、LiteLLM 尚未注册；
- `existing`：厂商与 LiteLLM 均存在；
- `removed`：LiteLLM 已注册、厂商列表不再返回；
- `unknown`：输入无法可靠归一化或 provider 不支持发现。

候选报告不得包含 API key，不得产生 LiteLLM 写操作。后续准入器只能消费已通过能力、费用和产品验收的候选。

候选富化优先使用官方 LiteLLM 成本表。共享 `openai/` adapter 通过 provider 的 `cost_map_prefix` 映射到官方成本条目；官方表缺失时只能使用有审查记录的 metadata override。缺少成本、能力或来源证据的候选继续隔离。

预准入摘要必须绑定完整候选契约哈希，覆盖 provider、候选 route、实际请求 model、underlying model、endpoint、环境变量名、价格、能力和元数据来源；任一字段变化都必须重新验收。跨 provider 业务 alias 冲突、provider 发现失败以及成本 namespace 不一致均为 fail-closed。

## 两阶段验收

未注册候选不会出现在 Fusion `/api/models`，因此不能用现有 Fusion 全模型脚本完成首次准入：

1. 预准入验收通过逐 provider 的 LiteLLM wildcard route 调用候选；route 必须绑定该 provider 的 endpoint、API key 引用和 metadata。共享 `openai/*` adapter 的厂商不得共用全局 `*` route。至少验证非流式文本、SSE、可选 tool calling、usage/cost 和声明能力。该阶段不得创建 DB model。
2. 预准入通过后只生成 dry-run 注册与 allowlist 计划；v1.93+ 的 DB model 密钥引用使用官方 `os.environ/变量名` 格式，实际写入必须经过发布门禁。
3. 注册后再运行 `MODEL_ACCEPTANCE_RUNBOOK.md` 的 Fusion SSE、Agent、视觉和真实 UI 验收。失败时回滚 allowlist/模型注册，不把预准入结果冒充产品验收。

## 失败与降级

- 厂商发现失败：保留上一份成功快照，记录 stale，不下线现有模型。
- 成本表同步失败：保留当前成本表并告警，不阻断聊天。
- 隔离迁移失败：停止升级，保留当前容器与数据库。
- 新模型验收失败：候选保持隔离，不进入 LiteLLM DB model 和 Fusion allowlist。
- provider 返回空列表：视为发现异常，不把全部现有模型标记为 removed。

## 验收标准

- 默认执行任何新脚本都只读，必须显式参数才允许写操作。
- 候选报告单元测试覆盖 new、existing、removed、unknown、空响应和重复模型。
- 预准入验收默认 dry-run，只有显式 `--apply` 才允许产生收费请求，且永远没有模型注册或 allowlist 写能力。
- 准入计划必须同时要求预准入通过、能力与价格 metadata 完整，并保持 dry-run。
- LiteLLM Compose 具备 readiness healthcheck。
- 成本表调度状态可被机器读取并产生失败退出码。
- 升级演练可从数据库副本启动 `v1.93.0`，现有 Fusion 模型目录和 virtual key 不发生意外变化。
- 真实模型验收复用 `docs/MODEL_ACCEPTANCE_RUNBOOK.md`，不得用 mock 代替最终集成证据。
