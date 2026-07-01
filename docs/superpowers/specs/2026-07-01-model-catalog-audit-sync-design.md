# 模型目录巡检/同步 v1 设计

## 背景

Fusion 前端模型选择器只展示 LiteLLM `/model/info` 中 `db_model=true` 的显式业务别名。上一次小米模型修复证明了三个目录容易漂移：

- LiteLLM 全量模型目录：包含 wildcard 路由和显式业务别名。
- Fusion `/api/models/`：只展示可给用户选择的 `db_model=true` 别名。
- Fusion LiteLLM virtual key allowlist：决定 fusion-api 实际可调用哪些模型。

如果三者不一致，多模型测验会把“配置问题”和“模型能力问题”混在一起。

## 目标

新增一个可重复运行的巡检/同步 v1 脚本，先把目录治理变成可观测、可 dry-run、可显式 apply 的机制。

v1 不做无人值守自动改生产，不引入数据库表，不改 `/api/models/` 产品语义。

## 范围

v1 覆盖：

- 对比 LiteLLM `db_model=true` 业务模型与 Fusion `/api/models/` 展示模型。
- 对比业务模型与 virtual key allowlist，发现 key 缺失和 key 多余模型。
- 识别 LiteLLM 中已知退役模型。
- 识别缺关键 metadata 的业务模型：`provider_key`、`provider_display`、`capabilities`、`pricing`。
- 识别重复的 `db_model=true` 别名，报告时按唯一别名处理。
- 输出 JSON 巡检报告，字段稳定，方便后续纳入 CI、Cron 或监控。
- 可选 `--apply` 仅同步 Fusion virtual key allowlist；模型注册/删除仍由现有治理脚本执行。

v1 不覆盖：

- 自动访问每个 provider 官方 `/models` 并推断新增模型。
- 自动注册未知新模型。
- 自动删除 LiteLLM 模型。
- 模型能力质量测评。

## 数据模型

巡检报告使用 JSON：

```json
{
  "summary": {
    "litellm_db_models": 14,
    "fusion_models": 14,
    "virtual_key_models": 14,
    "issue_count": 0
  },
  "issues": [
    {
      "code": "key_missing_db_model",
      "severity": "error",
      "model_name": "mimo-v2.5-pro",
      "message": "Fusion virtual key 缺少业务模型 mimo-v2.5-pro"
    }
  ],
  "sync_plan": {
    "allowlist_before": [],
    "allowlist_after": [],
    "add": [],
    "remove": []
  }
}
```

严重级别：

- `error`：会导致用户可见模型不可调用，或调用到目录外模型。
- `warning`：不一定阻塞调用，但会影响展示、分组、成本或能力判断。

## 同步策略

v1 只同步 virtual key allowlist：

- 目标集合：LiteLLM 中 `db_model=true` 且关键 metadata 完整的业务模型。
- 保留集合：allowlist 中非 Fusion 业务模型但也不属于已知退役模型的条目，避免误删别的服务共享 key 的模型。
- 删除集合：allowlist 中的已知退役模型。
- 新增集合：metadata 完整的业务模型中 allowlist 缺失的模型。

metadata 不完整的 `db_model=true` 别名只报告 warning，不自动加入 Fusion key。真实环境中可能存在 `chat-default`、`chat-premium`、`audio-structuring` 这类工具或其他服务别名，v1 不能把它们误判成用户可选模型。

`--dry-run` 默认只输出报告和计划；`--apply` 才调用 LiteLLM `/key/update`。

## 测试矩阵

| Case | 输入 | 预期 |
| --- | --- | --- |
| AUDIT-01 | LiteLLM db_model 与 Fusion models 一致 | 无 issue |
| AUDIT-02 | Fusion 展示了 LiteLLM 没有的模型 | `fusion_unknown_model` error |
| AUDIT-03 | virtual key 缺少 db_model | `key_missing_db_model` error，sync_plan add |
| AUDIT-04 | virtual key 含已知退役模型 | `key_deprecated_model` error，sync_plan remove |
| AUDIT-05 | db_model 缺 provider/pricing/capabilities metadata | `metadata_missing` warning |
| AUDIT-06 | dry-run 序列化 | 不输出 master key / virtual key 等秘密 |
| AUDIT-07 | apply | 只调用 `/key/update`，不注册或删除模型 |
| AUDIT-08 | LiteLLM 返回重复 db_model 别名 | `db_model_duplicate` warning，summary 按唯一别名计数 |
| AUDIT-09 | metadata 不完整且 Fusion 未展示的 db_model | 只 warning，不加入 sync_plan add |

## 验收

- 新增单测覆盖上述核心规则。
- 脚本 dry-run 能在生产 LiteLLM 上输出报告。
- 本地测试、ruff、架构检查通过。
- 提交并 push 后 GitHub Actions 和 dev 部署成功。
