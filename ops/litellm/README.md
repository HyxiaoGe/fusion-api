# LiteLLM 运维门禁

本目录保存 LiteLLM 的可版本化升级约束，不包含密钥、数据库地址或厂商配置。

## Compose 合并检查

在部署目录执行：

```bash
docker compose \
  -f docker-compose.yml \
  -f /path/to/fusion-api/ops/litellm/docker-compose.governance.yml \
  config --quiet
```

治理覆盖文件固定经过隔离验证的版本和 digest，增加 readiness healthcheck，并关闭 Watchtower 自动滚动。

## 升级前检查

```bash
python scripts/check_litellm_upgrade_readiness.py \
  --base-url http://127.0.0.1:4000 \
  --primary-backup /path/to/latest-postgres-dump.sql.gz \
  --secondary-backup-marker /path/to/latest-restic-success.marker
```

脚本只执行 GET 和本地文件检查。退出码为 `0` 才表示基础门禁通过；成本表未调度只产生 warning，因为首次升级前允许尚未启用调度。

`--secondary-backup-marker` 必须是 JSON，至少包含 `{"status":"success","snapshot_id":"..."}`；任意空文件或手工 `touch` 的普通文件不能通过。主备份为 `.gz` 时会完整读取并校验 gzip。

## 必要隔离证据

- 当日 PostgreSQL 备份完整恢复；
- `v1.93.0` 在数据库副本上完成 migration；
- readiness 返回 DB connected；
- DB model 别名与升级前一致；
- virtual key 数量和 allowlist 不发生意外变化；
- 成本表手动刷新和 6 小时调度成功；
- Fusion 全模型验收按 `docs/MODEL_ACCEPTANCE_RUNBOOK.md` 执行。

缺少任一证据时，不得把覆盖文件应用到运行中的 dev LiteLLM。

## 新模型候选发现

复制 `provider-registry.example.json` 到运维主机的受控配置目录，并按实际支持情况启用 provider。Registry 只记录地址和环境变量名，不保存 API key。

```bash
python scripts/orchestrate_litellm_model_candidates.py \
  --registry /path/to/provider-registry.json \
  --output /path/to/reports/model-candidates.json
```

协调器只执行 GET，并原子更新候选报告。任何厂商缺 key、请求失败或返回空列表时都会 fail-closed：不注册模型，也不会把现有模型批量判定为 removed。

报告中的 `new` 只代表候选。候选必须完成能力、费用和 Fusion 产品验收，之后才能单独生成 `/model/new` 与 virtual key allowlist 变更计划。

成本表状态检查器验证的是 scheduler cadence；LiteLLM 当前 status 响应没有独立的“GitHub 价格表拉取成功”字段。若响应包含 `source=bundled` 或 `fallback_reason`，检查器会失败；否则仍需在升级/抽检阶段做价格样本对账。
