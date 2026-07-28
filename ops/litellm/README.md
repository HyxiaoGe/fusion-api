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

`--secondary-backup-marker` 必须是 JSON，至少包含 `status=success`、完整 `snapshot_id`、带时区的 `completed_at` 和 `tag=daily`；门禁按快照完成时间而不是 marker 文件 mtime 判断新鲜度，因此复制或 `touch` 旧 marker 不能通过。主备份为 `.gz` 时会完整读取并校验 gzip。

### 生成 restic 成功 marker

`write_restic_success_marker.py` 只消费 `restic snapshots --latest 1 --json` 的结果，不执行备份、`forget` 或 `prune`，也不读取 restic 仓库密码。只有最新快照包含完整 `id`、时间带时区且未超过最大年龄时，才会原子替换 marker；校验或写入失败会保留旧 marker。

应当只在 `restic backup` 成功后调用 helper，例如在现有备份脚本中接入：

```bash
RESTIC_SUCCESS_MARKER="$HOME/backups/restic-success.json"

if "$RESTIC" backup "${PATHS[@]}" --tag daily; then
    if "$RESTIC" snapshots --tag daily --latest 1 --json \
        | python /path/to/fusion-api/scripts/write_restic_success_marker.py \
            --snapshots-json - \
            --output "$RESTIC_SUCCESS_MARKER" \
            --max-age-seconds 129600
    then
        log "restic 成功 marker 已更新"
    else
        log "WARN restic 快照 marker 生成失败"
        FAIL=1
    fi
else
    FAIL=1
fi
```

不要在 `finally`、失败分支或独立定时任务中无条件生成 marker。升级门禁使用同一路径：

```bash
python scripts/check_litellm_upgrade_readiness.py \
  --base-url http://127.0.0.1:4000 \
  --primary-backup /path/to/latest-postgres-dump.sql.gz \
  --secondary-backup-marker "$HOME/backups/restic-success.json"
```

也可以先保存只读快照 JSON，再从文件生成 marker：

```bash
restic snapshots --tag daily --latest 1 --json > /tmp/restic-latest.json
python scripts/write_restic_success_marker.py \
  --snapshots-json /tmp/restic-latest.json \
  --output "$HOME/backups/restic-success.json"
```

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

## 成本表快照与候选富化

LiteLLM Proxy 内部的 6 小时刷新负责运行时计费；候选管道另外保存同源的官方成本表快照，用于可审计地补齐新模型价格和能力：

```bash
python scripts/fetch_litellm_cost_map.py \
  --output /path/to/reports/litellm-cost-map.json \
  --status-output /path/to/reports/litellm-cost-map-status.json

python scripts/enrich_litellm_model_candidates.py \
  --candidate-report /path/to/reports/model-candidates.json \
  --registry /path/to/provider-registry.json \
  --cost-map /path/to/reports/litellm-cost-map.json \
  --cost-map-status /path/to/reports/litellm-cost-map-status.json \
  --output /path/to/reports/model-candidates-enriched.json
```

成本表状态包含抓取时间、模型数、SHA-256、ETag 和 Last-Modified。富化器会校验状态文件与数据文件的哈希、数量和新鲜度，只读本地文件：

- 成本表已收录且价格、能力证据完整：自动补齐 Fusion metadata；
- 成本表未收录或字段不完整：保留缺失字段，后续准入门禁 fail-closed；
- 厂商使用共享 `openai/` 前缀时，可在 registry 配置 `cost_map_prefix`；
- 官方表暂未收录但已有厂商正式证据时，可通过受审 `--overrides` 文件补齐；文件只允许 metadata，不保存密钥。

## 候选预准入与注册计划

从富化报告提取单个 `new` 候选后，先查看零请求计划；只有明确接受一次真实收费验收时才使用 `--apply`：

```bash
python scripts/check_litellm_candidate_preflight.py candidate.json --dry-run

LITELLM_CANDIDATE_KEY=... \
python scripts/check_litellm_candidate_preflight.py candidate.json --apply \
  > /path/to/reports/candidate-acceptance.json

python scripts/plan_litellm_candidate_admission.py \
  --candidate-report /path/to/reports/model-candidates-enriched.json \
  --candidate-acceptance-summary /path/to/reports/candidate-acceptance.json \
  --output /path/to/reports/candidate-admission-plan.json
```

准入器本身没有 apply 模式，也不发 HTTP。它仅在以下条件全部成立时生成 `/model/new` dry-run payload：

- 候选发现状态正常；
- 价格、能力和来源证据完整；
- endpoint 已由发现请求验证；
- 真实预准入文本、SSE、可选工具、usage/cost 全部通过；
- 验收摘要中的完整候选契约 SHA-256 与当前 provider、underlying model、endpoint、价格、能力和元数据证据完全一致；
- 没有 reasoning 标签泄漏等高风险质量问题。

生成的 v1.93+ payload 使用 LiteLLM 官方格式 `api_key: os.environ/变量名`，不会把真实厂商密钥写入报告。实际注册、allowlist 修改和注册后的 Fusion 产品验收仍属于发布门禁。

provider 发现失败会写入 `pipeline_issues` 并让准入 CLI 非零退出；共享 `openai/` adapter 只接受 provider namespace 一致的成本条目；若候选 `model_id` 已被其他 underlying model 用作业务 alias，发现阶段直接转入 `unknown` 隔离。

成本表状态检查器验证的是 scheduler cadence；LiteLLM 当前 status 响应没有独立的“GitHub 价格表拉取成功”字段。若响应包含 `source=bundled` 或 `fallback_reason`，检查器会失败；否则仍需在升级/抽检阶段做价格样本对账。
