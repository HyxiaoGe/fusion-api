# LiteLLM 运维门禁

本目录保存 LiteLLM 的可版本化升级约束，不包含密钥、数据库地址或厂商配置。
除明确写出绝对路径的备份集成示例外，下面所有 `python -m scripts...` 命令都
从 `fusion-api` 仓库根目录执行。

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
python -m scripts.check_litellm_upgrade_readiness \
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
python -m scripts.check_litellm_upgrade_readiness \
  --base-url http://127.0.0.1:4000 \
  --primary-backup /path/to/latest-postgres-dump.sql.gz \
  --secondary-backup-marker "$HOME/backups/restic-success.json"
```

也可以先保存只读快照 JSON，再从文件生成 marker：

```bash
restic snapshots --tag daily --latest 1 --json > /tmp/restic-latest.json
python -m scripts.write_restic_success_marker \
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

`litellm.candidate_preflight` 必须使用 `litellm_provider_wildcard`，每个
provider 还要声明独立的 `candidate/<provider>/*` route；实际 LiteLLM 配置
参见 `candidate-preflight-routes.example.yaml`。协调器先从普通
`/model/info` 找到 route deployment id，再用 `litellm_model_id` 读取原始
wildcard 契约，逐条核对 alias、underlying wildcard、`api_base` 和 provider
metadata。专用 key 的 allowlist 必须恰好等于全部公共候选 route，不能是
全局 `*` 或 underlying `openai/*`。candidate key 的环境变量和值都不得与
master key 相同。运维反代和 access log 还必须对 `/key/info?key=...` 查询
参数做脱敏。

每个 provider 还必须维护非空 `credential_generation`。首次可设为
`initial`，以后只要对应 API key 在同一环境变量名下轮换，就必须同步递增该
值。它会进入候选 route 和完整契约指纹，使旧凭据代际签发的验收自动失效。

LiteLLM v1.93 的原始 route 响应会省略 `litellm_params.api_key`，所以只读
协调器不能把“读取不到 key 引用”误判为配置错误。provider key 环境变量名由
受控 registry 与 route 配置共同维护；真正的凭据绑定还会由后续专用候选 key
请求在对应 `api_base` 上验证，验收摘要再通过完整候选指纹固化证据。

每份真实预准入摘要记录带时区的 `generated_at` 和 `expires_at`，默认有效期
7 天；缺失时间、过期、倒置区间或明显来自未来的证据都不能进入准入计划。

普通 `/model/info` 对 wildcard 展示的是 LiteLLM 已知目录的展开结果，可能
不包含刚由厂商发布但 LiteLLM 成本目录尚未收录的模型。因此 day-0 发现始终以
provider `/models` 为事实源；`/model/info` 只用于验证代理路由和现有注册状态。

这层隔离对共享 `openai/*` adapter 的厂商尤其重要：Qwen 与 Xiaomi 虽然都用
`openai/*`，候选请求仍分别使用 `candidate/qwen/<model_id>` 和
`candidate/xiaomi/<model_id>`，由各自 route 绑定不同 endpoint 与 provider
key。单个 provider route 异常只隔离该 provider；公共 key 异常才阻塞全部。

```bash
python -m scripts.orchestrate_litellm_model_candidates \
  --registry /path/to/provider-registry.json \
  --output /path/to/reports/model-candidates.json
```

协调器只执行 GET，并原子更新候选报告。任何厂商缺 key、请求失败或返回空列表时都会 fail-closed：不注册模型，也不会把现有模型批量判定为 removed。

报告中的 `new` 只代表候选。候选必须完成能力、费用和 Fusion 产品验收，之后才能单独生成 `/model/new` 与 virtual key allowlist 变更计划。

## 成本表快照与候选富化

LiteLLM Proxy 内部的 6 小时刷新负责运行时计费；候选管道另外保存同源的官方成本表快照，用于可审计地补齐新模型价格和能力：

```bash
python -m scripts.fetch_litellm_cost_map \
  --output /path/to/reports/litellm-cost-map.json \
  --status-output /path/to/reports/litellm-cost-map-status.json

python -m scripts.enrich_litellm_model_candidates \
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

正式 override 使用 schema v2（当前文件为 `candidate-overrides.json`）：审批记录
包含策略版本、审阅人、带时区的审阅/失效时间、HTTPS 官方来源、
`providers_sha256` 和覆盖完整文档的 `document_sha256`。区域定价必须把实际
账单币种、Fusion 的 USD 规范价、适用 API 地址和厂商区域价格对照一起放进
`pricing_provenance`。厂商跨区域公布的价格可能因小数位而产生轻微换算舍入，
校验允许区域比率最多 2% 的相对离散，但仍要求每档价格均来自官方双区证据。
厂商正式文档明确上下文窗口时，可声明正整数 `context_window_tokens`；富化器会
保留该受审字段，准入计划确定性映射为 `model_info.max_input_tokens`。未有正式
证据的模型保持未知，不推测精确值。
任一来源、时间、价格或配置变化都会使哈希失效；过期、
成本表冲突和区域地址不一致都会 fail-closed。schema v1 仅保留兼容，不应用于
新增正式 override。

成本表缺失的收费模型还必须增加精确的非 DB 候选路由并在 `model_info` 根级
声明每 token 成本，参考 `candidate-preflight-routes.example.yaml`。禁止把单模型
价格挂到厂商 wildcard；精确候选路由不会进入 Fusion 业务目录，正式准入后则由
`/model/new` 持久化相同成本字段。

## 统一只读治理周期

日常调度使用统一入口，不再手工拼接成本抓取、发现、富化和准入计划：

```bash
python -m scripts.run_litellm_governance_cycle \
  --registry /path/to/litellm-provider-registry.json \
  --output-dir /path/to/litellm-governance \
  --acceptance-dir /path/to/litellm-acceptance \
  --overrides /path/to/candidate-overrides.json
```

该命令只执行官方成本表和厂商目录 GET，并写入本地原子化产物；不会发送
completion、注册模型、修改 allowlist 或调用其他写 API。每次运行保存
`runs/<run_id>/`、manifest 和摘要：

- 成本抓取、厂商发现失败或验收文件损坏时写 `latest-failure.json`，不覆盖上一份
  `latest-success.json`；
- `candidate-queue.json` 将模型置为 `quarantined`、
  `preflight_required` 或 `admission_ready`；
- 验收摘要必须按候选完整契约指纹命名为 `<sha256>.json`，旧模型或旧元数据
  的结果不能复用；
- `removed` 只进入 `retirement-review.json`，永远不会自动删除。

`fusion-litellm-governance.service` 和 `.timer` 是 user systemd 模板，每 6 小时
按 Asia/Shanghai 运行，并带持久补跑和随机抖动。安装前先建立目录并替换模板
中的仓库路径。两个 service 固定使用独立 Python 3.11+ venv，不能依赖宿主
`/usr/bin/python3`：

```bash
python3.11 -m venv "$HOME/.local/share/fusion/litellm-governance-venv"
"$HOME/.local/share/fusion/litellm-governance-venv/bin/python" \
  -m pip install -r ops/litellm/requirements-governance.txt
"$HOME/.local/share/fusion/litellm-governance-venv/bin/python" -c \
  'import sys, httpx; assert sys.version_info >= (3, 11); print(sys.version, httpx.__version__)'

mkdir -p \
  "$HOME/.config/fusion" \
  "$HOME/.config/systemd/user" \
  "$HOME/.local/share/fusion/litellm-acceptance" \
  "$HOME/backups/litellm-governance"

install -m 0600 ops/litellm/litellm-governance.env.example \
  "$HOME/.config/fusion/litellm-governance.env"
# 直接编辑 0600 目标文件，只填写候选专用 key；不要复制 master/provider keys。
install -m 0600 /path/to/litellm-provider-registry.json \
  "$HOME/.config/fusion/litellm-provider-registry.json"
install -m 0644 ops/litellm/fusion-litellm-governance.service \
  "$HOME/.config/systemd/user/fusion-litellm-governance.service"
install -m 0644 ops/litellm/fusion-litellm-governance.timer \
  "$HOME/.config/systemd/user/fusion-litellm-governance.timer"
install -m 0644 ops/litellm/fusion-litellm-cost-sync.service \
  "$HOME/.config/systemd/user/fusion-litellm-cost-sync.service"
install -m 0644 ops/litellm/fusion-litellm-cost-sync.timer \
  "$HOME/.config/systemd/user/fusion-litellm-cost-sync.timer"
```

两个 service 不使用 systemd `EnvironmentFile=`。它会在 `ExecStartPre`
之前把文件中的任意变量注入进程，无法作为权限前置门禁。单元改由
`run_litellm_governance_unit.py` 先以 `O_NOFOLLOW` 打开并用 `fstat` 检查现有
`$HOME/project/litellm-proxy/.env`，通过后才按 registry 白名单读取 master
和 provider keys，避免复制第二份凭据后发生轮换漂移。独立
`litellm-governance.env` 只保存 `LITELLM_BASE_URL`、
`LITELLM_CANDIDATE_KEY` 等治理专用变量。启用前两份文件都必须归当前用户所有、
是普通文件且权限不宽于 `0600`：

```bash
chmod 0600 \
  "$HOME/project/litellm-proxy/.env" \
  "$HOME/.config/fusion/litellm-governance.env"

"$HOME/.local/share/fusion/litellm-governance-venv/bin/python" \
  -m scripts.check_litellm_governance_runtime \
  --secret-file "$HOME/project/litellm-proxy/.env" \
  --secret-file "$HOME/.config/fusion/litellm-governance.env"
```

wrapper 不输出文件内容；缺失文件、符号链接、owner 不匹配以及任何
group/other 权限都会在解析前失败。通过后也只向目标任务注入
`LITELLM_MASTER_KEY`、registry 声明且以 `_API_KEY` 结尾的 provider key，
以及三个治理变量；`PYTHONPATH`、`PYTHONHOME`、`LD_PRELOAD` 和其他任意
`.env` 变量不会进入目标环境。治理 env 若重复保存 master/provider key 同样
失败。registry 也用同样的 no-follow/owner/权限门禁读取，并被复制到只读 sealed
memfd；wrapper 和治理子进程消费同一份不可修改快照，避免密钥白名单与 endpoint
配置在两次读取间漂移。dev 当前 LiteLLM `.env` 为 `0664`，因此发布时必须先在明确授权下收紧
权限，不能直接启用 timer。systemd 单元对脚本、配置、目录和运行时使用
`Assert*` 而不是会静默跳过的 `Condition*`；缺失依赖必须进入 failed 状态并
保留 journal 证据。

启用 timer 属于运维变更，必须在目标环境发布授权后执行。运行时成本表的
schedule 仍是 LiteLLM 进程内存状态，所以由独立
`fusion-litellm-cost-sync.timer` 每 15 分钟幂等检查：

- 已按 6 小时健康调度时只 GET，不产生写入；
- 未调度或周期错误时才调用一次 schedule API，再 GET 复核；
- 新建调度后若 LiteLLM 尚未给出 `last_run` / `next_run`，只执行一次
  `reload/model_cost_map` 完成冷启动，再次 GET 复核；
- stale、fallback 或异常不会通过反复重排掩盖，service 非零退出并保留
  journal 证据。

成本守护使用固定 Asia/Shanghai `OnCalendar`，不依赖上一次 service 成功激活
时间；即使前一次因权限或配置 assert 失败，后续 15 分钟触发仍会保留。

可以先用默认 dry-run 手工查看计划：

```bash
python -m scripts.ensure_litellm_cost_map_sync
```

统一周期保存的是官方成本表审计快照，不能替代 Proxy 内部刷新；成本同步
守护也不能替代候选发现和 Fusion 准入。

## 候选预准入与注册计划

从 `candidate-queue.json` 提取单个 `preflight_required` 候选后，先查看零请求
计划；只有明确接受一次真实收费验收时才使用 `--apply`：

```bash
python -m scripts.check_litellm_candidate_preflight candidate.json --dry-run

LITELLM_CANDIDATE_KEY=... \
python -m scripts.check_litellm_candidate_preflight candidate.json --apply \
  > /path/to/litellm-acceptance/<candidate-fingerprint>.json
```

下一次统一治理周期会读取匹配指纹的验收摘要，并为通过的单候选生成带
`run_id` 的 admission plan。能力矩阵由候选声明驱动，除文本、SSE、
tool calling、usage 和 cost 外，还会按需验证视觉语义、reasoning 字段以及
保留完整 assistant tool message 的多轮工具调用。

计划器本身没有 apply 模式，也不发 HTTP。它仅在以下条件全部成立时生成
`/model/new` dry-run payload：

- 候选发现状态正常；
- 价格、能力和来源证据完整；
- endpoint 已由发现请求验证；
- 真实预准入文本、SSE、可选工具、usage/cost 全部通过；
- 验收摘要中的完整候选契约 SHA-256 与当前 provider、underlying model、endpoint、价格、能力和元数据证据完全一致；
- 没有 reasoning 标签泄漏等高风险质量问题。

生成的 v1.93+ payload 使用 LiteLLM 官方格式
`api_key: os.environ/变量名`，不会把真实厂商密钥写入报告。可以用完全隔离、
零厂商请求的脚本复核 DB model 跨重启解析契约：

```bash
bash scripts/validate_litellm_db_env_reference.sh
```

## 受控准入事务

`execute_litellm_candidate_admission.py` 的 dry-run 可以读取人工提取的单个
admission plan，且不发任何 HTTP：

```bash
python -m scripts.execute_litellm_candidate_admission \
  --plan /path/to/single-admission-plan.json \
  --output /path/to/admission-transaction.json
```

真实 apply 属于发布门禁，禁止直接消费任意 `--plan`。执行器必须从
`--governance-root/latest-success.json` 出发，验证 run path、manifest SHA、
candidate queue 与 admission plans artifact SHA，重算 queue candidate
fingerprint，并从候选原始契约重建 `/model/new` payload；随后还要同时确认
`run_id`、模型 ID 和候选 fingerprint。

执行器从环境读取 LiteLLM master key、Fusion virtual key 和厂商 key，在写前
读取 `/model/info` 与 `/key/info` 建立 CAS before-state，
依次执行注册、读回、allowlist、Fusion `/api/models/` 读回和目录审计；中途
失败会通过 CAS 尽力恢复原 allowlist，并只在能证明本事务 UUID 所有权时删除
本次新建的模型。

```bash
python -m scripts.execute_litellm_candidate_admission \
  --governance-root /path/to/litellm-governance \
  --candidate-fingerprint '<sha256>' \
  --output /path/to/admission-transaction.json \
  --expected-run-id '<run_id>' \
  --confirm-model-id '<model_id>' \
  --confirm-fingerprint '<sha256>' \
  --apply
```

只有 `/model/new` 明确返回本事务 UUID，且按 UUID、alias、underlying model、
`api_base`、治理 fingerprint 与完整 metadata 精确读回后，才会打开
allowlist。注册响应结果不确定时不会按名称猜测并删除模型，而是标记
`manual_cleanup_required`，由发布人员基于事务证据人工处置。

脚本成功只证明注册事务与 API 读回通过，不能替代真实登录态下的模型选择、
文本/流式/工具/刷新恢复等产品层验收。最终目录审计默认只读，并在存在 error
时非零退出：

```bash
python -m scripts.audit_litellm_model_catalog --dry-run
```

provider 发现失败会写入 `pipeline_issues` 并让准入 CLI 非零退出；共享 `openai/` adapter 只接受 provider namespace 一致的成本条目；若候选 `model_id` 已被其他 underlying model 用作业务 alias，发现阶段直接转入 `unknown` 隔离。

成本表状态检查器验证的是 scheduler cadence；LiteLLM 当前 status 响应没有独立的“GitHub 价格表拉取成功”字段。若响应包含 `source=bundled` 或 `fallback_reason`，检查器会失败；否则仍需在升级/抽检阶段做价格样本对账。
