# Fusion Monorepo 合并实施计划

> Spec: 本文件自带依据章节，无独立 spec
> Base: `fusion-api@c5b0775` / `fusion-ui@master`
> Branch: `feat/monorepo-merge`
> Target: 新建仓库 `HyxiaoGe/fusion`

**目标：** 把 `fusion-api` 与 `fusion-ui` 合并为单一仓库 `fusion`，保留两边完整 git 历史，并把当前分散在两处、共约 3600 行的发布流水线收敛为一份参数化的可复用 workflow + 一组可本地执行的部署脚本。

**本轮范围：** 仓库结构、git 历史、CI/CD、部署平台绑定、agent 与文档配置。不改任何业务代码，不改 Redis Stream 两段式流，不改 API 契约，不借机重构 `app/` 或 `src/` 下的任何模块。

**总体架构：** 新仓库采用 `apps/api` + `apps/ui` 两个应用目录，根目录只保留跨应用的编排物（workflow、部署脚本、台账、导航文档）。发布流水线拆成三层：`deploy.yml` 做 paths 过滤与分发，`_deploy-app.yml` 是参数化的 `workflow_call` 可复用 workflow，`ops/deploy/*.sh` 承载实际 shell 逻辑并可脱离 GitHub Actions 本地执行。

## 依据

本计划的每条判断都来自对两个仓库当前状态的实测，不依赖印象：

- **代码零耦合。** `fusion-api` 505 个 `.py`，`fusion-ui` 269 个 `.tsx` + 253 个 `.ts`，交集为 0。无共享包、无 workspace 依赖、无交叉 import。目录下沉后各自的相对路径、构建、测试均不变。
- **历史合并成本低。** 两边 `.git` 分别为 5.5M / 2.8M，常规 `merge --allow-unrelated-histories` 即可完整保留，无需 `filter-repo`。
- **发布流水线已被手工写了两遍。** 两边步骤名逐字相同的有 14 个（去重后）（`Checkout smoke scripts`、`Configure Docker credential directory`、`Cleanup old images`、`Push CI/CD metrics`、`通知飞书(部署结果)` 等），另有 3 对同职能不同名（`Capture current deployment` ↔ `…for rollback`、`Rollback previous deployment` ↔ `Roll back failed deployment`、`Smoke check candidate` ↔ `Verify health`）。抽象成可复用 workflow 有实证支撑，不是投机。
- **`deploy.yml` 的体量是内联 shell，不是 YAML 编排。** 全文 2553 行中 `deploy-dev` 单个 job 占 2339 行，其中四个步骤合计 1757 行（占 75%）：`Pull and restart fusion-api` 693、`Roll back failed deployment` 476、`Capture current deployment for rollback` 307、`Verify health` 281。全文 28 个 `run: |` 块、436 行 shell 控制流关键字。这套逻辑当前不可本地执行、不可单测，唯一验证方式是向 master 推送。
- **同职能步骤两边体量差 10 倍。** Capture 307 vs 30 行，Rollback 476 vs 65 行。差额中一部分是 API 的真实约束（alembic 迁移、flyai-adapter 必须与 api 同 SHA、litellm governance worker 与 model management worker 的暂停/恢复），一部分是重复的防御性检查。
- **双仓成本已有记录。** 50 份 plan 中 7 份横跨两仓；`docs/superpowers/plans/2026-06-28-agent-run-continuation.md` 明确写有"这份 spec 横跨 fusion-api 和 fusion-ui……实施时使用一个工作分支；最终按子仓分别提交，避免把半成品多次推送触发流水线"。
- **运行时已经耦合。** `fusion-ui/next.config.js` 通过 rewrites 把 `/api/*` 同源代理到 fusion-api，SSE 走此通道；`fusion-ui/docker-compose.yml` 直接引用 `http://fusion-api:8000`；两边共用同一组 self-hosted runner（Windows 构建 / Linux 部署）、同一个阿里云 ACR、同一台 dev server。

## Global Constraints

- **搬迁与重写必须分阶段，不得在同一次变更内完成。** 流水线操作真实 docker 重启与 alembic 迁移，二者混做会使发布故障无法二分定位。
- Task 1 完成前不得改动任何 workflow 逻辑，只允许改路径与新增 `paths:` 过滤。
- 每个 Task 结束后两个应用各自完成一次真实发布 + 一次真实回滚，未通过不得进入下一 Task。
- `paths:` 过滤是强制项。合并后无过滤会导致修改单个文件同时触发两条重型发布流水线（重建镜像、执行 Alembic、重启生产容器）。
- 抽取 shell 到 `ops/deploy/` 时**逐字搬运**，不得顺手"优化"。行为差异必须为零，抽取与改写分属不同 Task。
- 镜像 tag 由 `<sha>` 改为 `<app>-<sha>`，每个应用独立记录 last-deployed-sha。合并后单个 commit SHA 同时对应多个应用镜像，原"回滚到 SHA X"语义失效。
- 保留 `deploy-dev` 现有的 `cancel-in-progress: false`，并新增跨应用互斥：两条流水线操作同一台 dev server 的 docker，不得并发。
- API 侧的额外约束（alembic 迁移、flyai-adapter 同 SHA 校验、两个 worker 的暂停/恢复）以可选 hook 形式进入可复用 workflow，不得强加给 UI。
- 不删除任何现有校验分支。合并后无法确认必要性的检查一律保留，并在 plan 中登记待评估，而非就地删除。
- 两个原仓库在新仓库通过全部验证前保持可用，仅在 Task 5 归档。
- 提交信息中文，格式 `<type>: <中文描述>`，包含 `Co-Authored-By`。

## 目标结构

```
fusion/
├── apps/
│   ├── api/                    # 原 fusion-api 全部内容
│   └── ui/                     # 原 fusion-ui 全部内容
├── ops/
│   └── deploy/                 # 从 YAML 抽出的部署脚本，可本地执行
│       ├── capture-rollback-target.sh
│       ├── restart-service.sh
│       ├── verify-health.sh
│       ├── rollback.sh
│       └── notify-feishu.sh
├── docs/
│   ├── EXECUTION_LEDGER.md     # 合并两仓台账
│   └── superpowers/            # 合并两仓 plans / specs
├── .agents/skills/             # api 侧 9 个 + ui 侧 1 个
├── .github/workflows/
│   ├── deploy.yml              # 薄编排：paths 过滤 → 分发
│   ├── _deploy-app.yml         # workflow_call 可复用 workflow
│   └── pr-ci.yml               # paths 分流
└── CLAUDE.md                   # 根导航，apps/*/CLAUDE.md 为应用级
```

## Task 1：建仓、历史合并与目录下沉

1. 新建 `HyxiaoGe/fusion`，默认分支 `master`。
2. 分别以 `git remote add` + `merge --allow-unrelated-histories` 引入两仓历史，各自先在原仓库内 `git mv` 到 `apps/api` / `apps/ui` 再合并，避免根目录文件互相覆盖。
3. 12 个同名根文件按归属下沉：`CLAUDE.md`、`AGENTS.md`、`README.md`、`.gitignore`、`.dockerignore`、`.env.example`、`Dockerfile`、`docker-compose.yml`、`railway.json`、`.agents/`、`docs/`、`scripts/`。
4. workflow 原样复制两份，仅改动 checkout 后的工作目录、`.github/scripts/*` 路径、Dockerfile build context（`.` → `apps/api` / `apps/ui`），并为每条加 `paths:` 过滤。
5. `release-safety.yml` 拆为 `apps/api/release-safety.yml` 与 `apps/ui/release-safety.yml`，`release-safety-contract.sh` 改为接受契约文件路径参数。

**验证：** 两个应用各自 PR CI 通过；各自向 master 推送一次真实发布并成功；各自执行一次 `workflow_dispatch` 回滚并成功。修改 `apps/api` 下文件不得触发 UI 流水线，反之亦然。

## Task 2：抽取 shell 到 ops/deploy

按体量倒序逐个抽取，每抽一个独立验证一次：`Pull and restart`(693) → `Roll back failed deployment`(476) → `Capture current deployment`(307) → `Verify health`(281) → 其余。

1. 每个步骤的 `run: |` 块整体移入 `ops/deploy/<name>.sh`，入参改为显式环境变量或位置参数，YAML 侧只保留调用。
2. 为每个脚本补 bats 或 shunit2 用例，覆盖至少：正常路径、目标 SHA 非法、回滚锚点缺失三种情况。
3. PR CI 增加脚本 dry-run 与单测门禁。

**验证：** 每抽取一个脚本，两个应用各发布一次并回滚一次，行为与抽取前完全一致。全部抽完后 `deploy.yml` 的 YAML 行数应降至 400 行以内。

## Task 3：参数化合并与可复用 workflow

前置：Task 2 完成，两边 shell 已成为可读的独立文件，重复之处可见。

1. 以 UI 侧精简实现为基线，逐条比对 API 侧多出的检查，分类为「真实约束」与「重复防御」。真实约束做成可选 hook，重复防御合并。分类结果登记在本文件，不在本 Task 内删除任何检查。
2. 抽出 `_deploy-app.yml`，参数：应用名、镜像仓库、健康检查端点、迁移开关、依赖服务列表、回滚锚点校验策略。
3. `deploy.yml` 收敛为 paths 过滤 + 两次 `workflow_call`。
4. 镜像 tag 改为 `<app>-<sha>`，回滚改为按应用独立追踪 last-deployed-sha。
5. 新增跨应用 concurrency 互斥。

**验证：** 两个应用各发布、各回滚一次；构造一次跨应用并发推送，确认互斥生效；构造一次仅改 UI 的提交，确认 API 容器 SHA 不变且回滚目标仍然唯一可解析。

## Task 4：部署平台与外部绑定

1. Vercel root directory 改为 `apps/ui`；Railway 两个服务的 root directory 分别指向 `apps/api` / `apps/ui`。
2. ACR 仓库名保持不变，仅 tag 规则变更，确认存量镜像不受影响。
3. GitHub secrets / variables 迁移到新仓库；self-hosted runner 重新注册到新仓库。
4. 分支保护、required checks 按新 workflow 的 job 名重建。

**验证：** 逐项在平台控制台确认，并各触发一次真实构建。本 Task 全部改动在仓库之外，必须逐条勾选，不得依赖代码审查发现遗漏。

## Task 5：文档、台账与原仓库归档

1. 合并两份 `EXECUTION_LEDGER.md`，补一条本次合并记录。
2. 合并 `docs/superpowers/plans` 与 `specs`，修正 7 份跨仓 plan 中的 `../fusion-ui/` 相对路径。
3. 根 `CLAUDE.md` 改为导航，`apps/api/CLAUDE.md` 与 `apps/ui/CLAUDE.md` 承载应用级约定；`AGENTS.md` 同理。`.agents/skills/` 中 10 个 skill 合并去重，`fusion-next-step` 两边各有一份需统一。
4. 两个原仓库置为 archived，README 指向新仓库。

**验证：** `fusion-next-step` skill 能正确读取合并后台账；跨仓 plan 中不再存在 `../fusion-ui/` 路径；两个原仓库已归档。

## 不做什么

- 不改任何业务代码。目录搬迁之外，`app/` 与 `src/` 下不产生 diff。
- 不引入 npm workspace、pnpm workspace、Nx、Turborepo 等 monorepo 工具。当前两个应用零共享代码，工具链只会增加复杂度。
- 不在本轮做 OpenAPI 契约同步。`scripts/export_openapi.py` 已存在但 UI 未消费，合并后具备条件，另开 plan。
- 不合并 `docker-compose.yml`。两边网络拓扑不同（api 侧有 `middleware`、`litellm_net`、`flyai` 三个网络），合并需单独评估。
- 不调整 self-hosted runner 的机器配置或数量。

## 风险与回退

| 风险 | 触发点 | 回退方式 |
|---|---|---|
| 路径改错导致发布失败 | Task 1 | 原仓库未归档，直接切回原仓库发布 |
| shell 抽取引入行为差异 | Task 2 | 逐个抽取、逐个验证，单个 revert 即可定位 |
| 回滚锚点在过渡期不可解析 | Task 3 | tag 规则切换前先并行打两种 tag，确认新规则可解析后再停旧规则 |
| 平台绑定遗漏 | Task 4 | 全部在仓库外，以 checklist 逐条勾选，不依赖 code review |
