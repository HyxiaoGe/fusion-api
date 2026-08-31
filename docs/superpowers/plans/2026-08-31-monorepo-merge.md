# Fusion Monorepo 合并实施计划

> Spec: 本文件自带依据章节，无独立 spec
> Base: `fusion-api@43cee73` / `fusion-ui@77b7fc2`（2026-08-31 复核基线）
> Branch: `feat/monorepo-merge`
> Target: 新建仓库 `HyxiaoGe/fusion`
> Review: PR #83 第一轮评审已受理，修订记录见文末

**目标：** 把 `fusion-api` 与 `fusion-ui` 合并为单一仓库 `fusion`，保留两边完整 git 历史，并把当前分散在两处、共约 3600 行的发布流水线收敛为一份参数化的可复用 workflow + 一组可本地执行的部署脚本。

**本轮范围：** 仓库结构、git 历史、CI/CD、宿主机部署状态、部署平台绑定、agent 与文档配置。不改任何业务代码，不改 Redis Stream 两段式流，不改 API 契约，不借机重构 `app/` 或 `src/` 下的任何模块。

**总体架构：** 新仓库采用 `apps/api` + `apps/ui` 两个应用目录，根目录只保留跨应用的编排物。发布流水线拆成三层：顶层 orchestrator 负责变更检测与 `API → UI` 顺序编排，`_deploy-app.yml` 是参数化的 `workflow_call` 可复用 workflow，`ops/deploy/*.sh` 承载实际 shell 逻辑并可脱离 GitHub Actions 本地执行。

## 依据

以下数据基于 `fusion-api@43cee73` / `fusion-ui@77b7fc2` 实测，可复核：

- **代码零耦合。** `fusion-api` 516 个 `.py`、699 个文件；`fusion-ui` 524 个 `.ts`/`.tsx`、636 个文件。交叉计数为 0（api 侧 `.ts`/`.tsx` 为 0，ui 侧 `.py` 为 0）。无共享包、无 workspace 依赖、无交叉 import。
- **历史合并成本低。** 两边 `.git` 分别为 5.5M / 2.8M，常规 `merge --allow-unrelated-histories` 即可完整保留，无需 `filter-repo`。
- **发布流水线已被手工写了两遍。** 两边步骤名逐字相同的有 14 个（去重后），另有 3 对同职能不同名（`Capture current deployment` ↔ `…for rollback`、`Rollback previous deployment` ↔ `Roll back failed deployment`、`Smoke check candidate` ↔ `Verify health`）。抽象成可复用 workflow 有实证支撑。
- **`deploy.yml` 的体量是内联 shell，不是 YAML 编排。** 全文 2563 行中 `deploy-dev` 单个 job 占 2339 行（L189–L2527），其中四个步骤合计 1757 行（占该 job 的 75%）：`Pull and restart fusion-api` 693、`Roll back failed deployment` 476、`Capture current deployment for rollback` 307、`Verify health` 281。该逻辑当前不可本地执行、不可单测，唯一验证方式是向 master 推送。
- **同职能步骤两边体量差 10 倍。** Capture 307 vs 30 行，Rollback 476 vs 65 行。差额中一部分是 API 的真实约束（alembic 迁移、flyai-adapter 必须与 api 同 SHA、litellm governance 与 model management 两个 worker 的暂停/恢复），一部分是重复的防御性检查。分类结论见 Task 4，本计划不预判。
- **双仓成本已有记录。** `docs/superpowers/plans/2026-06-28-agent-run-continuation.md` 明确写有"这份 spec 横跨 fusion-api 和 fusion-ui……实施时使用一个工作分支；最终按子仓分别提交，避免把半成品多次推送触发流水线"。
- **运行时已经耦合。** `fusion-ui/next.config.js` 通过 rewrites 把 `/api/*` 同源代理到 fusion-api，SSE 走此通道；两边共用同一组 self-hosted runner 类型、同一个阿里云 ACR、同一台 dev server，且**共用同一个宿主机工作目录 `~/project/fusion`**。

## Global Constraints

- **搬迁、切换与流水线重写必须分三阶段，不得合并。** 流水线操作真实 docker 重启、alembic 迁移与用户上传文件目录，混做会使发布故障无法二分定位。
- **旧仓库在最终验收通过前保持完整发布与回滚能力。** 不得在切换完成前注销旧仓 runner、撤销旧仓 secrets 或归档旧仓。
- **宿主机持久化状态与代码 checkout 路径必须分开处理。** 二者当前在 `~/project/fusion` 下混放，不得以同一次路径替换处理。
- `paths:` **不得用作 event-level 过滤**（见 P0-4）。变更检测由 workflow 内首个 `changes` job 承担，且始终提供一个恒定存在的 required gate job。
- 跨应用顺序由顶层 orchestrator 的 DAG 保证，concurrency 只负责防并发，不承担顺序语义。
- 抽取 shell 到 `ops/deploy/` 时**逐字搬运**，不得顺手"优化"。行为差异必须为零，抽取与改写分属不同 Task。
- 不删除任何现有校验分支。合并后无法确认必要性的检查一律保留并登记待评估，不得就地删除。
- 每个 Task 有独立可执行验收条件；真实发布与回滚只在 Task 边界执行，不在每个脚本搬运后执行（见 P1-8）。
- 提交信息中文，格式 `<type>: <中文描述>`，包含 `Co-Authored-By`。

## 路径与状态清单（Path Inventory）

目录下沉**不是**纯路径替换。以下为实测的逐项清单，每项在 Task 1 / Task 2 中必须有对应处理与测试：

### A 类：宿主机持久化状态（最高风险，Task 2 处理）

| 位置 | 内容 | 风险 |
|---|---|---|
| `deploy.yml` L587 / L658 / L1994 | `cd ~/project/fusion` | 宿主机工作目录，与仓库名耦合 |
| `build-and-deploy.yml` L282 / L391 | `cd ~/project/fusion` | 同上，两应用共用同一目录 |
| `deploy.yml` L1036/1042/1077/1206 | `./fusion-api/storage/files` bind mount | **用户上传文件持久化**，路径变更会挂载到空目录 |
| `deploy.yml` L1819 | `${HOME}/project/fusion/.env` | 运行时环境变量来源 |
| `ops/litellm/fusion-litellm-cost-sync.service` | `%h/project/fusion/fusion-api` ×3（L5/L6 `AssertPathExists`、L13 `WorkingDirectory`） | **唯一直接耦合仓库 checkout 的 unit**，路径错则 unit 启动失败 |

宿主机当前布局为 `~/project/fusion/fusion-api/…`，目录名直接镜像仓库名。**该布局与新仓库的 `apps/api` 结构不一致，且不能自动跟随。**

**仓库内已有正确范式可循：** 另外两个 unit（`fusion-litellm-governance.service`、`fusion-litellm-model-management.service`）的 `WorkingDirectory` 指向 `%h/.local/share/fusion/litellm-governance-current` 与 `%h/.local/share/fusion/litellm-model-management-current`，即已与仓库 checkout 解耦的暂存目录。Task 2 应把 `cost-sync` 收敛到同一范式，而不是简单替换其路径字面量。

### B 类：测试硬读根 `.github`（Task 1 处理）

| 文件 | 引用数 | 失效方式 |
|---|---|---|
| `test/test_model_management_deploy_config.py` | 10 | `ROOT / ".github/..."` |
| `test/test_ci_container_contract.py` | 7 | `ROOT / ".github/..."` |
| `test/test_ci_cd_permission_boundary.py` | 6 | `ROOT / ".github/..."` |
| `test/test_knowledge_deploy_config.py` | 1 | **`Path(".github/...")` 为 CWD 相对**，失效方式与上面三者不同 |
| `test/test_litellm_governance_units.py` | 1 | `ROOT / ".github/..."` |
| `fusion-ui/src/scripts/buildAndDeployWorkflow.test.ts` | 12 | `process.cwd()` 相对 |

`buildAndDeployWorkflow.test.ts` L168–L169 通过 `filesUnder()` **枚举整个 `.github/workflows` 与 `.github/actions` 目录**并对全部结果做断言。合并到共享根 `.github/` 后，该测试会扫到 API 侧 workflow 并失败。这不是路径替换能解决的，需改为按应用限定扫描范围。

### C 类：Action 与构建上下文（Task 1 处理）

- `fusion-ui/.github/workflows/build-and-deploy.yml` L98：`uses: ./.github/actions/windows-docker-build`（本地 composite action，`defaults.run.working-directory` 对其无效）
- 两边 `.github/release-safety.yml` 与 `.github/scripts/release-safety-contract.sh`（contract script 当前假定仓库根唯一一份）
- 两边 `.github/scripts/*` 的调用路径
- Dockerfile build context（`.` → `apps/api` / `apps/ui`）
- `actions/setup-node` 的 `cache-dependency-path`
- 7 份跨仓 plan 中的 `../fusion-ui/` 相对路径与文档绝对路径

## Task 0：新仓基建与恢复点（新增）

前置于一切代码搬迁。目的是让 Task 1 的验收在新仓可执行，同时保证旧仓不失能。

1. 创建 `HyxiaoGe/fusion`，默认分支 `master`。
2. 为两个原仓库各打一份 `git bundle` 恢复点，异地留存。
3. 复制 `dev` Environment 及其 protection rule、branch policy、branch protection；secrets/variables 按 repo 层与 Environment 层分别对应建立。
4. **新增 runner 实例**（不迁移旧仓 runner），并为全部 runner 增加应用维度标签 `fusion-api` / `fusion-ui`；workflow 的 `runs-on` 同步改为带应用标签的组合。旧仓四个 runner 原样保留至 Task 5。
5. 产出宿主机状态清单：`~/project/fusion` 下的目录、bind mount 源、`.env`、systemd unit 的实际 `WorkingDirectory`，逐项记录当前值。

**验收：** 新仓可跑通一个 hello-world workflow，分别命中 `fusion-api` 与 `fusion-ui` 标签的 runner；旧仓两条流水线仍可正常发布与回滚（各执行一次）；bundle 可在临时目录成功还原。

## Task 1：历史合并与目录下沉

**本 Task 不改变任何部署行为，只验证 PR CI 与镜像构建。**

1. 两仓各自先在原仓库内 `git mv` 到 `apps/api` / `apps/ui`，再以 `merge --allow-unrelated-histories` 引入，避免根目录文件互相覆盖。
2. 12 个同名根文件按归属下沉：`CLAUDE.md`、`AGENTS.md`、`README.md`、`.gitignore`、`.dockerignore`、`.env.example`、`Dockerfile`、`docker-compose.yml`、`railway.json`、`.agents/`、`docs/`、`scripts/`。
3. 逐项处理路径清单 B 类与 C 类：
   - 5 个 API 测试文件的 25 处引用改为应用根相对，并统一 `Path()` 与 `ROOT /` 两种写法；
   - `buildAndDeployWorkflow.test.ts` 的 `filesUnder()` 扫描范围限定到本应用 workflow；
   - `release-safety.yml` 拆为 `apps/api/release-safety.yml` 与 `apps/ui/release-safety.yml`，`release-safety-contract.sh` 改为接受契约文件路径参数；
   - 本地 composite action 路径、Dockerfile context、`cache-dependency-path` 逐一修正。
4. PR CI 改造为：始终触发 → 首个 `changes` job 判定 `api` / `ui` / `shared` → 应用 job 用 `if` 跳过 → 末尾恒定 required gate job。明确 `.github/**`、`ops/**`、根配置、共享文档各触发哪一侧。
5. 部署 workflow 原样复制两份进新仓但**保持 disabled**，不接管发布。

**验收：** 新仓 PR CI 全绿；`apps/api` 与 `apps/ui` 的单测与 lint 在新仓通过（含上述被改动的 6 个测试文件）；两个应用的镜像可在新仓成功构建（不推送、不部署）；只改一侧的 PR 不触发另一侧的应用 job，但 required gate 仍产出成功结论。

## Task 2：受控 cutover

本 Task 是唯一改变生产行为的一步，独立执行、独立回退。

1. **宿主机状态迁移**（路径清单 A 类）：
   - 先备份 `~/project/fusion/fusion-api/storage/files`；
   - 确立与仓库布局解耦的稳定数据目录（例如 `~/project/fusion/data/api/storage/files`），bind mount 指向稳定目录；
   - 代码 checkout 目录与数据目录分离，必要时以软链接兼容过渡期；
   - 把 `cost-sync` unit 的 `WorkingDirectory` 与 `AssertPathExists` 收敛到 `%h/.local/share/fusion/*-current` 范式（与另两个 unit 一致），`systemctl --user daemon-reload` 后逐个验证启动；
   - `${HOME}/project/fusion/.env` 的位置与读取方保持一致。
2. 启用新仓顶层 orchestrator：`detect changes → deploy API → deploy UI`，只改一侧时跳过另一侧。环境级 concurrency 仅防并发。
3. 首次切换选择低峰时段，切换后旧仓 workflow 置为 disabled 但不删除。

**验收：** 新仓完成一次真实 API 发布 + 回滚；一次真实 UI 发布 + 回滚；一次同时修改两边的提交，确认执行顺序为 API → UI；确认上传文件在切换前后可正常读取（切换前上传一个文件，切换后下载校验）；三个 systemd unit 均 active（`cost-sync` 需确认已脱离仓库 checkout 路径）；**旧仓回退演练一次**（重新 enable 旧 workflow 并成功发布）。任一项失败即回退到旧仓。

## Task 3：抽取 shell 到 ops/deploy

按体量倒序逐个抽取：`Pull and restart`(693) → `Roll back failed deployment`(476) → `Capture current deployment`(307) → `Verify health`(281) → 其余。

1. 每个步骤的 `run: |` 块整体移入 `ops/deploy/<name>.sh`，入参改为显式环境变量或位置参数，YAML 侧只保留调用。
2. 每个脚本配 fixture + dry-run + contract test，覆盖至少：正常路径、目标 SHA 非法、回滚锚点缺失三种情况。PR CI 增加脚本单测门禁。
3. **不在每个脚本搬运后向 master 发布。** 真实发布与回滚在 Task 边界执行一次。

**验收：** 全部脚本单测通过；`deploy.yml` 的 YAML 行数降至 400 行以内；Task 结束时执行一次完整 API → UI 真实发布与回滚验收。

## Task 4：参数化 reusable workflow 与发布台账

前置：Task 3 完成，两边 shell 已成为可读独立文件，重复之处可见。

1. 以 UI 侧精简实现为基线，逐条比对 API 侧多出的检查，分类为「真实约束」与「重复防御」。真实约束做成可选 hook，重复防御合并。**分类结果登记在本文件，本 Task 内不删除任何检查。**
2. 抽出 `_deploy-app.yml`，参数：应用名、镜像仓库、健康检查端点、迁移开关、依赖服务列表、回滚锚点校验策略。
3. **保留各应用现有 ACR repository 与 `<sha>` tag**（`seanfield/fusion-api` 与 `seanfield/fusion-ui` 已天然区分应用，见 P1-6）。新增 per-app 发布台账/manifest，记录每个应用的 last deployed SHA，解决 path-filtered 发布后两应用 SHA 分叉导致的回滚目标定位问题。
4. Vercel root directory 改为 `apps/ui`；Railway 两个服务的 root directory 分别指向 `apps/api` / `apps/ui`。
5. 分支保护、required checks 按新 workflow 的 job 名重建（gate job 为 required，应用 job 不是）。

**验收：** 两应用各发布、各回滚一次；构造仅改 UI 的提交，确认 API 的 last deployed SHA 未变且回滚目标仍可唯一解析；平台绑定逐条在控制台勾选确认并各触发一次真实构建。

## Task 5：文档、台账与旧仓归档

1. 合并两份 `EXECUTION_LEDGER.md`，补一条本次合并记录。
2. 合并 `docs/superpowers/plans` 与 `specs`，修正 7 份跨仓 plan 中的 `../fusion-ui/` 相对路径。
3. 根 `CLAUDE.md` 改为导航，`apps/api/CLAUDE.md` 与 `apps/ui/CLAUDE.md` 承载应用级约定；`AGENTS.md` 同理。`.agents/skills/` 中 10 个 skill 合并去重（`fusion-next-step` 两边各有一份需统一）。
4. **平台元数据边界**（见 P1-9）：git 历史合并不迁移 Issues、PR、Actions runs、releases、webhooks、deploy keys、Environment protection、branch rules。明确：
   - 复制：Environment protection、branch rules、secrets/variables、webhooks；
   - 仅旧仓保留：Issues、PR、Actions run 历史、releases；
   - commit message 中的 `#NN` 在新仓不再解析为原 PR，归档说明中标注旧仓地址供追溯；
   - 旧仓 runner 与旧仓 workflow 在本 Task 退役，此前一直保留。
5. 两个原仓库置为 archived，README 指向新仓库。

**验收：** `fusion-next-step` skill 能正确读取合并后台账；跨仓 plan 中不再存在 `../fusion-ui/` 路径；旧仓 runner 已注销、仓库已归档。

## 不做什么

- 不改任何业务代码。目录搬迁之外，`app/` 与 `src/` 下不产生 diff。
- 不引入 npm/pnpm workspace、Nx、Turborepo。当前两应用零共享代码，工具链只增复杂度。
- 不在本轮做 OpenAPI 契约同步。`scripts/export_openapi.py` 已存在但 UI 未消费，合并后具备条件，另开 plan。
- 不合并 `docker-compose.yml`。两边网络拓扑不同（api 侧有 `middleware`、`litellm_net`、`flyai` 三个网络），需单独评估。
- 不把两应用镜像并入同一个 ACR repository。
- 不迁移旧仓 Issues 与 PR 历史。

## 风险与回退

| 风险 | 触发点 | 回退方式 |
|---|---|---|
| 上传文件目录挂载到空目录 | Task 2 | 切换前全量备份 + 切换后下载校验；失败则恢复备份并回退 bind mount |
| `cost-sync` unit 找不到 WorkingDirectory | Task 2 | unit 文件版本化，`daemon-reload` 前保留旧版本，逐个验证后再提交 |
| 新仓 runner 未就绪导致无法发布 | Task 1/2 | 旧仓 runner 与 workflow 全程保留至 Task 5，可随时 re-enable |
| required check 因 path 跳过而永久 Pending | Task 1 | 恒定 gate job 始终产出结论，不使用 event-level `paths:` |
| 跨应用发布顺序错误导致 SSE 契约不匹配 | Task 2 | orchestrator DAG 强制 API → UI；失败则整条流水线中止 |
| shell 抽取引入行为差异 | Task 3 | 逐个抽取 + 单测，单个 revert 即可定位 |
| 整体方案失败 | 任意 | Task 0 的 git bundle + 未归档的旧仓，可完整回到合并前状态 |

## 评审修订记录

PR #83 第一轮评审共 9 条，全部受理。逐条处理如下：

| # | 评审意见 | 核实结果 | 处理 |
|---|---|---|---|
| P0-1 | 计划基线过期 | **成立。** master 已从 `c5b0775` 前进 57 个提交至 `43cee73`，`deploy.yml` 2553 → 2563 行 | 已合并 master，全部数字基于 `43cee73` / `77b7fc2` 重算（py 505→516，文件 678→699，ts/tsx 522→524） |
| P0-2 | Task 1 与 Task 4 顺序矛盾 | **成立。** 原计划要求 Task 1 完成真实发布，但 secrets/runner/Environment 到 Task 4 才迁移 | 新增 Task 0 承载新仓基建；Task 1 改为只验证 PR CI 与镜像构建，不做发布；采用新增 runner 实例 + 应用标签，旧仓 runner 保留至 Task 5 |
| P0-3 | concurrency 不保证 API → UI 顺序 | **成立。** concurrency 仅防并发，等待顺序不等于分发顺序 | 改为顶层 orchestrator DAG `detect changes → deploy API → deploy UI`，写入 Global Constraints |
| P0-4 | `paths:` 与 required check 冲突 | **成立。** 被 path filter 跳过的 workflow 不产出结论，required check 会永久 Pending；且 changed-files 过滤有 300 文件上限，首次迁移提交远超此数 | 移除 event-level `paths:`；改为始终触发 + `changes` job + `if` 跳过 + 恒定 required gate |
| P0-5 | 部署脚本含宿主机真实路径与持久化数据 | **成立。** 实测：两边 workflow 共 5 处 `cd ~/project/fusion`；4 处 `./fusion-api/storage/files` bind mount；1 处 `${HOME}/project/fusion/.env`。systemd 侧经逐个核实为 **1 个** unit（`cost-sync`）耦合仓库 checkout 路径，另两个已用 `%h/.local/share/fusion/*-current` 暂存目录解耦 | 新增「路径与状态清单」A 类；宿主机状态迁移独立为 Task 2，含备份、稳定 data 目录、软链接兼容、systemd 更新与上传文件读写校验；`cost-sync` 按仓库内既有的解耦范式收敛 |
| P1-6 | 镜像 tag 歧义判断不成立 | **成立，原判断有误。** 实测 `seanfield/fusion-api` 与 `seanfield/fusion-ui` 已是不同 ACR repository，同 SHA 不产生歧义 | 撤销 `<app>-<sha>` 方案，保留各自 `<sha>` tag；改为新增 per-app 发布台账解决 last deployed SHA 分叉 |
| P1-7 | "相对路径均不变"过于乐观 | **成立且可量化。** 实测 5 个 API 测试文件 25 处硬读根 `.github`（其中 `test_knowledge_deploy_config.py` 为 CWD 相对，失效方式不同）；UI `buildAndDeployWorkflow.test.ts` 12 处，且 L168 枚举整个 workflows 目录做断言，合并后会扫到 API 侧 workflow | 新增「路径与状态清单」B/C 类逐项列出，每项在 Task 1 有对应处理与验收 |
| P1-8 | 每抽一个脚本就发布回滚成本过高 | **成立。** master 是发布通道，不适合作为单步搬运的测试环境 | 改为每脚本 fixture/dry-run/contract test，真实发布与回滚只在 Task 边界执行；Task 3 末尾保留一次完整 API → UI 验收 |
| P1-9 | 缺少平台元数据迁移边界 | **成立。** 原计划未涉及 | Task 5 新增元数据边界章节，明确复制项、仅旧仓保留项、`#NN` 引用语义、旧仓退役时点 |

**未独立核实的项：** P0-2 中关于 runner 为 repo-scoped、标签仅为通用值、`dev` Environment 仅允许 `master`、secrets 分布于 repo 与 Environment 两层的描述，来自仓库 owner 的评审陈述，本次未通过 API 独立验证。Task 0 第 5 步会在执行时逐项确认实际配置。
