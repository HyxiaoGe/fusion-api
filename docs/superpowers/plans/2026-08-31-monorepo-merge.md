# Fusion Monorepo 合并实施计划

> Spec: 本文件自带依据章节，无独立 spec
> Base: `fusion-api@43cee73` / `fusion-ui@77b7fc2`（2026-08-31 复核基线）
> Branch: `feat/monorepo-merge`
> Target: 新建仓库 `HyxiaoGe/fusion`
> Review: PR #83 三轮评审（9 + 6 + 5 条）均已受理，修订记录见文末

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
- **宿主机持久化状态必须迁出仓库 checkout 树，而不只是换一个子目录。** 目标路径不得位于任何仓库 checkout 根之下（含新仓的 `~/project/fusion`）。持久化状态与代码 checkout 当前在 `~/project/fusion` 下混放。
- **跨仓互斥由 drain 门禁承担，`flock` 只覆盖新仓与宿主机侧操作。** GitHub concurrency group 只在单个 repository 内协调。**实测两个旧仓 workflow 均无任何 `flock` 调用**（`grep -c flock` = 0），而本计划不改动旧仓配置，因此仅在宿主机放置锁文件不产生跨仓互斥 —— 旧 workflow 会直接绕过。故：新仓 orchestrator 与宿主机手工操作持同一把 `flock`；cutover 期间的三仓互斥由 Task 2 的 drain 门禁保证（见 2.2）。
- **`disable` 不等于停止。** GitHub 的 disable 只阻止后续触发，不影响已 queued / in-progress 的 run。任何"冻结"步骤都必须配合对三个仓库在途 run 的查询、等待或取消，直至归零。
- **cutover 必须以「运行态归属新仓」收尾。** 任何回退演练之后都要重新切回新仓并复验，验收不得停在"旧仓重新发布成功"这一步。
- **secrets 不可从旧仓复制。** GitHub API 只能列出名称与元数据，不能读回值；一律从原始凭据源重新注入，找不到原始值的必须轮换，禁止从 workflow 日志或运行环境反向导出。
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
| `deploy.yml` L1036/1042/1077/1206 | `./fusion-api/storage/files` bind mount | 文件对象持久化，路径变更会挂载到空目录。**严重程度取决于生产实际后端**，见下方说明 |
| `deploy.yml` L1819 | `${HOME}/project/fusion/.env` | 运行时环境变量来源 |
| `ops/litellm/fusion-litellm-cost-sync.service` | `%h/project/fusion/fusion-api` ×3（L5/L6 `AssertPathExists`、L13 `WorkingDirectory`） | **唯一直接耦合仓库 checkout 的 unit**，路径错则 unit 启动失败 |

**关于 `storage/files` 的严重程度（待 Task 0 判定）：** 存储后端有三种 —— `local` / `minio` / `oss`（`.env.example` 默认 `local`），由宿主机 `${HOME}/project/fusion/.env` 的 `STORAGE_BACKEND` 决定，**不在仓库内，本计划无法确认生产实际取值**。

- 若为 `local`：`app/services/storage/local_storage.py` 的 `upload()` 直接写原始字节、`download()` 直接读回，该目录是**原件的唯一副本**；且 `app/db/models.py` 有两处 `storage_key = Column(String(600), nullable=False)` 与 `UniqueConstraint("storage_backend", "storage_key")` 硬引用，文件丢失会留下指向空对象的 DB 行，属数据不一致而非单纯缺文件。Task 2 的备份、四项校验与回退演练全部必要。
- 若为 `minio` / `oss`：对象存于远端，本地目录重要性大幅下降，Task 2 可相应减负。

注意 `deploy.yml` L1428 对 `/app/storage/files` 挂载的断言是**无条件**的，因此"挂载存在"不能证明后端为 `local`。另 L1036–1042 存在一段从容器内 `tar` 抢救文件的兼容路径，说明该数据历史上曾只存在于容器内并出过问题 —— 是这批数据值得谨慎对待的旁证。

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
- 跨仓 plan 中的相对路径与文档绝对路径。**实测存量很小**：`docs/superpowers/` 下含 `../fusion-ui/` 的仅 `2026-06-30-search-read-planner-ledger.md` 一份（另一份是本计划自身）；提到 `fusion-ui` 但不含该相对路径模式的另有 6 份，只需按需核对

## Task 0：新仓基建、恢复点与外部绑定清点

前置于一切代码搬迁。目的是让后续 Task 的验收在新仓可执行，同时保证旧仓不失能。**本 Task 不改动旧仓任何配置，也不执行真实发布。**

1. 创建 `HyxiaoGe/fusion`，默认分支 `master`。
2. **恢复点（含未提交资产）**：
   - 产出两仓的 dirty / untracked 精确清单（`git status --porcelain --untracked-files=all`）；
   - 未提交文件另做独立可恢复归档并记录哈希 —— `git bundle` 只包含 refs 与可达 object，**不含未跟踪文件**；
   - bundle 以完整 refs 创建（`--all`）并执行 `git bundle verify`；
   - bundle 与未提交资产各在临时目录还原一次并校验；
   - 迁移分支只从精确 HEAD 创建，不在旧仓 master 上执行目录下沉提交。
3. 复制 `dev` Environment 及其 custom branch policy。**branch protection 分两步 bootstrap**（P1-18）：本 Task 只复制**非 check 类**保护规则；required status check 暂不设置 —— 两个旧仓当前的 required check 名为 `PR container validation`，而新仓此时尚未产生过该 check，且 GitHub 要求 required check 在目标仓库最近 7 天内成功运行过，直接复制会使新仓 PR 永久 Pending。gate 在新仓成功运行一次后（Task 1）再设为 required。
4. **secrets 重新注入（非复制）**：
   - 通过 API 导出旧仓 secret **名称清单**（repo 层与 Environment 层分别导出）；
   - 从密码管理器、部署主机安全配置或原始凭据源向新仓注入；
   - 原始值不可得的 secret 一律轮换；
   - webhook secret 与 deploy key 私钥同此处理；
   - 新仓侧只验证存在性、权限边界与目标服务连通性，**禁止输出 secret 值**；
   - 逐项登记「来源、注入位置、验证结果、是否轮换」。
5. **新增 runner 实例**（不迁移旧仓 runner），并为全部 runner 增加应用维度标签 `fusion-api` / `fusion-ui`；workflow 的 `runs-on` 同步改为带应用标签的组合。旧仓四个 runner（`dev-server-fusion-api`、`windows-build-api-01`、`dev-server-fusion-ui`、`windows-build-01`）原样保留至 Task 5。
6. **宿主机状态清单**：`~/project/fusion` 下的目录、bind mount 源、`.env`、三个 systemd unit 的实际 `WorkingDirectory`，逐项记录当前值、文件数、总字节数、权限与 owner。**其中必须先判定 `STORAGE_BACKEND` 的实际取值**（读宿主机 `${HOME}/project/fusion/.env`，取值为 `local` / `minio` / `oss` 之一），该结果决定 Task 2 中 `storage/files` 迁移的分量：`local` 时该目录为原件唯一副本，全套备份与校验必做；`minio` / `oss` 时可减负。判定结果记入本文件。
7. **外部平台使用状态清单**：逐个 Vercel / Railway 服务记录 —— 是否活跃、跟踪哪个 repo 与 branch、负责 dev 还是 production、是否自动部署。**据此判定每个绑定归属 Task 2 还是 Task 4**：凡仍活跃且影响运行态的，必须并入 Task 2 的同一次受控 cutover；只有不影响当前运行态的纯重构项才留到 Task 4。
8. 在宿主机部署跨仓 `flock` 锁文件与获取脚本，供三个仓库的部署入口共用。

**验收：** 新仓可跑通 hello-world workflow 并分别命中 `fusion-api` / `fusion-ui` 标签的 runner；bundle 与未提交资产归档均通过还原验证；secrets 清单逐项登记完毕且新仓连通性验证通过；平台清单产出且每个绑定已判定 Task 归属。旧仓侧**只做只读核验**：runner 在线、workflow 配置未变、最近一次成功发布记录存在。**不在本 Task 执行真实发布或回滚**（真实旧仓回退演练集中到 Task 2 的 cutover runbook 执行一次）。

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
5. **部署 workflow 改造为 `workflow_call` 外壳**（P0-16）。实测两边现有 workflow 均**不暴露 `workflow_call`**，普通 workflow 无法被 orchestrator 以 job-level `uses` + `needs` 调用，用 `workflow_dispatch` 触发子 workflow 也无法在同一 run 内形成可证明的依赖链 —— 若不在此处理，Task 2 的 orchestrator 无可调用对象。故生成两份**逻辑原样、尚未参数化合并**的 app-specific reusable workflow：
   - `.github/workflows/_deploy-api.yml`，仅暴露 `workflow_call`；
   - `.github/workflows/_deploy-ui.yml`，仅暴露 `workflow_call`。

   这**只改变调用外壳，不合并也不优化部署逻辑**，仍满足"搬迁与重写分离"。两份在本 Task 保持不被任何已启用的 workflow 调用。Task 4 再合并为参数化 `_deploy-app.yml`。
6. **`changes` 与 required gate 行为契约**（P1-19）：
   - `changes` 使用 checkout 后的**完整 Git diff**，不依赖 GitHub 的 300-file event 过滤；
   - 分别定义 PR、push、首次 push、merge commit、`workflow_dispatch` 五种情形的 base / head；
   - required gate 使用 `if: always()`；
   - gate 必须验证「按 `changes` 结果**应当运行**的 app job 成功」，**不得**因 app job 为 `skipped` 就无条件通过；
   - API 未变化而 UI 变化时，UI job **不得**因 `needs: deploy-api` 处于 `skipped` 而被级联跳过（用 `always()` + 显式结果判定，不用裸 `needs`）；
   - 恒定 gate 的 display name 在本 Task 确定并记入文档，供 Task 0/4 的 branch protection 引用。
7. **`workflow_dispatch` 回滚契约**（P0-16）：
   - 目标应用参数 `api | ui | both`；
   - API / UI **各自独立**的 rollback SHA，不假设两应用相同；
   - 手动回滚时 `changes` **不得**按路径判定后跳过目标应用；
   - 明确单应用与双应用回滚的执行顺序、失败中止行为与最终台账更新规则。

**验收：** 新仓 PR CI 全绿；`apps/api` 与 `apps/ui` 的单测与 lint 在新仓通过（含上述被改动的 6 个测试文件）；两个应用的镜像可在新仓成功构建（不推送、不部署）；只改一侧的 PR 不触发另一侧的应用 job，但 required gate 仍产出成功结论；构造一次「API 未变、UI 变」的 PR，确认 UI job **未**被级联跳过且 gate 正确判定；构造一次「应运行的 app job 失败」，确认 gate **失败**而非放行；`_deploy-api.yml` / `_deploy-ui.yml` 通过 `workflow_call` 语法校验且未被任何已启用 workflow 调用。

## Task 2：受控 cutover

本 Task 是唯一改变生产行为的一步，独立执行、独立回退。**必须按 runbook 顺序执行，不得跳步。**

### 2.1 宿主机状态迁移（路径清单 A 类）

持久化状态一律迁出仓库 checkout 树。**执行前先取 Task 0 第 6 步判定的 `STORAGE_BACKEND` 结果**：非 `local` 时上传文件一项可按实际情况减负，其余不变。

| 内容 | 目标路径 |
|---|---|
| 上传文件 | `~/.local/share/fusion/api/storage/files` |
| 运行时配置 | `~/.config/fusion/runtime.env` |
| systemd release/current | `~/.local/share/fusion/*-current`（沿用既有范式） |

- 迁移前全量备份，并记录文件数、总字节数、权限与 owner，对文件清单生成校验摘要；
- bind mount 与 `.env` 读取方指向新路径；
- 把 `cost-sync` unit 的 `WorkingDirectory` 与 `AssertPathExists` 收敛到 `%h/.local/share/fusion/*-current` 范式（与另两个 unit 一致），`daemon-reload` 后逐个验证启动；
- 旧路径仅保留**有明确退役时间**的过渡软链接，退役时间写入本文件。

### 2.2 Cutover Runbook

1. 冻结三个仓库（旧 `fusion-api`、旧 `fusion-ui`、新 `fusion`）的 master 合并与手动 dispatch；
2. **Drain 门禁（显式，不可跳过）**：查询三个仓库全部 `queued` 与 `in_progress` 的 workflow run，等待其自然结束或安全取消，**直至三仓在途 run 归零**并二次确认。`disable` 只阻止后续触发，不影响已在途的 run，因此本步不能被"冻结"或"disable"替代；
3. disable 旧仓两条 workflow，**再次确认无在途 run**，然后 enable 新仓 orchestrator（`detect changes → deploy API → deploy UI`）；
4. 由新仓发布、验证，并记录部署所有权归新仓（新仓 orchestrator 与宿主机手工操作全程持 `flock`）；
5. **回退演练**：先 disable 新仓 → drain → enable 旧仓，由旧仓发布并验证成功；
6. **回切收尾（不可省略）**：disable 旧仓 → drain → enable 新仓，重新部署新仓当前 SHA，验证最终镜像 ID、发布台账与部署所有权**全部归新仓**。

每次 enable / disable 切换之间都要重跑第 2 步的 drain，三仓不得在任一时刻同时处于 enabled。第 6 步是本 Task 的终点：缺少它，验收会停在"旧仓重新发布成功"，而运行态实际停留在旧仓版本。

**关于跨仓锁的取舍：** 评审提出两个方案 —— (1) 先发独立 PR 给旧仓加同一把 `flock`；(2) 不宣称锁覆盖三仓，改用严格 drain。**本计划采用方案 2**，理由是方案 1 要在 cutover 前夕修改旧仓的部署 job，而旧仓正是失败时的唯一回退路径，动它会削弱兜底。配合 runbook 的 drain 门禁与"三仓不同时 enabled"纪律，风险窗口已被覆盖。若后续认为需要双保险，方案 1 可作为独立前置 PR 追加 —— 这是可推翻的决策。

### 2.3 外部平台绑定

Task 0 第 7 步判定为「仍活跃且影响运行态」的 Vercel / Railway 绑定，在本 Task 的同一次 cutover 内完成切换，不得延后到 Task 4。

**验收：** 新仓完成一次真实 API 发布 + 回滚；一次真实 UI 发布 + 回滚；一次同时修改两边的提交，确认执行顺序为 API → UI；上传文件迁移前后的**文件数、总字节数、权限/owner、校验摘要四项全部一致**，并另做一次单文件上传下载验证；三个 systemd unit 均 active 且 `cost-sync` 已脱离仓库 checkout 路径；runbook 每次切换前的 drain 均确认三仓在途 run 为 0；第 5 步旧仓回退演练成功；第 6 步回切后镜像 ID、台账、发布所有权均指向新仓。任一项失败即回退到旧仓并停止推进。

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
3. **保留各应用现有 ACR repository 与 `<sha>` tag**（`seanfield/fusion-api` 与 `seanfield/fusion-ui` 已天然区分应用，见 P1-6）。新增 per-app 发布台账解决两应用 last deployed SHA 分叉导致的回滚目标定位问题，其语义按下表固定（P1-20）：

   | 项 | 约定 |
   |---|---|
   | 权威来源 | **运行容器的 immutable image ref + image ID**（`docker inspect` 可得），不是台账文件 |
   | 台账定位 | 权威来源的**可恢复投影**，用于快速查询与审计，丢失可从运行态重建 |
   | 存储位置 | 宿主机 checkout 树之外（`~/.local/share/fusion/<app>/release-ledger.json`） |
   | 更新时机 | 发布验证通过后，在**同一把部署 `flock` 内原子替换**（写临时文件 + `rename`） |
   | 失败与自动回滚 | 回滚完成后按回滚后的实际运行镜像重写台账，不保留失败中间态 |
   | 手动回滚 | 按 `workflow_dispatch` 的目标应用与 SHA 更新对应条目，另一应用条目不变 |
   | 宿主机丢失 | 从运行容器 image ref + ID 重建；容器也不存在则从 ACR tag 与部署记录人工恢复 |
   | 禁止项 | **不得把部署时变化的 manifest commit 回 master** —— 会递归触发发布流水线 |
4. 处理 Task 0 判定为「不影响当前运行态」的平台项（例如非活跃服务的 root directory 归位）。**影响运行态的绑定已在 Task 2 切换完毕，本 Task 不得留有此类项。**
5. **required context 迁移**（P1-18）：仅在 gate 的 job display name 确实变化时迁移 required context，迁移后**验证旧 context 已从保护规则中移除**，避免残留一个永不再产生的 required check。应用 job 始终不设为 required。

**验收：** 两应用各发布、各回滚一次；构造仅改 UI 的提交，确认 API 的 last deployed SHA 未变且回滚目标仍可唯一解析；平台绑定逐条在控制台勾选确认并各触发一次真实构建。

## Task 5：文档、台账与旧仓归档

1. 合并两份 `EXECUTION_LEDGER.md`，补一条本次合并记录。
2. 合并 `docs/superpowers/plans`（54 份）与 `specs`（23 份），修正 `2026-06-30-search-read-planner-ledger.md` 中的 `../fusion-ui/` 相对路径（实测存量仅此一份）。

   **同时做一次分诊，不要原样搬运。** 实测当前两套记录体系互不索引：台账 `EXECUTION_LEDGER.md` 按「领域 + commit SHA」追踪（71 处 SHA 引用），54 份 plan 中仅 **4 份**被台账点名；checkbox 追踪已基本废弃（25 份有 box 但 0 勾选、10 份无 box、仅 9 份全勾）。因此**单看一份 plan 无法判断它是活的还是已完成** —— 这正是 `AGENTS.md` 第 11 条要求"必须先读台账再查 git log"的原因。

   迁移时至少做到：为每份 plan 标注「已完成 / 进行中 / 已废弃」，或按状态分目录（如 `plans/archive/`）。**否则等于把这个索引缺口原样搬进新仓。** 本步不要求补齐 checkbox，只要求可判定死活。
3. 根 `CLAUDE.md` 改为导航，`apps/api/CLAUDE.md` 与 `apps/ui/CLAUDE.md` 承载应用级约定；`AGENTS.md` 同理。`.agents/skills/` 中 10 个 skill 合并去重（`fusion-next-step` 两边各有一份需统一）。
4. **平台元数据边界**（见 P1-9）：git 历史合并不迁移 Issues、PR、Actions runs、releases、webhooks、deploy keys、Environment protection、branch rules。明确：
   - 复制：Environment protection、branch rules、secrets/variables、webhooks；
   - 仅旧仓保留：Issues、PR、Actions run 历史、releases；
   - commit message 中的 `#NN` 在新仓不再解析为原 PR，归档说明中标注旧仓地址供追溯；
   - 旧仓 runner 与旧仓 workflow 在本 Task 退役，此前一直保留。
5. 两个原仓库置为 archived，README 指向新仓库。

**验收：** `fusion-next-step` skill 能正确读取合并后台账；跨仓 plan 中不再存在 `../fusion-ui/` 路径；旧仓 runner 已注销、仓库已归档。

## 不做什么

- 不改任何业务代码。目录搬迁之外，`apps/api/app/` 与 `apps/ui/src/` 下的**业务模块**不产生 diff。**唯一例外**：`apps/ui/src/scripts/buildAndDeployWorkflow.test.ts` 属路径清单 B 类（CI 契约测试，需限定 `filesUnder()` 扫描范围），虽位于 `src/` 之下但必须修改。API 侧 5 个同类测试位于 `test/`，与 `app/` 平级，不涉及此例外。
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
| 三仓 workflow 同时操作 dev server | Task 2 | 宿主机 `flock` 跨仓互斥 + cutover 期间冻结三仓发布 |
| 验收停在旧仓、运行态未归新仓 | Task 2 | runbook 第 6 步回切收尾为强制项，验镜像 ID / 台账 / 所有权三项 |
| secret 原始值不可得 | Task 0 | 该项一律轮换，不从日志或运行环境反向导出 |
| 未提交资产在迁移中丢失 | Task 0 | bundle 之外另做未跟踪文件归档 + 哈希 + 还原验证 |
| 平台仍跟踪旧仓导致发布链未真正切换 | Task 2 | Task 0 产出平台使用状态清单，活跃绑定并入 Task 2 同一次 cutover |
| orchestrator 无可调用的部署单元 | Task 2 | Task 1 先产出 `_deploy-api.yml` / `_deploy-ui.yml` 两份 `workflow_call` 外壳 |
| 在途 run 绕过冻结继续操作 dev server | Task 2 | runbook 第 2 步 drain 门禁，每次 enable/disable 切换前重跑 |
| 新仓 required check 从未运行导致 PR 永久 Pending | Task 0 | branch protection 分两步 bootstrap，gate 成功运行一次后再设 required |
| gate 因 app job skipped 而误放行 | Task 1 | gate 用 `if: always()` 并验证「应运行的 job 成功」，含专门的反例验收 |
| 台账与运行态不一致 | Task 4 | 权威来源为运行容器 image ref + ID，台账仅为可恢复投影，锁内原子替换 |
| 整体方案失败 | 任意 | Task 0 的 git bundle + 未提交资产归档 + 未归档的旧仓，可完整回到合并前状态 |

## 关于本 PR 自身的合并边界

`.github/workflows/deploy.yml` 的触发条件为 `on.push.branches: [master]`，**无 `paths` 过滤**。因此即便本 PR 只新增文档，合并到 master 仍会触发完整的 API 发布流水线：`publish`（Windows runner 构建并推 ACR）+ `deploy-dev`（alembic 迁移 + 容器重启）。

本 PR 的合并**不是操作层面无副作用的文档合并**，应按一次正式发布处理：选择合适时间窗口，合并后单独记录 CI 结论、部署结果与运行态验证。

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

**第一轮遗留的「未独立核实」项已由第二轮复核结清**，见下节。

### 第二轮（PR #83，6 条）

owner 已通过 GitHub API 独立复核第一轮标注为"未独立核实"的项，结论属实：runner 为 `dev-server-fusion-api` / `windows-build-api-01` / `dev-server-fusion-ui` / `windows-build-01`，均 repo-scoped 且标签仅通用值；两仓 `dev` Environment 均用 custom branch policy 且只允许 `master`；secrets 确实分布于 repo 层与 Environment 层。该标注已从文档中移除。

| # | 评审意见 | 核实结果 | 处理 |
|---|---|---|---|
| P0-10 | 跨仓 cutover 缺互斥与最终状态闭环 | **成立。** GitHub concurrency group 只在单 repository 内协调，无法阻止三仓同时操作同一台 dev server | Global Constraints 增加宿主机 `flock` 跨仓互斥与"cutover 必须以运行态归属新仓收尾"两条；Task 2 新增 6 步 cutover runbook，第 6 步回切收尾为强制项；Task 0 增加锁文件部署 |
| P0-11 | secrets 不能从旧仓直接复制 | **成立。** GitHub API 只能列出名称与元数据，不能读回值 | Global Constraints 增加禁止复制与禁止反向导出；Task 0 第 4 步改为名称清单导出 + 原始源重新注入 + 不可得则轮换 + 逐项登记来源/注入位置/验证结果/是否轮换 |
| P0-12 | 持久化状态仍未脱离 checkout | **成立，原方案自相矛盾。** 初稿示例 `~/project/fusion/data/api/storage/files` 仍位于新仓 checkout 根 `~/project/fusion` 之下，与自身 Global Constraint 冲突 | 固定为仓库外路径：上传文件 `~/.local/share/fusion/api/storage/files`、运行时配置 `~/.config/fusion/runtime.env`；Global Constraint 改为"必须迁出 checkout 树，而不只是换子目录"；验收增加文件数、总字节数、权限/owner 与校验摘要四项比对 |
| P0-13 | 平台 cutover 边界需前移判定 | **成立。** Task 2 切换 Actions 发布权但平台绑定留到 Task 4，会导致发布链未真正切换 | Task 0 新增第 7 步平台使用状态清单（活跃性、跟踪 repo/branch、dev 或 production、是否自动部署）并据此判定 Task 归属；活跃且影响运行态的并入 Task 2；Task 4 收窄为不影响运行态的项 |
| P0-14 | git bundle 不保护未提交资产 | **原理成立，计数不适用于本仓库当前状态。** `git bundle` 只含 refs 与可达 object，确实不含未跟踪文件。但评审所述"9 个未跟踪 plan/spec 文件"在本会话 checkout（`a5ad5b5`）中不成立 —— `git status --porcelain --untracked-files=all` 返回 0 项，工作区干净。该状态应属 owner 本机工作区；由于迁移将从该工作区执行，防护仍然必要 | Task 0 第 2 步增加 dirty/untracked 精确清单、未提交文件独立归档与哈希、`--all` 创建 bundle 并 `git bundle verify`、两者各做还原验证、迁移分支只从精确 HEAD 创建 |
| P1-15 | Task 0 不必重复真实发布与回滚 | **成立。** Task 0 不修改旧仓 runner/workflow，无改动发布的风险高于收益 | Task 0 验收改为只读核验（runner 在线、workflow 配置未变、最近成功记录存在）；真实旧仓回退演练集中到 Task 2 runbook 第 5 步执行一次 |
| — | 本 PR 合并会触发 master 发布 | **成立。** 实测 `deploy.yml` 为 `on.push.branches: [master]` 且无 `paths` 过滤，合并将触发 `publish` + `deploy-dev`（含 alembic 迁移与容器重启） | 新增「关于本 PR 自身的合并边界」章节，明确按一次正式发布处理 |

### 第三轮（PR #83，5 条）

| # | 评审意见 | 核实结果 | 处理 |
|---|---|---|---|
| P0-16 | Task 2 orchestrator 在 Task 4 前无可调用的部署单元 | **成立。** 实测两边 workflow 均不暴露 `workflow_call`（`grep workflow_call` 无命中）。普通 workflow 无法被 job-level `uses` + `needs` 调用，`workflow_dispatch` 也无法在同一 run 内形成可证明的依赖链 —— 原顺序下 Task 2 的 DAG 无对象可调 | Task 1 新增第 5 步：产出 `_deploy-api.yml` / `_deploy-ui.yml` 两份**逻辑原样、仅暴露 `workflow_call`** 的外壳，只改调用形式不动部署逻辑，仍满足搬迁与重写分离；Task 4 再合并为参数化 `_deploy-app.yml`。同时新增第 7 步 `workflow_dispatch` 回滚契约（目标应用 `api\|ui\|both`、双 SHA 独立、手动回滚不被 `changes` 跳过、顺序与中止规则） |
| P0-17 | `flock` 未覆盖旧仓，disable 前缺 drain | **成立。** 实测两个旧仓 workflow 的 `flock` 调用数均为 **0**，而本计划声明不改动旧仓配置 —— 仅在宿主机放锁文件不产生互斥，旧 workflow 直接绕过。且 GitHub 的 disable 只阻止后续触发，不影响在途 run | Global Constraints 中 `flock` 范围诚实收窄为「新仓 orchestrator + 宿主机手工操作」，并新增「disable 不等于停止」一条；Task 2 runbook 第 2 步改为**显式 drain 门禁**（查询三仓 queued/in-progress → 等待或安全取消 → 归零并二次确认），每次 enable/disable 切换前重跑。**两方案中选择方案 2（严格 drain）**，理由与可推翻性写入 runbook |
| P1-18 | 新仓 branch protection 需要 bootstrap 顺序 | **成立。** 实测两个旧仓的 required check 同名 `PR container validation`；新仓此时未产生过该 check，而 GitHub 要求 required check 近 7 天内在目标仓成功运行过 | Task 0 第 3 步改为只复制非 check 类保护，required check 暂不设置；Task 1 确定恒定 gate 的精确 display name 并在成功运行一次后设为 required；Task 4 仅在 job 名确实变化时迁移 required context 并验证旧 context 已移除 |
| P1-19 | `changes` 与 gate 的行为契约需具体化 | **成立。** 原计划只写了结构，未定义语义 | Task 1 新增第 6 步契约：`changes` 用 checkout 后完整 Git diff 不依赖 300-file 过滤；分别定义五种触发情形的 base/head；gate 用 `if: always()`；gate 须验证「应运行的 app job 成功」而非 skipped 即放行；UI job 不得因 `needs: deploy-api` 的 skipped 被级联跳过。验收增加两个反例用例 |
| P1-20 | per-app 台账缺权威存储与原子更新语义 | **成立。** 原计划只写"新增台账"四字 | Task 4 第 3 步新增语义表：权威来源为运行容器 immutable image ref + image ID，台账仅为可恢复投影；存于 checkout 树外；发布验证通过后在同一把 `flock` 内原子替换；定义失败/自动回滚、手动回滚、宿主机丢失三种恢复路径；明确禁止把部署时变化的 manifest commit 回 master 以免递归触发发布 |
