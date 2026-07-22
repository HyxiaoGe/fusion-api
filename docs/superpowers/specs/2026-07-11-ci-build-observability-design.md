# fusion-api CI 构建可观测性设计

## 目标

在不改变生产镜像、测试内容、镜像推送和部署行为的前提下，为 Windows 自托管 Runner 的构建任务增加阶段耗时、失败分类、Job Summary 和失败日志 artifact，使偶发失败可以直接从 GitHub Actions 页面定位。

## 方案选择

采用“单个 PowerShell 编排步骤 + always 汇总步骤”。继续只构建一次 production 镜像，并在同一个临时容器内执行 CI 检查；PowerShell 负责按阶段执行、实时输出、写日志和保存机器可读结果。

不把每个检查拆成独立 GitHub Actions step，避免重复启动容器或引入容器生命周期管理；不只记录总耗时，因为总耗时无法区分依赖安装、架构检查、Ruff 和单测失败。

## 阶段模型

构建任务记录以下阶段：

1. `docker-build`：构建 production 镜像。
2. `ci-dependencies`：在临时容器中安装 `requirements-ci.txt`。
3. `architecture`：执行 `scripts/check_architecture.py`。
4. `ruff`：执行 `ruff check .`。
5. `unit-tests`：执行 unittest discovery。
6. `image-push`：推送 ACR 镜像。

每个阶段保存名称、展示名、开始 UTC 时间、结束 UTC 时间、耗时秒数、状态和退出码。阶段状态只有 `success`、`failure` 和 `skipped`。

## 执行与日志

新增仓库脚本 `scripts/ci/run_windows_container_ci.ps1`，负责前五个阶段。脚本创建 `_ci-logs`，每个阶段使用独立 UTF-8 日志文件，并将输出实时回显到控制台。

临时容器保持单次启动，但其 shell 命令按阶段分别执行。容器名称包含 GitHub run id，`finally` 中始终删除容器。任一阶段失败后，后续检查阶段标记为 skipped，脚本保存结果后返回原始非零退出码。

镜像推送保留为 workflow step；该 step 记录自己的阶段结果，以便区分 ACR 登录和推送问题。敏感参数只通过现有 secrets/action 传递，日志不打印凭据。

## Job Summary 与 artifact

新增 `if: always()` 汇总步骤，读取阶段结果并写入 `$GITHUB_STEP_SUMMARY`，内容包括：

- Runner 名称与镜像标签。
- 每个阶段的状态、耗时和退出码。
- 首个失败阶段及对应日志文件。
- 总耗时。

新增 `actions/upload-artifact`：仅 build job 失败时上传 `_ci-logs`，artifact 名称包含 run id，保留 7 天。成功任务不上传日志，避免 artifact 长期占用。

## 错误处理

- Docker build 失败：分类为 `docker-build`。
- CI 依赖安装失败或超时：分类为 `ci-dependencies`。
- 架构、Ruff、单测失败：分别归类。
- ACR 登录失败：保留现有 step 失败信息；summary 标记推送前失败。
- Docker push 失败：分类为 `image-push`。
- 汇总或 artifact 上传不得覆盖原始 Job 失败结果。
- Cleanup 使用 `if: always()`，继续清理本地临时容器和当前镜像。

## 测试

扩展 `test/test_ci_container_contract.py`，验证：

- workflow 调用新的 Windows CI 脚本。
- 五个核心阶段名称稳定存在。
- summary 使用 `if: always()`。
- 失败时上传 `_ci-logs`，保留 7 天。
- production 镜像仍只构建一次并推送同一标签。
- 脚本包含 finally 容器清理、UTF-8 日志和原始退出码传播。

本地验证运行相关契约测试和 Ruff；不启动本地 Fusion 服务。推送分支后观察真实 Windows Runner，确认成功路径耗时没有明显回退，并检查 Job Summary。失败 artifact 的上传条件由契约测试覆盖，不通过故意破坏 CI 来验证。

## 非目标

- 不修改 Dockerfile、依赖版本或部署 job。
- 不改变测试集合和超时上限。
- 不引入外部日志或监控服务。
- 不处理 fusion-ui；它在本 PR 验证后使用独立设计与 PR。
