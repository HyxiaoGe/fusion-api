# Fusion 生产压测工具

`runner.py` 提供公共 HTTP 与真实认证 SSE 两类阶梯压测。它不会读取浏览器存储；一次性邮箱、密码、访问令牌和刷新令牌只保留在进程内存。JSON 结果只暴露压测指标、`run_id`、邮箱指纹，以及清理孤立数据所需的非凭据 Agent run/trace ID。

## 安全边界

- 生产域名必须显式传入 `--confirm-production`。
- 账密注册/登录只允许性能工具在受控环境调用；实际认证运行前必须由密钥管理器向进程环境注入至少 32 字符、无首尾空白的 `FUSION_PERF_INTERNAL_AUTH_TOKEN`。该 secret 没有 CLI 参数，缺失或无效时会在 `/auth/register`、`/auth/login` 请求发出前 fail closed；携带该头的请求拒绝任何自动重定向，secret 也不会写入结果、repr 或进度日志。
- HTTP 默认阶梯为 `1,5,10,25,50`，SSE 默认阶梯为 `1,3,5`。
- HTTP 在至少 20 个样本后，错误率或超时率达到 5% 会停止；连续失败达到 10 次也会停止。
- SSE 任意 flow 失败或收到 error frame 后，不再提升并发。
- 每个 SSE flow 在发请求前登记唯一 `conversation_id`，`finally` 中按 ID 精确删除，并吊销注册和 Fusion 登录产生的两个刷新令牌。
- auth-service 暂无删除账号的公开接口，一次性账号会保留。可用结果中的 `run_id` 精确重建账号邮箱：`fusion-perf+<run_id>@seanfield.org`，由运维在压测后删除；日志和结果不会输出完整邮箱。
- runner 会从 `agent_event.run_started` 提取非凭据的 `agent_run_ids` / `agent_trace_ids`。error frame 只计数，不保存或输出 frame 内容。

## 压测后数据清理

runner 会先通过 Fusion API 按清单删除 conversation；当前迁移已为新的 `agent_steps.trace_id` 写入增加 `NOT VALID` 级联外键，因此删除会话会级联清理本轮新建的 `agent_sessions`、`agent_steps` 与 progress snapshot，不再需要手工按 trace ID 删除。

auth-service 暂无公开删号接口。runner 吊销全部 refresh token 后，仍必须由运维按 `run_id` 重建唯一邮箱 `fusion-perf+<run_id>@seanfield.org`，在 Fusion/auth 两库中先查询确认，再用邮箱与用户 ID 双条件精确删除一次性用户、refresh token 和 login log。禁止使用宽泛的 `LIKE 'fusion-perf%'` 批量删除。最后复查本轮用户、token、登录记录、conversation、Agent run/step 均为 0，并对比测试前后的全局计数。

## 使用

仓库 `.env.example` 中的 `fusion-client` 已不是生产实际应用 ID。runner 默认使用当前公开生产应用 ID，也可通过 `FUSION_PERF_CLIENT_ID` 或 `--client-id` 覆盖：

先由受控作业平台或密钥管理器向进程注入内部认证 token，禁止把 token 明文写进命令行、脚本、结果文件或 shell trace。启动前只验证变量存在，不要输出变量值：

```bash
python -c 'import os, sys; v = os.environ.get("FUSION_PERF_INTERNAL_AUTH_TOKEN", ""); sys.exit(len(v) < 32 or v != v.strip())'
```

```bash
python -m scripts.perf.runner \
  --mode all \
  --model-id '<低成本模型 ID>' \
  --confirm-production
```

只跑无认证 HTTP 基线：

```bash
python -m scripts.perf.runner --mode http --confirm-production
```

退出码 `2` 表示触发硬停止门禁、清理不完整或参数错误。不要把 stdout/stderr 重定向到包含环境变量或 shell trace 的日志中。

## L1-L4 完整生产流程

`full_runner.py` 覆盖登录/模型/会话读链路、短/长 SSE、断线恢复、停止生成和 30 分钟稳态。生产运行强制使用审查后的并发上限、1800 秒 soak 和 Prometheus 资源硬门禁。Prometheus 仅绑定在生产主机 loopback 时，可先建立本地端口转发：

```bash
ssh -N -L 19999:127.0.0.1:9999 dev
```

然后执行：

```bash
.venv311/bin/python -m scripts.perf.full_runner \
  --model-id deepseek-chat \
  --prometheus-url http://127.0.0.1:19999 \
  --confirm-production \
  --output docs/performance/YYYY-MM-DD-full-production-run-import.json
```

完整 runner 在每个阶梯后检查容器重启/OOM、Redis rejected/evicted、API 内存、PostgreSQL 连接和主机可用内存；监控不可用时 fail closed。L4 每个 tick 同样执行硬门禁。结果会先经过管理员压测导入 schema 校验，只保留聚合指标；账号、token、conversation/message ID、游标和模型正文不会写入文件。

最终窗口资源报告可通过 `scripts.perf.prometheus_report` 从 Prometheus `query_range` 生成；查询窗口最多 2 小时、最多 1000 个点、单响应最多 2 MiB。
