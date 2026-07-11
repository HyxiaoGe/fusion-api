# 2026-07-11 生产环境性能基线

## 结论

- `fusion-api` 当前受 **1 vCPU / 单 Uvicorn worker** 限制；公共只读 `/api/models/` 在 API 直连链路约 **143–146 RPS** 后吞吐不再增长，50 并发时 API CPU 约 100%，数据库与 Redis 仍有充足余量。
- nginx → Next.js rewrite → API 的完整源站链在 25 并发下为 **129.26 RPS**，P95 **274 ms**，500/500 成功；代理层没有先于 API 饱和。
- 公网 HTTP/2 复用链路在 1→3→5→10 VU 阶梯下 86/86 成功，但 P95 **2.38 s**、最大 **3.13 s**，同时源站资源接近空闲。主要延迟来自 Cloudflare/LAX、公网和隧道路径，不是 API 计算或数据库。
- 真实认证 SSE 在 `deepseek-chat`、最大 32 tokens、1→3→5 并发下共 **9/9 成功**、0 error frame；5 并发 P95 TTFT **1.44 s**、P95 完成 **1.47 s**。
- SSE 阶梯峰值约为：API CPU **18.6%**、API 内存 **238 MiB / 1 GiB**、PostgreSQL CPU **7.1%**；所有关键容器 0 重启、0 OOM，Redis 0 rejected connection、0 eviction。

## 环境边界

本轮直接使用唯一生产环境 `https://fusion.seanfield.org`。主机为 8 核、11 GiB 内存；关键限制如下：

| 组件 | CPU 限制 | 内存限制 | 运行形态 |
|---|---:|---:|---|
| fusion-api | 1 核 | 1 GiB | 1 Uvicorn worker |
| fusion-ui | 未单独限核 | 256 MiB | Next.js 代理 `/api/*` |
| nginx | 未单独限核 | 128 MiB | Cloudflare Tunnel 后的入口 |
| PostgreSQL | 未单独限核 | 1 GiB | Fusion / Auth 等共享实例 |
| Redis | 未单独限核 | 256 MiB | 流状态与缓存 |

## 分层结果

### 公共只读接口 `/api/models/`

| 链路 | 请求与并发 | 成功率 | 吞吐 | P50 | P95 | 最大值 |
|---|---:|---:|---:|---:|---:|---:|
| API 直连 | 200 / c10 | 100% | 145.67 RPS | 66 ms | 78 ms | 86 ms |
| API 直连 | 500 / c25 | 100% | 145.61 RPS | 167 ms | 255 ms | 269 ms |
| API 直连 | 1000 / c50 | 100% | 142.63 RPS | 322 ms | 453 ms | 474 ms |
| 完整源站链 | 200 / c10 | 100% | 124.26 RPS | 77 ms | 95 ms | 101 ms |
| 完整源站链 | 500 / c25 | 100% | 129.26 RPS | 187 ms | 274 ms | 289 ms |
| 公网 HTTP/2 | 1→3→5→10 VU | 100%（86/86） | 1.87 iter/s | 691 ms | 2.38 s | 3.13 s |

公网阶梯按每个 VU 请求后等待 1 秒执行，因此 `iter/s` 不是容量上限；该阶段的目的在于观察浏览器式连接复用下的用户端延迟。P95 超过 2 秒软门禁后没有继续升压。

### 真实认证 SSE

| 并发 | flows | 成功 | P50 TTFT | P95 TTFT | P95 完成 | error frame |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 2.39 s | 2.39 s | 4.37 s | 0 |
| 3 | 3 | 3 | 1.58 s | 1.58 s | 1.58 s | 0 |
| 5 | 5 | 5 | 1.44 s | 1.44 s | 1.47 s | 0 |

首个单流包含认证/JWKS/userinfo、模型和连接冷启动成本，因此明显慢于后续热路径。该结果是小输出真实链路基线，不代表长回答、联网工具、多轮上下文或大量新用户同时冷鉴权的性能。

## 数据清理与安全门禁

- runner 内部生成一次性账号；密码、access token、refresh token 只存在进程内存，不写结果、不进入命令行。
- 注册后使用生产 Fusion client ID 再次登录，确保 JWT audience 与生产 API 一致。
- 9 个会话全部按精确 ID 经公开 API 删除；注册与登录产生的 2 枚 refresh token 全部吊销。
- API 删除无法级联的 9 条 `agent_steps` 已按精确 trace ID 删除；对应 `agent_sessions` 复查为 0。
- Fusion 与 auth-service 的一次性用户各删除 1 条，auth 登录日志删除 2 条；两库 `fusion-perf+...` 账号复查均为 0。
- 清理后全局计数恢复到测试前：`conversations=996`、`messages=2183`、`agent_sessions=905`、`agent_steps=1559`。
- 所有关键容器重启数为 0、OOM 为 false；Redis `rejected_connections=0`、`evicted_keys=0`。

## 当前性能判断

1. **主要用户体验风险在公网路径抖动。** 公网 P95 已超过 2 秒，而源站相同接口 P95 小于 300 ms；优先继续观察 Cloudflare Tunnel 路由和地域，而不是先扩数据库。
2. **API 的明确容量边界是单核约 145 RPS。** 对当前低流量足够，但突发公共读取会先撞 API CPU；提高 worker 数前必须同时调整 1 核 cgroup 限制，并验证连接池与内存。
3. **小规模流式对话余量健康。** 5 个并发短回答没有出现流错误，资源峰值远低于门禁；长输出、联网工具和恢复/停止应作为单独场景，不与这份短回答基线混为一谈。

## 复测入口

仓库内 `scripts/perf/runner.py` 提供生产显式确认、HTTP/SSE 阶梯、一次性认证、脱敏结果和精确清理清单。使用与清理要求见 `scripts/perf/README.md`。
