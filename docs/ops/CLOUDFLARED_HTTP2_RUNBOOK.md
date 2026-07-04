# cloudflared http2 持久化操作手册

## 背景

上传/预览硬化 v1.1 定位到公网路径存在固定慢请求和间歇性不稳定：

- `https://fusion.seanfield.org/api/files/upload` 的空鉴权探针约 1.5s，后端直连约 8ms。
- 图片内容接口后端 `x-process-time` 只有毫秒级，但公网下载经 Cloudflare Tunnel 明显变慢。
- dev 主机 `cloudflared` 当前走默认 QUIC，DNS 解析到 `198.18.x.x` fake-ip 后日志出现 `timeout: no recent network activity`。

临时验证过 `cloudflared --protocol http2 ... tunnel run` 能注册连接；把同等配置写入 config 后，`cloudflared tunnel ingress validate` 通过。

## 持久化步骤

在 dev 主机执行：

```bash
sudo cp /etc/cloudflared/config.yml /etc/cloudflared/config.yml.bak-$(date +%Y%m%d%H%M%S)

sudo awk 'BEGIN{done=0} /^ingress:/{if(!done){print "protocol: http2"; done=1}} {print}' \
  /etc/cloudflared/config.yml | sudo tee /etc/cloudflared/config.yml.tmp >/dev/null

sudo mv /etc/cloudflared/config.yml.tmp /etc/cloudflared/config.yml
cloudflared --config /etc/cloudflared/config.yml tunnel ingress validate
sudo systemctl restart cloudflared
```

## 验证命令

```bash
systemctl is-active cloudflared
journalctl -u cloudflared --since "5 minutes ago" --no-pager | tail -n 120

curl -sS -w '\nHTTP=%{http_code} total=%{time_total} starttransfer=%{time_starttransfer} appconnect=%{time_appconnect}\n' \
  -o /tmp/fusion-upload-probe.out \
  -X POST https://fusion.seanfield.org/api/files/upload \
  -F provider=test \
  -F model=test \
  -F conversation_id=test
```

预期：

- `systemctl is-active cloudflared` 返回 `active`。
- 最近日志出现 `protocol=http2` 的注册连接，不再持续刷 QUIC timeout。
- 未登录上传探针返回 JSON `401`，不是 Cloudflare `530 / 1033`。
- 公网基础耗时应明显低于故障期的 1.5s 级固定往返；如果仍慢，需要继续排查 Cloudflare、出口网络或 DNS。
