import process from "node:process";

import { createAdapterServer } from "./server.js";

function requiredEnvironment(name) {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name} 必须配置`);
  return value;
}

try {
  const server = createAdapterServer({
    token: requiredEnvironment("FLYAI_ADAPTER_TOKEN"),
    apiKey: requiredEnvironment("FLYAI_API_KEY"),
    homeDir: "/tmp/flyai-home",
  });
  const port = Number(process.env.PORT ?? "8080");
  if (!Number.isInteger(port) || port < 1 || port > 65_535) throw new Error("PORT 无效");
  const host = process.env.HOST?.trim() || "0.0.0.0";
  if (!new Set(["0.0.0.0", "127.0.0.1", "::1"]).has(host)) throw new Error("HOST 无效");
  server.listen(port, host, () => {
    process.stdout.write(`flyai-adapter 已启动 host=${host} port=${port}\n`);
  });
} catch {
  process.stderr.write("flyai-adapter 启动失败\n");
  process.exitCode = 1;
}
