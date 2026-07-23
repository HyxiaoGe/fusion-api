import { createHash, timingSafeEqual } from "node:crypto";
import http from "node:http";
import path from "node:path";
import { TextDecoder } from "node:util";

import { runFlyaiCli } from "./flyai-runner.js";
import { projectProviderResponse } from "./projector.js";
import { buildCliInvocation, validateSearchRequest, validateUserScope } from "./schema.js";
import { parseStrictJson } from "./strict-json.js";

const MAX_REQUEST_BYTES = 16 * 1024;
const ERROR_STATUS = Object.freeze({
  unauthorized: 401,
  not_found: 404,
  method_not_allowed: 405,
  invalid_request: 400,
  concurrency_limited: 429,
  provider_timeout: 504,
  provider_output_too_large: 502,
  invalid_provider_response: 502,
  provider_error: 502,
  credential_unavailable: 503,
  request_aborted: 499,
  internal_error: 500,
});

class ConcurrencyGate {
  constructor(globalLimit = 2) {
    this.globalLimit = globalLimit;
    this.activeGlobal = 0;
    this.activeUsers = new Set();
  }

  tryAcquire(userScope) {
    if (this.activeUsers.has(userScope) || this.activeGlobal >= this.globalLimit) return false;
    this.activeGlobal += 1;
    this.activeUsers.add(userScope);
    return true;
  }

  release(userScope) {
    if (!this.activeUsers.delete(userScope)) return;
    this.activeGlobal = Math.max(0, this.activeGlobal - 1);
  }
}

function secretDigest(value) {
  return createHash("sha256").update(value).digest();
}

function isAuthorized(header, token) {
  if (typeof header !== "string" || !header.startsWith("Bearer ")) return false;
  const presented = header.slice("Bearer ".length);
  if (!presented) return false;
  return timingSafeEqual(secretDigest(presented), secretDigest(token));
}

function sendJson(response, statusCode, payload) {
  if (response.destroyed || response.writableEnded) return;
  const body = JSON.stringify(payload);
  response.writeHead(statusCode, {
    "cache-control": "no-store",
    "content-length": Buffer.byteLength(body),
    "content-type": "application/json; charset=utf-8",
    "x-content-type-options": "nosniff",
  });
  response.end(body);
}

function sendError(response, code) {
  const safeCode = code in ERROR_STATUS ? code : "internal_error";
  sendJson(response, ERROR_STATUS[safeCode], { error_code: safeCode });
}

async function readRequestBody(request) {
  const contentLength = Number(request.headers["content-length"] ?? 0);
  if (!Number.isFinite(contentLength) || contentLength < 0 || contentLength > MAX_REQUEST_BYTES) {
    throw new Error("invalid_request");
  }
  const chunks = [];
  let size = 0;
  let exceeded = false;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > MAX_REQUEST_BYTES) {
      exceeded = true;
      continue;
    }
    chunks.push(chunk);
  }
  if (exceeded) throw new Error("invalid_request");
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks));
  } catch {
    throw new Error("invalid_request");
  }
  try {
    return parseStrictJson(text, { maxDepth: 12, maxNodes: 2_000, maxStringChars: 2_048 });
  } catch {
    throw new Error("invalid_request");
  }
}

function resolveRoute(pathname) {
  if (pathname === "/v1/search/flights") return "flight";
  if (pathname === "/v1/search/trains") return "train";
  return null;
}

export function createAdapterServer({
  token,
  apiKey,
  runner,
  runnerOptions = {},
  clock = () => new Date(),
  homeDir = "/tmp",
} = {}) {
  if (typeof token !== "string" || !token.trim()) {
    throw new Error("FLYAI_ADAPTER_TOKEN 必须配置");
  }
  if (typeof apiKey !== "string" || !apiKey.trim()) {
    throw new Error("FLYAI_API_KEY 必须配置");
  }
  if (!path.isAbsolute(homeDir)) {
    throw new Error("FlyAI HOME 必须是绝对路径");
  }

  const gate = new ConcurrencyGate(2);
  const execute =
    runner ??
    (({ command, argv, signal }) =>
      runFlyaiCli({
        ...runnerOptions,
        command,
        argv,
        signal,
        apiKey,
        homeDir,
      }));

  const server = http.createServer(async (request, response) => {
    let parsedUrl;
    try {
      parsedUrl = new URL(request.url ?? "/", "http://adapter.internal");
    } catch {
      sendError(response, "not_found");
      return;
    }

    if (parsedUrl.pathname === "/health") {
      if (request.method !== "GET") {
        sendError(response, "method_not_allowed");
        return;
      }
      if (parsedUrl.search) {
        sendError(response, "invalid_request");
        return;
      }
      sendJson(response, 200, { status: "healthy" });
      return;
    }

    const kind = resolveRoute(parsedUrl.pathname);
    if (kind === null) {
      sendError(response, "not_found");
      return;
    }
    if (request.method !== "POST") {
      sendError(response, "method_not_allowed");
      return;
    }
    if (parsedUrl.search || !isAuthorized(request.headers.authorization, token)) {
      sendError(response, parsedUrl.search ? "invalid_request" : "unauthorized");
      return;
    }
    if (!(request.headers["content-type"] ?? "").toLowerCase().startsWith("application/json")) {
      sendError(response, "invalid_request");
      return;
    }

    let userScope;
    let normalizedRequest;
    try {
      userScope = validateUserScope(request.headers["x-fusion-user-scope"]);
      normalizedRequest = validateSearchRequest(await readRequestBody(request), kind);
    } catch {
      sendError(response, "invalid_request");
      return;
    }

    if (!gate.tryAcquire(userScope)) {
      sendError(response, "concurrency_limited");
      return;
    }
    const controller = new AbortController();
    const cancel = () => controller.abort();
    const cancelOnClose = () => {
      if (!response.writableEnded) cancel();
    };
    request.once("aborted", cancel);
    response.once("close", cancelOnClose);
    if (request.aborted || response.destroyed) cancel();
    try {
      const invocation = buildCliInvocation(normalizedRequest, kind);
      const providerPayload = await execute({ ...invocation, signal: controller.signal });
      const items = projectProviderResponse(providerPayload, normalizedRequest, kind);
      sendJson(response, 200, {
        observed_at: clock().toISOString(),
        request: normalizedRequest,
        items,
      });
    } catch (error) {
      sendError(response, error?.code ?? "provider_error");
    } finally {
      request.off("aborted", cancel);
      response.off("close", cancelOnClose);
      gate.release(userScope);
      // HTTP 客户端先感知 abort，服务端仍需等待 CLI 进程组和临时目录清理。
      // 该事件只表示服务端本次业务请求已完全收尾，不携带任何查询或凭证数据。
      server.emit("flyai-request-settled");
    }
  });
  return server;
}
