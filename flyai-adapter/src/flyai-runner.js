import { spawn } from "node:child_process";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { TextDecoder } from "node:util";
import { fileURLToPath } from "node:url";

import { parseStrictJson } from "./strict-json.js";

const ALLOWED_COMMANDS = new Set(["search-flight", "search-train"]);
const DEFAULT_BUNDLE_PATH = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
  "node_modules",
  "@fly-ai",
  "flyai-cli",
  "dist",
  "flyai-bundle.cjs",
);

export class FlyaiRunnerError extends Error {
  constructor(code) {
    super(code);
    this.name = "FlyaiRunnerError";
    this.code = code;
  }
}

function safeError(code) {
  return new FlyaiRunnerError(code);
}

function killProcessGroup(child, signal) {
  if (!child.pid) return;
  try {
    if (process.platform === "win32") child.kill(signal);
    else process.kill(-child.pid, signal);
  } catch (error) {
    if (error?.code === "ESRCH") return;
    try {
      child.kill(signal);
    } catch {
      // 进程已经退出。
    }
  }
}

function captureBounded(stream, maxBytes, onLimit) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    let exceeded = false;
    stream.on("data", (chunk) => {
      size += chunk.length;
      if (size > maxBytes) {
        if (!exceeded) {
          exceeded = true;
          onLimit();
        }
        return;
      }
      chunks.push(chunk);
    });
    stream.once("error", reject);
    stream.once("end", () => resolve(Buffer.concat(chunks)));
  });
}

function waitForExit(child) {
  return new Promise((resolve) => {
    let settled = false;
    child.once("error", (error) => {
      if (settled) return;
      settled = true;
      resolve({ code: null, signal: null, spawnError: error });
    });
    child.once("close", (code, signal) => {
      if (settled) return;
      settled = true;
      resolve({ code, signal, spawnError: null });
    });
  });
}

function validateInvocation({ command, argv, apiKey, homeDir, executablePath, bundlePath }) {
  if (!ALLOWED_COMMANDS.has(command) || !Array.isArray(argv) || !argv.every((item) => typeof item === "string")) {
    throw safeError("invalid_invocation");
  }
  if (!path.isAbsolute(executablePath) || !path.isAbsolute(bundlePath) || !path.isAbsolute(homeDir)) {
    throw safeError("invalid_invocation");
  }
  if (typeof apiKey !== "string" || !apiKey.trim()) throw safeError("credential_unavailable");
}

export async function runFlyaiCli({
  command,
  argv,
  apiKey,
  homeDir = "/tmp",
  executablePath = process.execPath,
  bundlePath = DEFAULT_BUNDLE_PATH,
  timeoutMs = 20_000,
  maxStdoutBytes = 256 * 1024,
  maxStderrBytes = 16 * 1024,
  signal,
}) {
  validateInvocation({ command, argv, apiKey, homeDir, executablePath, bundlePath });
  if (signal?.aborted) throw safeError("request_aborted");

  await mkdir(homeDir, { recursive: true, mode: 0o700 });
  const callHome = await mkdtemp(path.join(homeDir, "flyai-call-"));
  try {
    const child = spawn(executablePath, [bundlePath, command, ...argv], {
      cwd: callHome,
      detached: process.platform !== "win32",
      env: {
        FLYAI_API_KEY: apiKey.trim(),
        HOME: callHome,
        LANG: "C.UTF-8",
        LC_ALL: "C.UTF-8",
        TZ: "Asia/Shanghai",
        TMPDIR: callHome,
      },
      shell: false,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });

    let failureCode = null;
    let killTimer = null;
    let killPromise = null;
    const terminate = (code) => {
      if (failureCode !== null) return;
      failureCode = code;
      killProcessGroup(child, "SIGTERM");
      killPromise = new Promise((resolve) => {
        killTimer = setTimeout(() => {
          killProcessGroup(child, "SIGKILL");
          resolve();
        }, 1_000);
      });
    };

    const abort = () => terminate("request_aborted");
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();

    const timeout = setTimeout(() => terminate("provider_timeout"), timeoutMs);
    const exitPromise = waitForExit(child);
    const stdoutPromise = captureBounded(child.stdout, maxStdoutBytes, () =>
      terminate("provider_output_too_large"),
    );
    const stderrPromise = captureBounded(child.stderr, maxStderrBytes, () =>
      terminate("provider_output_too_large"),
    );

    let exit;
    let stdout;
    try {
      [exit, stdout] = await Promise.all([exitPromise, stdoutPromise, stderrPromise]);
    } catch {
      terminate("provider_error");
      await Promise.allSettled([exitPromise, stdoutPromise, stderrPromise]);
      if (killPromise) await killPromise;
      if (killTimer !== null) clearTimeout(killTimer);
      throw safeError(failureCode ?? "provider_error");
    } finally {
      clearTimeout(timeout);
      signal?.removeEventListener("abort", abort);
    }

    if (killPromise) await killPromise;
    if (killTimer !== null) clearTimeout(killTimer);
    if (failureCode !== null) throw safeError(failureCode);
    if (exit.spawnError || exit.code !== 0 || exit.signal !== null) throw safeError("provider_error");

    let text;
    try {
      text = new TextDecoder("utf-8", { fatal: true }).decode(stdout);
    } catch {
      throw safeError("invalid_provider_response");
    }
    try {
      return parseStrictJson(text, { maxDepth: 20, maxNodes: 10_000, maxStringChars: 64 * 1024 });
    } catch {
      throw safeError("invalid_provider_response");
    }
  } finally {
    await rm(callHome, { recursive: true, force: true });
  }
}
