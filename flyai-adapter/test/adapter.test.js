import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import test from "node:test";

import { runFlyaiCli } from "../src/flyai-runner.js";
import { createAdapterServer } from "../src/server.js";
import { parseStrictJson } from "../src/strict-json.js";

const TOKEN = "adapter-test-token";
const API_KEY = "flyai-test-key";

async function listen(server) {
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  return `http://127.0.0.1:${address.port}`;
}

async function close(server) {
  server.close();
  await once(server, "close");
}

function providerEnvelope(itemList = [], overrides = {}) {
  return {
    status: 0,
    message: "success",
    data: { itemList },
    systemMessage: "",
    ...overrides,
  };
}

function rawFlight({ item = {}, journey = {}, segment = {} } = {}) {
  return {
    totalDuration: 135,
    ticketPrice: 880,
    jumpUrl: "https://a.feizhu.com/flight/CZ1234",
    journeys: [
      {
        journeyType: "直达",
        segments: [
          {
            depCityName: "深圳市",
            depStationName: "深圳宝安国际机场",
            depStationCode: "SZX",
            depTerm: "T3",
            depDateTime: "2026-08-02 08:30:00",
            arrCityName: "上海市",
            arrStationName: "上海虹桥国际机场",
            arrStationCode: "SHA",
            arrTerm: "T2",
            arrDateTime: "2026-08-02 10:45:00",
            marketingTransportName: "南方航空",
            marketingTransportNo: "CZ1234",
            seatClassName: "经济舱",
            ...segment,
          },
        ],
        ...journey,
      },
    ],
    ...item,
  };
}

function rawTrain({ item = {}, journey = {}, segment = {} } = {}) {
  return {
    totalDuration: 32,
    price: 74.5,
    jumpUrl: "https://a.feizhu.com/train/G100",
    journeys: [
      {
        journeyType: "直达",
        segments: [
          {
            depCityName: "深圳市",
            depStationName: "深圳北站",
            depStationCode: "IOQ",
            depDateTime: "2026-08-02 09:00:00",
            arrCityName: "广州市",
            arrStationName: "广州南站",
            arrStationCode: "IZQ",
            arrDateTime: "2026-08-02 09:32:00",
            marketingTransportName: "中国铁路",
            marketingTransportNo: "G100",
            seatClassName: "二等座",
            ...segment,
          },
        ],
        ...journey,
      },
    ],
    ...item,
  };
}

function createRunner({
  flightPayload = providerEnvelope([rawFlight()]),
  trainPayload = providerEnvelope([rawTrain()]),
  delayMs = 0,
} = {}) {
  const calls = [];
  let active = 0;
  let maxActive = 0;
  const runner = async ({ command, argv, signal }) => {
    calls.push({ command, argv: [...argv] });
    active += 1;
    maxActive = Math.max(maxActive, active);
    try {
      if (delayMs) {
        await new Promise((resolve, reject) => {
          const timer = setTimeout(resolve, delayMs);
          signal?.addEventListener(
            "abort",
            () => {
              clearTimeout(timer);
              const error = new Error("request_aborted");
              error.code = "request_aborted";
              reject(error);
            },
            { once: true },
          );
        });
      }
      return command === "search-flight" ? flightPayload : trainPayload;
    } finally {
      active -= 1;
    }
  };
  return { runner, calls, getMaxActive: () => maxActive };
}

async function post(baseUrl, pathname, body, { token = TOKEN, userScope = "user-1", signal } = {}) {
  return fetch(`${baseUrl}${pathname}`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${token}`,
      "content-type": "application/json",
      "x-fusion-user-scope": userScope,
    },
    body: JSON.stringify(body),
    signal,
  });
}

function processExists(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error?.code !== "ESRCH";
  }
}

async function waitForFile(filePath, timeoutMs = 1_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      return JSON.parse(await readFile(filePath, "utf8"));
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
  }
  throw new Error(`等待文件超时：${filePath}`);
}

async function waitForProcessesGone(pids, timeoutMs = 800) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (pids.every((pid) => !processExists(pid))) return;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  assert.deepEqual(pids.filter(processExists), []);
}

test("启动配置缺少密钥时立即失败", () => {
  assert.throws(() => createAdapterServer({ token: "", apiKey: API_KEY }), /FLYAI_ADAPTER_TOKEN/);
  assert.throws(() => createAdapterServer({ token: TOKEN, apiKey: "" }), /FLYAI_API_KEY/);
});

test("健康检查不触发供应商调用，业务端点要求 Bearer", async () => {
  const { runner, calls } = createRunner();
  const server = createAdapterServer({ token: TOKEN, apiKey: API_KEY, runner });
  const baseUrl = await listen(server);
  try {
    const health = await fetch(`${baseUrl}/health`);
    assert.equal(health.status, 200);
    assert.deepEqual(await health.json(), { status: "healthy" });
    assert.equal(calls.length, 0);

    const unauthorized = await post(
      baseUrl,
      "/v1/search/flights",
      { origin: "深圳", destination: "上海", departure_date: "2026-08-02" },
      { token: "wrong-token" },
    );
    assert.equal(unauthorized.status, 401);
    assert.deepEqual(await unauthorized.json(), { error_code: "unauthorized" });
    assert.equal(calls.length, 0);
  } finally {
    await close(server);
  }
});

test("真实航班 itemList 使用固定 argv 并只投影安全字段", async () => {
  const { runner, calls } = createRunner({
    flightPayload: providerEnvelope([
      rawFlight(),
      rawFlight(),
      rawFlight({
        item: { ticketPrice: 900, jumpUrl: "https://evil.example/steal" },
        segment: {
          marketingTransportName: undefined,
          marketingTransportNo: "MU5678",
          depDateTime: "2026-08-02T11:00:00+08:00",
          arrDateTime: "2026-08-02T13:20:00+08:00",
        },
      }),
    ]),
  });
  const server = createAdapterServer({
    token: TOKEN,
    apiKey: API_KEY,
    runner,
    clock: () => new Date("2026-07-22T12:00:00.000Z"),
  });
  const baseUrl = await listen(server);
  try {
    const response = await post(baseUrl, "/v1/search/flights", {
      origin: "深圳",
      destination: "上海",
      departure_date: "2026-08-02",
      cabin_class: "经济舱",
      max_price_yuan: 1000,
      departure_hour_start: 7,
      departure_hour_end: 12,
      sort_by: "price_asc",
      limit: 2,
    });
    assert.equal(response.status, 200);
    assert.deepEqual(calls, [
      {
        command: "search-flight",
        argv: [
          "--origin", "深圳",
          "--destination", "上海",
          "--dep-date", "2026-08-02",
          "--journey-type", "1",
          "--seat-class-name", "经济舱",
          "--max-price", "1000",
          "--dep-hour-start", "7",
          "--dep-hour-end", "12",
          "--sort-type", "3",
        ],
      },
    ]);
    assert.deepEqual(await response.json(), {
      observed_at: "2026-07-22T12:00:00.000Z",
      request: {
        origin: "深圳",
        destination: "上海",
        departure_date: "2026-08-02",
        cabin_class: "经济舱",
        max_price_yuan: 1000,
        departure_hour_start: 7,
        departure_hour_end: 12,
        sort_by: "price_asc",
        limit: 2,
      },
      items: [
        {
          transport_no: "CZ1234",
          operator_name: "南方航空",
          departure: {
            city: "深圳市",
            station_name: "深圳宝安国际机场",
            station_code: "SZX",
            terminal: "T3",
            scheduled_at: "2026-08-02T08:30:00+08:00",
          },
          arrival: {
            city: "上海市",
            station_name: "上海虹桥国际机场",
            station_code: "SHA",
            terminal: "T2",
            scheduled_at: "2026-08-02T10:45:00+08:00",
          },
          duration_minutes: 135,
          travel_class: "经济舱",
          journey_type: "direct",
          price: { currency: "CNY", amount_minor: 88000 },
          booking_url: "https://a.feizhu.com/flight/CZ1234",
        },
        {
          transport_no: "MU5678",
          departure: {
            city: "深圳市",
            station_name: "深圳宝安国际机场",
            station_code: "SZX",
            terminal: "T3",
            scheduled_at: "2026-08-02T11:00:00+08:00",
          },
          arrival: {
            city: "上海市",
            station_name: "上海虹桥国际机场",
            station_code: "SHA",
            terminal: "T2",
            scheduled_at: "2026-08-02T13:20:00+08:00",
          },
          duration_minutes: 135,
          travel_class: "经济舱",
          journey_type: "direct",
          price: { currency: "CNY", amount_minor: 90000 },
        },
      ],
    });
  } finally {
    await close(server);
  }
});

test("城市和车站请求均精确匹配，错地点、错日期和非直达结果被丢弃", async () => {
  const payload = providerEnvelope([
    rawTrain(),
    rawTrain({ segment: { marketingTransportNo: "G101", depCityName: "珠海市" } }),
    rawTrain({ segment: { marketingTransportNo: "G102", depStationName: "深圳站" } }),
    rawTrain({ segment: { marketingTransportNo: "G103", depDateTime: "2026-08-03 09:00:00" } }),
    rawTrain({
      journey: { journeyType: 2 },
      segment: { marketingTransportNo: "G104" },
    }),
  ]);
  const { runner } = createRunner({ trainPayload: payload });
  const server = createAdapterServer({ token: TOKEN, apiKey: API_KEY, runner });
  const baseUrl = await listen(server);
  try {
    const cityResponse = await post(baseUrl, "/v1/search/trains", {
      origin: "深圳市",
      destination: "广州市",
      departure_date: "2026-08-02",
      sort_by: "duration_asc",
    });
    assert.equal(cityResponse.status, 200);
    assert.deepEqual((await cityResponse.json()).items.map((item) => item.transport_no), ["G100", "G102"]);

    const stationResponse = await post(baseUrl, "/v1/search/trains", {
      origin: "深圳北站",
      destination: "广州南站",
      departure_date: "2026-08-02",
    });
    assert.equal(stationResponse.status, 200);
    assert.deepEqual((await stationResponse.json()).items.map((item) => item.transport_no), ["G100"]);

    const codeResponse = await post(baseUrl, "/v1/search/trains", {
      origin: "IOQ",
      destination: "IZQ",
      departure_date: "2026-08-02",
    });
    assert.equal(codeResponse.status, 200);
    assert.deepEqual((await codeResponse.json()).items.map((item) => item.transport_no), ["G100", "G102"]);
  } finally {
    await close(server);
  }
});

test("投影层强制价格、出发小时和舱等约束，并按价格本地排序", async () => {
  const { runner } = createRunner({
    flightPayload: providerEnvelope([
      rawFlight({
        item: { ticketPrice: 900 },
        segment: { marketingTransportNo: "CZ900", depDateTime: "2026-08-02 10:00:00" },
      }),
      rawFlight({
        item: { ticketPrice: 700 },
        segment: { marketingTransportNo: "CZ700", depDateTime: "2026-08-02 07:00:00" },
      }),
      rawFlight({
        item: { ticketPrice: undefined },
        segment: { marketingTransportNo: "CZ-NO-PRICE", depDateTime: "2026-08-02 08:00:00" },
      }),
      rawFlight({
        item: { ticketPrice: 600 },
        segment: { marketingTransportNo: "CZ-EARLY", depDateTime: "2026-08-02 06:59:59" },
      }),
      rawFlight({
        item: { ticketPrice: 500 },
        segment: {
          marketingTransportNo: "CZ-BUSINESS",
          depDateTime: "2026-08-02 08:30:00",
          seatClassName: "公务舱",
        },
      }),
      rawFlight({
        item: { ticketPrice: 1_001 },
        segment: { marketingTransportNo: "CZ-EXPENSIVE", depDateTime: "2026-08-02 09:00:00" },
      }),
      rawFlight({
        item: { ticketPrice: 800 },
        segment: {
          marketingTransportNo: "CZ-NO-CLASS",
          depDateTime: "2026-08-02 09:30:00",
          seatClassName: undefined,
        },
      }),
    ]),
  });
  const server = createAdapterServer({ token: TOKEN, apiKey: API_KEY, runner });
  const baseUrl = await listen(server);
  try {
    const response = await post(baseUrl, "/v1/search/flights", {
      origin: "深圳",
      destination: "上海",
      departure_date: "2026-08-02",
      cabin_class: "经济舱",
      max_price_yuan: 1000,
      departure_hour_start: 7,
      departure_hour_end: 10,
      sort_by: "price_asc",
      limit: 5,
    });
    assert.equal(response.status, 200);
    assert.deepEqual((await response.json()).items.map((item) => item.transport_no), ["CZ700", "CZ900"]);
  } finally {
    await close(server);
  }
});

test("座席约束拒绝缺失值，出发时间和时长排序在本地确定执行", async () => {
  const payload = providerEnvelope([
    rawTrain({
      item: { totalDuration: 60, price: 80 },
      segment: { marketingTransportNo: "G3", depDateTime: "2026-08-02 10:00:00" },
    }),
    rawTrain({
      item: { totalDuration: 40, price: 60 },
      segment: { marketingTransportNo: "G1", depDateTime: "2026-08-02 08:00:00" },
    }),
    rawTrain({
      item: { totalDuration: 30, price: 50 },
      segment: {
        marketingTransportNo: "G2",
        depDateTime: "2026-08-02 09:00:00",
        seatClassName: "一等座",
      },
    }),
    rawTrain({
      item: { totalDuration: 35, price: 55 },
      segment: {
        marketingTransportNo: "G4",
        depDateTime: "2026-08-02 07:00:00",
        seatClassName: undefined,
      },
    }),
  ]);
  const { runner } = createRunner({ trainPayload: payload });
  const server = createAdapterServer({ token: TOKEN, apiKey: API_KEY, runner });
  const baseUrl = await listen(server);
  try {
    const departureResponse = await post(baseUrl, "/v1/search/trains", {
      origin: "深圳",
      destination: "广州",
      departure_date: "2026-08-02",
      seat_class: "二等座",
      sort_by: "departure_asc",
    });
    assert.equal(departureResponse.status, 200);
    assert.deepEqual((await departureResponse.json()).items.map((item) => item.transport_no), ["G1", "G3"]);

    const durationResponse = await post(baseUrl, "/v1/search/trains", {
      origin: "深圳",
      destination: "广州",
      departure_date: "2026-08-02",
      sort_by: "duration_asc",
    });
    assert.equal(durationResponse.status, 200);
    assert.deepEqual((await durationResponse.json()).items.map((item) => item.transport_no), ["G2", "G4", "G1", "G3"]);
  } finally {
    await close(server);
  }
});

test("recommended 保留供应商顺序，price_asc 将缺价格结果稳定置后", async () => {
  const payload = providerEnvelope([
    rawFlight({ item: { ticketPrice: 900 }, segment: { marketingTransportNo: "CZ-A" } }),
    rawFlight({
      item: { ticketPrice: undefined },
      segment: { marketingTransportNo: "CZ-B", seatClassName: undefined },
    }),
    rawFlight({ item: { ticketPrice: 700 }, segment: { marketingTransportNo: "CZ-C" } }),
  ]);
  const { runner } = createRunner({ flightPayload: payload });
  const server = createAdapterServer({ token: TOKEN, apiKey: API_KEY, runner });
  const baseUrl = await listen(server);
  try {
    for (const [sortBy, expected] of [
      ["recommended", ["CZ-A", "CZ-B", "CZ-C"]],
      ["price_asc", ["CZ-C", "CZ-A", "CZ-B"]],
    ]) {
      const response = await post(baseUrl, "/v1/search/flights", {
        origin: "深圳",
        destination: "上海",
        departure_date: "2026-08-02",
        sort_by: sortBy,
      });
      assert.equal(response.status, 200);
      assert.deepEqual((await response.json()).items.map((item) => item.transport_no), expected);
    }
  } finally {
    await close(server);
  }
});

test("供应商 envelope 必须成功且 itemList 类型正确，真实空列表才返回空结果", async () => {
  for (const [payload, errorCode] of [
    [{ status: 7, message: "denied", data: { itemList: [] } }, "provider_error"],
    [{ message: "missing status", data: { itemList: [] } }, "invalid_provider_response"],
    [{ status: 0, data: { itemList: {} } }, "invalid_provider_response"],
  ]) {
    const { runner } = createRunner({ flightPayload: payload });
    const server = createAdapterServer({ token: TOKEN, apiKey: API_KEY, runner });
    const baseUrl = await listen(server);
    try {
      const response = await post(baseUrl, "/v1/search/flights", {
        origin: "深圳",
        destination: "上海",
        departure_date: "2026-08-02",
      });
      assert.equal(response.status, 502);
      assert.deepEqual(await response.json(), { error_code: errorCode });
    } finally {
      await close(server);
    }
  }

  const { runner } = createRunner({ flightPayload: providerEnvelope([]) });
  const server = createAdapterServer({ token: TOKEN, apiKey: API_KEY, runner });
  const baseUrl = await listen(server);
  try {
    const response = await post(baseUrl, "/v1/search/flights", {
      origin: "深圳",
      destination: "上海",
      departure_date: "2026-08-02",
    });
    assert.equal(response.status, 200);
    assert.deepEqual((await response.json()).items, []);
  } finally {
    await close(server);
  }
});

test("闭合 schema 拒绝未知字段、选项注入与非法时间", async () => {
  const { runner, calls } = createRunner();
  const server = createAdapterServer({ token: TOKEN, apiKey: API_KEY, runner });
  const baseUrl = await listen(server);
  try {
    for (const body of [
      { origin: "深圳", destination: "上海", departure_date: "2026-08-02", debug: true },
      { origin: "--debug", destination: "上海", departure_date: "2026-08-02" },
      { origin: "深圳\n--debug", destination: "上海", departure_date: "2026-08-02" },
      { origin: "深圳", destination: "上海", departure_date: "2026-08-02", seat_class: "二等座" },
      {
        origin: "深圳",
        destination: "上海",
        departure_date: "2026-08-02",
        departure_hour_start: 20,
        departure_hour_end: 8,
      },
    ]) {
      const response = await post(baseUrl, "/v1/search/flights", body);
      assert.equal(response.status, 400);
      assert.deepEqual(await response.json(), { error_code: "invalid_request" });
    }
    assert.equal(calls.length, 0);
  } finally {
    await close(server);
  }
});

test("严格 JSON 拒绝重复键、NaN、尾随内容、过深和过多节点", () => {
  assert.throws(() => parseStrictJson('{"a":1,"a":2}'), /duplicate_key/);
  assert.throws(() => parseStrictJson('{"a":NaN}'), /invalid_json/);
  assert.throws(() => parseStrictJson('{"a":1} trailing'), /invalid_json/);
  assert.throws(
    () => parseStrictJson(`${"[".repeat(25)}0${"]".repeat(25)}`, { maxDepth: 20 }),
    /json_too_deep/,
  );
  assert.throws(() => parseStrictJson("[1,2,3]", { maxNodes: 3 }), /json_too_many_nodes/);
});

test("同用户并发为 1，全局并发为 2", async () => {
  const { runner, getMaxActive } = createRunner({ delayMs: 120 });
  const server = createAdapterServer({ token: TOKEN, apiKey: API_KEY, runner });
  const baseUrl = await listen(server);
  const body = { origin: "深圳", destination: "上海", departure_date: "2026-08-02" };
  try {
    const first = post(baseUrl, "/v1/search/flights", body, { userScope: "same-user" });
    await new Promise((resolve) => setTimeout(resolve, 20));
    const sameUser = await post(baseUrl, "/v1/search/flights", body, { userScope: "same-user" });
    assert.equal(sameUser.status, 429);
    assert.deepEqual(await sameUser.json(), { error_code: "concurrency_limited" });

    const second = post(baseUrl, "/v1/search/flights", body, { userScope: "user-2" });
    await new Promise((resolve) => setTimeout(resolve, 20));
    const globalOverflow = await post(baseUrl, "/v1/search/flights", body, { userScope: "user-3" });
    assert.equal(globalOverflow.status, 429);
    await Promise.all([first, second]);
    assert.equal(getMaxActive(), 2);
  } finally {
    await close(server);
  }
});

test("客户端断开会把 AbortSignal 传给 runner 并释放并发槽", async () => {
  let aborted = false;
  let callCount = 0;
  const runner = ({ signal }) => {
    callCount += 1;
    if (callCount > 1) return Promise.resolve(providerEnvelope([]));
    return new Promise((resolve, reject) => {
      signal.addEventListener(
        "abort",
        () => {
          aborted = true;
          const error = new Error("request_aborted");
          error.code = "request_aborted";
          reject(error);
        },
        { once: true },
      );
    });
  };
  const server = createAdapterServer({ token: TOKEN, apiKey: API_KEY, runner });
  const baseUrl = await listen(server);
  const controller = new AbortController();
  try {
    const first = post(
      baseUrl,
      "/v1/search/flights",
      { origin: "深圳", destination: "上海", departure_date: "2026-08-02" },
      { signal: controller.signal },
    );
    await new Promise((resolve) => setTimeout(resolve, 30));
    controller.abort();
    await assert.rejects(first, /abort/iu);
    const deadline = Date.now() + 500;
    while (!aborted && Date.now() < deadline) await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(aborted, true);

    const retry = await post(baseUrl, "/v1/search/flights", {
      origin: "深圳",
      destination: "上海",
      departure_date: "2026-08-02",
    });
    assert.equal(retry.status, 200);
  } finally {
    await close(server);
  }
});

test("默认 HTTP 执行链路在客户端断开后清理真实 fake CLI 进程组", async (t) => {
  if (process.platform === "win32") t.skip("Windows 不支持 POSIX 进程组断言");
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "flyai-http-abort-"));
  const fixturePath = path.join(tempDir, "fixture.mjs");
  const pidPath = path.join(tempDir, "http-abort-pids.json");
  await writeFile(
    fixturePath,
    `
      import { spawn } from "node:child_process";
      import { writeFile } from "node:fs/promises";
      import path from "node:path";
      import { fileURLToPath } from "node:url";
      const grandchild = spawn(process.execPath, ["-e", "process.on('SIGTERM',()=>{});setInterval(()=>{},1000)"], { stdio: "ignore" });
      await writeFile(path.join(path.dirname(fileURLToPath(import.meta.url)), "http-abort-pids.json"), JSON.stringify({ leader: process.pid, grandchild: grandchild.pid }));
      process.on("SIGTERM", () => {});
      setInterval(() => {}, 1000);
    `,
  );
  const server = createAdapterServer({
    token: TOKEN,
    apiKey: API_KEY,
    homeDir: tempDir,
    runnerOptions: {
      executablePath: process.execPath,
      bundlePath: fixturePath,
      timeoutMs: 10_000,
    },
  });
  const baseUrl = await listen(server);
  const controller = new AbortController();
  try {
    const response = post(
      baseUrl,
      "/v1/search/flights",
      { origin: "深圳", destination: "上海", departure_date: "2026-08-02" },
      { signal: controller.signal },
    );
    const pids = await waitForFile(pidPath);
    const requestSettled = once(server, "flyai-request-settled", {
      signal: AbortSignal.timeout(3_000),
    });
    controller.abort();
    await assert.rejects(response, /abort/iu);
    await requestSettled;
    await waitForProcessesGone(Object.values(pids), 2_000);
    assert.deepEqual((await readdir(tempDir)).sort(), ["fixture.mjs", "http-abort-pids.json"]);
  } finally {
    await close(server);
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("CLI runner 使用绝对 argv 与最小环境", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "flyai-runner-test-"));
  const capturePath = path.join(tempDir, "capture.json");
  const fixturePath = path.join(tempDir, "fixture.mjs");
  await writeFile(
    fixturePath,
    `
      import { writeFile } from "node:fs/promises";
      import path from "node:path";
      import { fileURLToPath } from "node:url";
      await writeFile(path.join(path.dirname(fileURLToPath(import.meta.url)), "capture.json"), JSON.stringify({ argv: process.argv.slice(2), env: process.env }));
      process.stdout.write('{"status":0,"message":"success","data":{"itemList":[]},"systemMessage":""}');
    `,
  );
  try {
    const result = await runFlyaiCli({
      command: "search-flight",
      argv: ["--origin", "深圳; touch /tmp/never"],
      apiKey: API_KEY,
      homeDir: tempDir,
      executablePath: process.execPath,
      bundlePath: fixturePath,
    });
    assert.deepEqual(result.data.itemList, []);
    const capture = JSON.parse(await readFile(capturePath, "utf8"));
    assert.deepEqual(capture.argv, ["search-flight", "--origin", "深圳; touch /tmp/never"]);
    assert.equal(capture.env.FLYAI_API_KEY, API_KEY);
    assert.deepEqual(
      Object.keys(capture.env).filter((key) => key !== "__CF_USER_TEXT_ENCODING").sort(),
      ["FLYAI_API_KEY", "HOME", "LANG", "LC_ALL", "TMPDIR", "TZ"],
    );
    for (const forbidden of [
      "DEBUG_FLYAI",
      "DEBUG_FLYAI_MCP_URL",
      "NODE_OPTIONS",
      "NODE_PATH",
      "HTTP_PROXY",
      "HTTPS_PROXY",
      "ALL_PROXY",
      "NO_PROXY",
      "NODE_EXTRA_CA_CERTS",
      "SSL_CERT_FILE",
      "SSL_CERT_DIR",
    ]) {
      assert.equal(capture.env[forbidden], undefined);
    }
    assert.equal(capture.env.HOME.startsWith(`${tempDir}${path.sep}`), true);
    assert.equal(capture.env.TMPDIR, capture.env.HOME);
    assert.deepEqual((await readdir(tempDir)).sort(), ["capture.json", "fixture.mjs"]);
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("CLI runner 超时或中断后清理忽略 SIGTERM 的整个进程组", async (t) => {
  if (process.platform === "win32") t.skip("Windows 不支持 POSIX 进程组断言");
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "flyai-runner-tree-"));
  const fixturePath = path.join(tempDir, "fixture.mjs");
  await writeFile(
    fixturePath,
    `
      import { spawn } from "node:child_process";
      import { writeFile } from "node:fs/promises";
      import path from "node:path";
      import { fileURLToPath } from "node:url";
      const grandchild = spawn(process.execPath, ["-e", "process.on('SIGTERM',()=>{});setInterval(()=>{},1000)"], { stdio: "ignore" });
      await writeFile(path.join(path.dirname(fileURLToPath(import.meta.url)), process.argv[3]), JSON.stringify({ leader: process.pid, grandchild: grandchild.pid }));
      process.on("SIGTERM", () => {});
      setInterval(() => {}, 1000);
    `,
  );
  try {
    const timeoutRun = runFlyaiCli({
      command: "search-flight",
      argv: ["timeout-pids.json"],
      apiKey: API_KEY,
      homeDir: tempDir,
      executablePath: process.execPath,
      bundlePath: fixturePath,
      timeoutMs: 1_000,
    });
    const timeoutRejection = assert.rejects(timeoutRun, (error) => error.code === "provider_timeout");
    const timeoutPids = await waitForFile(path.join(tempDir, "timeout-pids.json"), 2_000);
    await timeoutRejection;
    await waitForProcessesGone(Object.values(timeoutPids));

    const controller = new AbortController();
    const abortedRun = runFlyaiCli({
      command: "search-flight",
      argv: ["abort-pids.json"],
      apiKey: API_KEY,
      homeDir: tempDir,
      executablePath: process.execPath,
      bundlePath: fixturePath,
      signal: controller.signal,
      timeoutMs: 10_000,
    });
    const abortedRejection = assert.rejects(abortedRun, (error) => error.code === "request_aborted");
    const abortPids = await waitForFile(path.join(tempDir, "abort-pids.json"), 2_000);
    controller.abort();
    await abortedRejection;
    await waitForProcessesGone(Object.values(abortPids));
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("CLI runner 拒绝 stdout/stderr 洪泛和恶意 JSON", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "flyai-runner-limits-"));
  const fixturePath = path.join(tempDir, "fixture.mjs");
  try {
    for (const [outputStream, output] of [
      ["stdout", "x".repeat(4096)],
      ["stderr", "x".repeat(4096)],
    ]) {
      await writeFile(fixturePath, `process.${outputStream}.write(${JSON.stringify(output)});\n`);
      await assert.rejects(
        runFlyaiCli({
          command: "search-flight",
          argv: [],
          apiKey: API_KEY,
          homeDir: tempDir,
          executablePath: process.execPath,
          bundlePath: fixturePath,
          maxStdoutBytes: 128,
          maxStderrBytes: 128,
        }),
        (error) => error.code === "provider_output_too_large",
      );
    }
    for (const output of [
      '{"a":1,"a":2}',
      '{"a":NaN}',
      '{"a":1}\n{"b":2}',
      `${"[".repeat(30)}0${"]".repeat(30)}`,
      Buffer.from([0xc3, 0x28]),
    ]) {
      const source = Buffer.isBuffer(output)
        ? `process.stdout.write(Buffer.from([${[...output].join(",")}]))`
        : `process.stdout.write(${JSON.stringify(output)})`;
      await writeFile(fixturePath, `${source};\n`);
      await assert.rejects(
        runFlyaiCli({
          command: "search-flight",
          argv: [],
          apiKey: API_KEY,
          homeDir: tempDir,
          executablePath: process.execPath,
          bundlePath: fixturePath,
        }),
        (error) => error.code === "invalid_provider_response",
      );
    }
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("HTTP 失败响应不回显供应商 stderr、原始结果或 Key", async () => {
  const runner = async () => {
    const error = new Error(`secret=${API_KEY}`);
    error.code = "provider_error";
    throw error;
  };
  const server = createAdapterServer({ token: TOKEN, apiKey: API_KEY, runner });
  const baseUrl = await listen(server);
  try {
    const response = await post(baseUrl, "/v1/search/flights", {
      origin: "深圳",
      destination: "上海",
      departure_date: "2026-08-02",
    });
    assert.equal(response.status, 502);
    const text = await response.text();
    assert.equal(text.includes(API_KEY), false);
    assert.deepEqual(JSON.parse(text), { error_code: "provider_error" });
  } finally {
    await close(server);
  }
});
