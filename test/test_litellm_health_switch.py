"""LITELLM_HEALTH_ENABLED 开关与 Redis 多实例协调测试。

背景：/health 会对 LiteLLM DB 里每个模型各打一次真实 completion（qwen reasoning
模型每次数百 reasoning token），多 worker 与多服务（fusion-api / ai-audio-assistant-web）
各自起探测循环会重复烧钱。修复后：
1. LITELLM_HEALTH_ENABLED 默认 false —— 不启动后台循环、不请求 /health；
2. 开启后用 Redis round-claim 协调 —— 每周期全集群最多一轮探测，结果写共享快照。
"""

import asyncio
import json
import os
import unittest
from unittest.mock import patch

from app.ai import litellm_health


class _FakeAsyncRedis:
    """SET NX EX / GET 语义的最小 fake，模拟跨实例共享的 Redis。"""

    def __init__(self):
        self._data = {}

    async def set(self, name, value, nx=False, ex=None):
        if nx and name in self._data:
            return None
        self._data[name] = value
        return True

    async def get(self, name):
        return self._data.get(name)


class _SyncFake:
    """同步读客户端（Redis 快照读取路径）。"""

    def __init__(self, inner):
        self._inner = inner

    def get(self, name):
        return self._inner._data.get(name)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, responses, **_kwargs):
        self._responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url):
        return _FakeResponse(self._responses[url.rsplit("/", 1)[-1]])


class LitellmHealthSwitchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._saved = {}
        for var in ("LITELLM_HEALTH_ENABLED", "LITELLM_HEALTH_INTERVAL_SECONDS"):
            self._saved[var] = os.environ.pop(var, None)
        with litellm_health._lock:
            litellm_health._by_alias.clear()
        litellm_health._refresh_task = None
        litellm_health._last_redis_sync_at = 0.0
        litellm_health._sync_redis_client = None

    def tearDown(self):
        for var, saved in self._saved.items():
            if saved is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = saved
        task = litellm_health._refresh_task
        litellm_health._refresh_task = None
        if task is not None and not task.done():
            task.cancel()

    # ── 开关：enabled / disabled ────────────────────────────────

    def test_disabled_by_default(self):
        """默认关闭：避免 dev 环境继续产生全模型探活费用。"""
        self.assertFalse(litellm_health._is_enabled())

    def test_enabled_flag_parsing(self):
        os.environ["LITELLM_HEALTH_ENABLED"] = "true"
        self.assertTrue(litellm_health._is_enabled())

    async def test_start_disabled_does_not_start_loop(self):
        """disabled 时 startup 不得启动任何后台循环。"""
        await litellm_health.start()
        self.assertIsNone(litellm_health._refresh_task)

    async def test_start_enabled_starts_loop(self):
        """enabled 时启动一个后台循环。"""
        os.environ["LITELLM_HEALTH_ENABLED"] = "true"
        await litellm_health.start()
        self.assertIsNotNone(litellm_health._refresh_task)
        await litellm_health.stop()
        self.assertIsNone(litellm_health._refresh_task)

    def test_disabled_get_health_returns_unknown(self):
        """disabled 时健康状态回退 unknown（FE 按可用处理，不阻塞用户）。"""
        self.assertEqual(
            litellm_health.get_health("qwen3.7-max"),
            {"status": "unknown", "error": None, "checked_at": None},
        )

    def test_disabled_reader_never_touches_redis(self):
        """disabled 时读取不碰 Redis 快照，直接返回进程内状态。"""
        with patch.object(litellm_health, "_get_sync_redis") as mock:
            litellm_health.get_health("qwen3.7-max")
        mock.assert_not_called()

    # ── Redis round-claim：多 worker / 多服务协调 ────────────────

    async def test_concurrent_claim_single_winner(self):
        """模拟多个实例同时启动（含跨服务）：只有一方抢到本轮探测权。"""
        fake = _FakeAsyncRedis()
        with patch.object(litellm_health, "_get_async_redis", return_value=fake):
            results = await asyncio.gather(
                litellm_health._try_claim_round(),
                litellm_health._try_claim_round(),
                litellm_health._try_claim_round(),
            )
        self.assertEqual(sum(1 for r in results if r), 1)

    async def test_claim_rejected_while_held(self):
        """同一周期内第二个实例抢不到探测权。"""
        fake = _FakeAsyncRedis()
        with patch.object(litellm_health, "_get_async_redis", return_value=fake):
            self.assertTrue(await litellm_health._try_claim_round())
            self.assertFalse(await litellm_health._try_claim_round())

    async def test_claim_fails_closed_when_redis_unavailable(self):
        """Redis 不可用时跳过本轮：宁可 unknown，也不能失去协调地重复探测烧钱。"""
        fake = _FakeAsyncRedis()

        async def _boom(*_a, **_k):
            raise ConnectionError("redis down")

        fake.set = _boom
        with patch.object(litellm_health, "_get_async_redis", return_value=fake):
            self.assertFalse(await litellm_health._try_claim_round())

    async def test_two_workers_share_one_probe_per_round(self):
        """两个 worker 同时启动时，每轮最多执行一次探测（_fetch_once）。"""
        fake = _FakeAsyncRedis()
        with patch.object(litellm_health, "_get_async_redis", return_value=fake):
            calls = []

            async def _fake_fetch():
                calls.append(1)

            with patch.object(litellm_health, "_fetch_once", side_effect=_fake_fetch):

                async def _worker_round():
                    if await litellm_health._try_claim_round():
                        await litellm_health._fetch_once()

                await asyncio.gather(_worker_round(), _worker_round())
        self.assertEqual(len(calls), 1)

    # ── 共享快照：探测结果写 Redis，其它实例可读 ─────────────────

    async def test_probe_writes_shared_snapshot(self):
        """探测成功后健康结果写入 Redis 共享快照（供其它 worker / 其它服务读取）。"""
        fake = _FakeAsyncRedis()
        responses = {
            "info": {
                "data": [
                    {
                        "model_name": "kimi-k2.7-code",
                        "model_info": {"id": "uuid-k27"},
                    }
                ]
            },
            "health": {
                "healthy_endpoints": [{"model_id": "uuid-k27"}],
                "unhealthy_endpoints": [],
            },
        }
        with (
            patch.object(litellm_health, "_get_async_redis", return_value=fake),
            patch.object(
                litellm_health.httpx,
                "AsyncClient",
                side_effect=lambda **kwargs: _FakeAsyncClient(responses, **kwargs),
            ),
            patch.object(litellm_health.time, "time", return_value=456.0),
        ):
            await litellm_health._fetch_once()

        self.assertEqual(
            litellm_health.get_health("kimi-k2.7-code"),
            {"status": "healthy", "error": None, "checked_at": 456.0},
        )
        raw = fake._data[litellm_health._SNAPSHOT_KEY]
        payload = json.loads(raw)
        self.assertEqual(payload["checked_at"], 456.0)
        self.assertEqual(payload["by_alias"]["kimi-k2.7-code"]["status"], "healthy")

    async def test_reader_syncs_from_shared_snapshot(self):
        """另一个实例（worker/服务）启动后，能从 Redis 快照恢复健康状态。"""
        os.environ["LITELLM_HEALTH_ENABLED"] = "true"
        fake = _FakeAsyncRedis()
        fake._data[litellm_health._SNAPSHOT_KEY] = json.dumps(
            {
                "checked_at": 456.0,
                "by_alias": {"kimi-k2.7-code": {"status": "healthy", "error": None}},
            }
        )
        with patch.object(litellm_health, "_get_sync_redis", return_value=_SyncFake(fake)):
            self.assertEqual(
                litellm_health.get_health("kimi-k2.7-code"),
                {"status": "healthy", "error": None, "checked_at": 456.0},
            )

    async def test_probe_failure_keeps_stale_state(self):
        """探测失败时保留旧快照，不清空健康状态。"""
        fake = _FakeAsyncRedis()
        responses = {
            "info": {
                "data": [
                    {
                        "model_name": "kimi-k2.7-code",
                        "model_info": {"id": "uuid-k27"},
                    }
                ]
            },
            "health": {
                "healthy_endpoints": [{"model_id": "uuid-k27"}],
                "unhealthy_endpoints": [],
            },
        }
        with (
            patch.object(litellm_health, "_get_async_redis", return_value=fake),
            patch.object(
                litellm_health.httpx,
                "AsyncClient",
                side_effect=lambda **kwargs: _FakeAsyncClient(responses, **kwargs),
            ),
            patch.object(litellm_health.time, "time", return_value=456.0),
        ):
            await litellm_health._fetch_once()
        self.assertEqual(litellm_health.get_health("kimi-k2.7-code")["status"], "healthy")

        # 下一轮探测失败（LiteLLM 不可达）→ 旧状态保留
        class _BadClient(_FakeAsyncClient):
            async def get(self, url):
                raise RuntimeError("litellm down")

        with patch.object(
            litellm_health.httpx,
            "AsyncClient",
            side_effect=lambda **kwargs: _BadClient(responses, **kwargs),
        ):
            await litellm_health._fetch_once()
        self.assertEqual(litellm_health.get_health("kimi-k2.7-code")["status"], "healthy")

    async def test_record_success_newer_than_snapshot_is_preserved(self):
        """P1：本地 record_success 比 Redis 快照新时，不被旧快照的 unhealthy 覆盖。"""
        os.environ["LITELLM_HEALTH_ENABLED"] = "true"
        fake = _FakeAsyncRedis()
        # Redis 仍存上一轮 unhealthy@456 快照
        fake._data[litellm_health._SNAPSHOT_KEY] = json.dumps(
            {
                "checked_at": 456.0,
                "by_alias": {"kimi-k2.7-code": {"status": "unhealthy", "error": "服务商暂时不可用"}},
            }
        )
        # 随后真实调用成功，本地写入 healthy@789
        litellm_health.record_success("kimi-k2.7-code", checked_at=789.0)

        with patch.object(litellm_health, "_get_sync_redis", return_value=_SyncFake(fake)):
            health = litellm_health.get_health("kimi-k2.7-code")
        self.assertEqual(
            health,
            {"status": "healthy", "error": None, "checked_at": 789.0},
        )

    async def test_snapshot_removes_aliases_absent_from_new_snapshot(self):
        """更新快照里消失的 alias 应收敛为 unknown（除非本地有更新的 record_success）。"""
        os.environ["LITELLM_HEALTH_ENABLED"] = "true"
        fake = _FakeAsyncRedis()
        fake._data[litellm_health._SNAPSHOT_KEY] = json.dumps(
            {
                "checked_at": 456.0,
                "by_alias": {"kimi-k2.7-code": {"status": "unhealthy", "error": "服务商暂时不可用"}},
            }
        )
        with patch.object(litellm_health, "_get_sync_redis", return_value=_SyncFake(fake)):
            self.assertEqual(litellm_health.get_health("kimi-k2.7-code")["status"], "unhealthy")

        # 新快照不再包含该 alias → 收敛为 unknown
        fake._data[litellm_health._SNAPSHOT_KEY] = json.dumps(
            {
                "checked_at": 789.0,
                "by_alias": {"qwen3.7-max": {"status": "healthy", "error": None}},
            }
        )
        litellm_health._last_redis_sync_at = 0.0
        with patch.object(litellm_health, "_get_sync_redis", return_value=_SyncFake(fake)):
            self.assertEqual(litellm_health.get_health("kimi-k2.7-code")["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
