"""并行工具执行 + emitter→Redis 适配器。

spec §4.3。execute_tools_parallel 走 emitter 的
tool_call_started/completed 协议，并把 execute_tool_with_retry
（瞬时重试 + 30s timeout）夹在 start/completed 之间。
"""

import asyncio
import json
import time
import uuid
from typing import Optional

from app.core.logger import app_logger as logger
from app.services.agent.emitter import AgentEventEmitter
from app.services.stream_state_service import append_chunk

# 单次工具调用超时
AGENT_TOOL_TIMEOUT = 30
# 瞬时故障重试次数
AGENT_TOOL_MAX_RETRIES = 1


class AgentEventRedisWriter:
    """把 emitter 的 (conv_id, chunk_type, payload:dict) 调用转成
    stream_state_service.append_chunk(conv_id, chunk_type, content:str, block_id:str)。

    Task 9 引入的 adapter — emitter 不直接知道 stream_state_service 的接口形态，
    通过本 adapter 桥接：payload JSON 序列化进 content 字段，block_id 留空。
    """

    async def append_chunk(
        self, conversation_id: str, chunk_type: str, payload: dict
    ) -> None:
        await append_chunk(
            conversation_id,
            chunk_type,
            json.dumps(payload, ensure_ascii=False),
            "",  # block_id 不适用于 agent_event chunk
        )


async def execute_tool_with_retry(
    handler,
    args: dict,
    max_retries: int = AGENT_TOOL_MAX_RETRIES,
):
    """带重试的工具执行（仅瞬时故障重试），返回 ToolResult。

    永久性错误（not_found / invalid / 401 / 403 / 404 / 400 / rate_limit）不重试。
    超时：单次 AGENT_TOOL_TIMEOUT 秒，超时被视为可重试失败。
    """
    from app.services.tool_handlers import ToolResult

    for attempt in range(max_retries + 1):
        try:
            result = await asyncio.wait_for(
                handler.execute(args),
                timeout=AGENT_TOOL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            result = ToolResult(status="failed", error_message="工具调用超时")

        if result.status == "success":
            return result

        # 永久性错误不重试
        err = (result.error_message or "").lower()
        is_permanent = any(kw in err for kw in ["not_found", "invalid", "rate_limit", "400", "401", "403", "404"])
        if is_permanent or attempt >= max_retries:
            return result

        logger.warning(f"工具 {handler.tool_name} 执行失败（{attempt + 1}/{max_retries + 1}），1s 后重试")
        await asyncio.sleep(1)

    return result


async def execute_tools_parallel(
    tool_calls: list[dict],
    conversation_id: str,
    user_id: str,
    model_id: str,
    provider: str,
    trace_id: str = None,
    step_number: int = None,
    emitter: Optional[AgentEventEmitter] = None,
) -> list:
    """
    并行执行所有 tool_calls。

    统一走 handler.execute_with_emitter 的协议（base 发 tool_call_started /
    completed agent_event），但中间塞入 execute_tool_with_retry（瞬时重试 +
    30s timeout）。tool_call_logs 仍通过 handler.log 写入。

    返回 [(tool_call: dict, result: ToolResult, handler: BaseToolHandler|None,
           block_id: str, log_id: str), ...]
    """
    from app.services.tool_handlers import ToolResult
    from app.services.tool_handlers import get_handler as _get_handler

    async def _run_one(tc: dict):
        handler = _get_handler(tc["name"])
        block_id = f"blk_{uuid.uuid4().hex[:12]}"
        log_id = str(uuid.uuid4())

        if not handler:
            logger.warning(f"未知的 tool_call: {tc['name']}")
            result = ToolResult(status="failed", error_message=f"未知工具: {tc['name']}")
            return tc, result, None, block_id, log_id

        # 解析参数
        try:
            args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
        except json.JSONDecodeError:
            args = {}

        # ── emitter 路径 ──
        # 直接复用 handler.execute_with_emitter 的发 start/completed 协议，
        # 但在中间塞入 execute_tool_with_retry（瞬时重试 + 30s timeout）。
        # 不能 monkey-patch handler.execute（handler 是模块级 singleton，
        # 同 gather 内并发 tool_call 会相互覆盖）。这里手动复刻
        # execute_with_emitter 的契约：tool_call_started → 重试执行 → tool_call_completed。
        if emitter is not None:
            await emitter.tool_call_started(
                tool_call_id=tc["id"],
                tool_name=handler.tool_name,
                arguments=args,
            )
            _start_mono = time.monotonic()
            try:
                result = await execute_tool_with_retry(handler, args)
            except BaseException as _exc:  # noqa: BLE001 — 必须先 emit completed 再 raise
                _dur_ms = int((time.monotonic() - _start_mono) * 1000)
                synthetic_failed = ToolResult(
                    status="failed",
                    error_message=f"{type(_exc).__name__}: {_exc}",
                )
                await emitter.tool_call_completed(
                    tool_call_id=tc["id"],
                    tool_name=handler.tool_name,
                    status="failed",
                    duration_ms=_dur_ms,
                    result_summary=handler._build_result_summary(synthetic_failed),
                    error=f"{type(_exc).__name__}: {_exc}",
                )
                # CancelledError 必须向上传播，让外层 generate_to_redis 的
                # except CancelledError 块发出 run_interrupted（用户中止反馈）。
                # 否则 cancel 会被吃掉，FE 看不到中止反馈。
                # 注意：cancel 路径会跳过 try 块外的 handler.log（acceptable，
                # emit failed completed 已经到 FE，DB 日志不补偿避免 await 二次取消）。
                # 普通 Exception 则吞掉，返回 failed ToolResult，让消息流仍能完成。
                if isinstance(_exc, asyncio.CancelledError):
                    raise
                result = synthetic_failed
            else:
                _dur_ms = int((time.monotonic() - _start_mono) * 1000)
                # 同步 ToolResult.duration_ms（部分 handler 不写）
                if result.duration_ms is None:
                    result.duration_ms = _dur_ms
                await emitter.tool_call_completed(
                    tool_call_id=tc["id"],
                    tool_name=handler.tool_name,
                    status=result.status,
                    duration_ms=_dur_ms,
                    result_summary=handler._build_result_summary(result),
                    error=result.error_message if result.status != "success" else None,
                )
        else:
            # 兼容 emitter 缺省路径（不应在 generate_to_redis 内触发）
            result = await execute_tool_with_retry(handler, args)

        # 异步记录日志（tool_call_logs 路径不变）
        await handler.log(
            log_id=log_id,
            conversation_id=conversation_id,
            user_id=user_id,
            model_id=model_id,
            provider=provider,
            result=result,
            input_params=args,
            trace_id=trace_id,
            step_number=step_number,
        )

        return tc, result, handler, block_id, log_id

    results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls])
    return list(results)
