-- 原子初始化一轮 Redis Stream。
-- KEYS: lock, chunks, meta, stop_guard
-- ARGV: task_id, user_id, model, message_id, conversation_id, started_at, lock_ttl, stream_ttl, stream_mode, message_sequence

local lock_key = KEYS[1]
local stream_key = KEYS[2]
local meta_key = KEYS[3]
local stop_guard_key = KEYS[4]
local task_id = ARGV[1]
local stream_ttl = tonumber(ARGV[8])
local stream_mode = ARGV[9] or "initial"

-- stop partial 正在持久化时不能替换当前流；检查必须发生在任何写操作之前。
if redis.call("EXISTS", stop_guard_key) == 1 then
    return 0
end

redis.call("DEL", stream_key, meta_key)
redis.call(
    "HSET",
    meta_key,
    "status", "streaming",
    "user_id", ARGV[2],
    "model", ARGV[3],
    "message_id", ARGV[4],
    "conversation_id", ARGV[5],
    "started_at", ARGV[6],
    "task_id", task_id,
    "stream_mode", stream_mode
)
if ARGV[10] and ARGV[10] ~= "" then
    redis.call("HSET", meta_key, "message_sequence", ARGV[10])
end
redis.call("EXPIRE", meta_key, stream_ttl)
redis.call("SET", lock_key, task_id, "EX", tonumber(ARGV[7]))
redis.call("XADD", stream_key, "*", "type", "start", "content", "", "block_id", "")
redis.call("EXPIRE", stream_key, stream_ttl)
return 1
