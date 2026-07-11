-- 原子初始化一轮 Redis Stream。
-- KEYS: lock, chunks, meta
-- ARGV: task_id, user_id, model, message_id, conversation_id, started_at, lock_ttl, stream_ttl

local lock_key = KEYS[1]
local stream_key = KEYS[2]
local meta_key = KEYS[3]
local task_id = ARGV[1]
local stream_ttl = tonumber(ARGV[8])

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
    "task_id", task_id
)
redis.call("EXPIRE", meta_key, stream_ttl)
redis.call("SET", lock_key, task_id, "EX", tonumber(ARGV[7]))
redis.call("XADD", stream_key, "*", "type", "start", "content", "", "block_id", "")
redis.call("EXPIRE", stream_key, stream_ttl)
return 1
