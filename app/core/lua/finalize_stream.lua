-- finalize_stream.lua
-- 原子写入流结束标记（仅当锁匹配时才执行）
--
-- KEYS[1]: lock_key
-- KEYS[2]: stream_key
-- KEYS[3]: meta_key
-- ARGV[1]: task_id
-- ARGV[2]: entry_type ("done" | "error")
-- ARGV[3]: entry_content
-- ARGV[4]: done_ttl
--
-- 返回: 1 成功, 0 跳过（锁不匹配）

local lock_key = KEYS[1]
local stream_key = KEYS[2]
local meta_key = KEYS[3]
local task_id = ARGV[1]
local entry_type = ARGV[2]
local entry_content = ARGV[3]
local done_ttl = tonumber(ARGV[4])

-- 检查锁：必须严格匹配当前 task_id 才允许写入
-- lock 不存在（false）或 value 不匹配 → 一律跳过
local current = redis.call("GET", lock_key)
if current ~= task_id then
    return 0
end

-- 写结束标记到 Stream
redis.call("XADD", stream_key, "*", "type", entry_type, "content", entry_content)

-- 更新 meta 状态
redis.call("HSET", meta_key, "status", entry_type == "done" and "done" or "error")

-- 缩短 TTL
redis.call("EXPIRE", stream_key, done_ttl)
redis.call("EXPIRE", meta_key, done_ttl)

-- 释放锁
redis.call("DEL", lock_key)
return 1
