-- cancel_stream.lua
-- 原子取消流：检查 meta 状态和 message_id 防止误杀新一轮
--
-- KEYS[1]: lock_key
-- KEYS[2]: stream_key
-- KEYS[3]: meta_key
-- ARGV[1]: done_ttl
-- ARGV[2]: message_id (可选，为空则不校验)
--
-- 返回: 1 已取消, 0 跳过

local lock_key = KEYS[1]
local stream_key = KEYS[2]
local meta_key = KEYS[3]
local done_ttl = tonumber(ARGV[1])
local expected_msg_id = ARGV[2]

-- 只有 streaming 状态才执行取消
local status = redis.call("HGET", meta_key, "status")
if status ~= "streaming" then
    return 0
end

-- 如果传了 message_id，校验是否匹配（防止取消旧流时误杀新流）
if expected_msg_id and expected_msg_id ~= "" then
    local current_msg_id = redis.call("HGET", meta_key, "message_id")
    if current_msg_id ~= expected_msg_id then
        return 0
    end
end

-- 删除 lock
redis.call("DEL", lock_key)

-- 写 error entry 让 SSE 读取器正常结束
redis.call("XADD", stream_key, "*", "type", "error", "content", "用户中止")

-- 更新 meta 状态
redis.call("HSET", meta_key, "status", "cancelled")

-- 缩短 TTL
redis.call("EXPIRE", stream_key, done_ttl)
redis.call("EXPIRE", meta_key, done_ttl)

return 1
