-- 原子检查 reader 所属流，并只在当前 message/task 的 streaming 流丢锁时落 interrupted 终态。
-- KEYS: lock, chunks, meta
-- ARGV: expected_message_id, expected_task_id, error_content, ended_at, done_ttl
-- 返回: {state, entry_id}; state = active|terminal|orphaned|replaced|missing

local expected_message_id = ARGV[1]
local expected_task_id = ARGV[2]
local status = redis.call("HGET", KEYS[3], "status")
if not status then
    return {"missing", ""}
end

local current_message_id = redis.call("HGET", KEYS[3], "message_id") or ""
local current_task_id = redis.call("HGET", KEYS[3], "task_id") or ""
if expected_message_id ~= "" and current_message_id ~= expected_message_id then
    return {"replaced", ""}
end
if expected_task_id ~= "" and current_task_id ~= expected_task_id then
    return {"replaced", ""}
end
if status ~= "streaming" then
    return {"terminal", ""}
end

local lock_owner = redis.call("GET", KEYS[1])
if lock_owner then
    if lock_owner == current_task_id then
        return {"active", ""}
    end
    return {"replaced", ""}
end
if current_task_id == "" then
    return {"missing", ""}
end

local entry_id = redis.call(
    "XADD", KEYS[2], "*",
    "type", "error",
    "content", ARGV[3],
    "block_id", ""
)
redis.call(
    "HSET", KEYS[3],
    "status", "error",
    "error_code", "stream_interrupted",
    "reason", "orphaned_stream",
    "ended_at", ARGV[4]
)
redis.call("EXPIRE", KEYS[2], tonumber(ARGV[5]))
redis.call("EXPIRE", KEYS[3], tonumber(ARGV[5]))
redis.call("DEL", KEYS[1])
return {"orphaned", entry_id}
