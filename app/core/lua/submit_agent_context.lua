-- 原子校验并提交一次 Agent 上下文结果。
-- KEYS: request_hash, notify_list, stream_meta
-- ARGV: user_id, conversation_id, run_id, now, result_json, notify_ttl

local status = redis.call("HGET", KEYS[1], "status")
if not status then
    return "not_found"
end

if redis.call("HGET", KEYS[1], "user_id") ~= ARGV[1]
    or redis.call("HGET", KEYS[1], "conversation_id") ~= ARGV[2]
    or redis.call("HGET", KEYS[1], "run_id") ~= ARGV[3] then
    return "forbidden"
end

local expires_at = tonumber(redis.call("HGET", KEYS[1], "expires_at") or "0")
if expires_at <= tonumber(ARGV[4]) then
    redis.call("HSET", KEYS[1], "status", "expired")
    return "expired"
end

if status == "resolved" then
    if redis.call("HGET", KEYS[1], "result_json") == ARGV[5] then
        return "idempotent"
    end
    return "conflict"
end
if status ~= "pending" then
    return status == "expired" and "expired" or "conflict"
end

local expected_task_id = redis.call("HGET", KEYS[1], "task_id") or ""
if redis.call("HGET", KEYS[3], "status") ~= "streaming"
    or redis.call("HGET", KEYS[3], "user_id") ~= ARGV[1]
    or redis.call("HGET", KEYS[3], "conversation_id") ~= ARGV[2]
    or redis.call("HGET", KEYS[3], "task_id") ~= expected_task_id then
    return "stale"
end

redis.call("HSET", KEYS[1], "status", "resolved", "result_json", ARGV[5])
redis.call("LPUSH", KEYS[2], "resolved")
redis.call("EXPIRE", KEYS[2], tonumber(ARGV[6]))
return "accepted"
