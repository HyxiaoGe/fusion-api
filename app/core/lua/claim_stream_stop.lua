-- 原子占有一轮流的 stop 权，防止 partial 落库期间被新流替换。
-- KEYS: lock, meta, stop_guard
-- ARGV: message_id, task_id, guard_ttl

local lock_key = KEYS[1]
local meta_key = KEYS[2]
local guard_key = KEYS[3]
local expected_message_id = ARGV[1]
local expected_task_id = ARGV[2]
local guard_ttl = tonumber(ARGV[3])

if redis.call("HGET", meta_key, "status") ~= "streaming" then
    return 0
end
if redis.call("GET", lock_key) ~= expected_task_id then
    return 0
end
if redis.call("HGET", meta_key, "task_id") ~= expected_task_id then
    return 0
end
if redis.call("HGET", meta_key, "message_id") ~= expected_message_id then
    return 0
end

local claimed = redis.call("SET", guard_key, expected_task_id, "EX", guard_ttl, "NX")
if claimed then
    return 1
end
return 0
