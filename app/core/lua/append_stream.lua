-- 仅允许当前 task 向 streaming 状态的 Stream 追加 entry，并原子刷新 TTL。
-- KEYS: lock, chunks, meta
-- ARGV: task_id, stream_ttl, field_1, value_1, ...
-- 返回: {1, entry_id} 成功；{0, "ownership_lost"} 已失去写入权

local task_id = ARGV[1]
if redis.call("GET", KEYS[1]) ~= task_id then
    return {0, "ownership_lost"}
end
if redis.call("HGET", KEYS[3], "task_id") ~= task_id then
    return {0, "ownership_lost"}
end
if redis.call("HGET", KEYS[3], "status") ~= "streaming" then
    return {0, "ownership_lost"}
end

local fields = {}
for index = 3, #ARGV do
    table.insert(fields, ARGV[index])
end
local entry_id = redis.call("XADD", KEYS[2], "*", unpack(fields))
redis.call("EXPIRE", KEYS[2], tonumber(ARGV[2]))
return {1, entry_id}
