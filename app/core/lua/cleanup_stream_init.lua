-- 仅清理仍由本次 task 持有的模糊失败初始化，防止迟到 cleanup 删除新一轮。
-- KEYS: lock, chunks, meta
-- ARGV: task_id

local task_id = ARGV[1]
if redis.call("GET", KEYS[1]) ~= task_id then
    return 0
end
if redis.call("HGET", KEYS[3], "task_id") ~= task_id then
    return 0
end

redis.call("DEL", KEYS[1], KEYS[2], KEYS[3])
return 1
