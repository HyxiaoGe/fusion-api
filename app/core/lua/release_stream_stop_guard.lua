-- 仅由占有 stop guard 的 task 释放，避免迟到请求删除新 guard。
-- KEYS[1]: stop_guard
-- ARGV[1]: task_id

if redis.call("GET", KEYS[1]) == ARGV[1] then
    redis.call("DEL", KEYS[1])
    return 1
end
return 0
