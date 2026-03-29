-- release_lock.lua
-- 原子释放锁（仅当 value 匹配时才删除）
-- 标准 Redlock 释放模式
--
-- KEYS[1]: lock_key
-- ARGV[1]: task_id
--
-- 返回: 1 已释放, 0 跳过（锁不匹配）

if redis.call("GET", KEYS[1]) == ARGV[1] then
    redis.call("DEL", KEYS[1])
    return 1
end
return 0
