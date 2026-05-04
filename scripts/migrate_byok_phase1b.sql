-- Phase B: 删除 model_credentials 旧表
-- 前置：fusion-api 代码必须已部署 commit ff9d304（删除所有 ModelCredential 引用）才能跑此脚本
DROP TABLE IF EXISTS model_credentials;
