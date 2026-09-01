# Fusion API（已迁移）

Fusion 的后端已经迁入统一仓库，不再在本仓库独立开发或发布：

- 新仓库：[HyxiaoGe/fusion](https://github.com/HyxiaoGe/fusion)
- 后端目录：[backend/](https://github.com/HyxiaoGe/fusion/tree/master/backend)
- 新问题与改动：[HyxiaoGe/fusion/issues](https://github.com/HyxiaoGe/fusion/issues)

本仓库保留历史提交、Issues、Pull Requests、Actions runs 与 Releases，供迁移前记录追溯；请勿再从这里发起新开发或发布。

数据库迁移继续遵循 `expand/contract`，镜像回滚绝不执行 `alembic downgrade`；现行发布与回滚契约以新仓库为准。
在旧仓发布链正式停用前，ACR 中的 SHA 标签继续作为手动回滚来源；停用后的回滚目标以新仓 per-app 台账和 digest 为准。
