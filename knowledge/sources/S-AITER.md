# S-AITER：ROCm AITER 官方仓库

## 本地完整原文

[完整原文阅读](../../references/S-AITER/reader.html) · [全文文本](../../references/S-AITER/fulltext.txt) · [原始文件](../../references/S-AITER/repository.tar.gz)

固定源码提交：f8785ca672040d56ad8bdb03c5de26d26c0c6978；归档范围：entire_repository。

资料形态：按已核对范围整理的本地非代码释义。原文版本：main snapshot 2026-09-20。

## 来源支持的机制与定义

AITER 是 AMD ROCm 的算子实现项目，提供面向相应硬件的 attention、矩阵或相关算子路径。项目覆盖不代表每种形状和设备都有相同实现。

## InfraAgents 如何据此推导

本库将模型语义、dtype、shape、layout、架构与具体 dispatch 取交集，确定可用路径及是否需要转换。

## 阅读范围与出处身份

AMD 算子库的覆盖与硬件适配入口。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
