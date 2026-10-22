# S-MLX-MEM：MLX：Unified Memory

## 本地完整原文

[完整原文阅读](../../references/S-MLX-MEM/reader.html) · [全文文本](../../references/S-MLX-MEM/fulltext.txt) · [原始文件](../../references/S-MLX-MEM/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：online 2026-09-20。

## 来源支持的机制与定义

MLX 利用 Apple Silicon 的统一内存，CPU/GPU 可以对共享内存中的数组执行操作，设备选择发生在操作执行层。

## InfraAgents 如何据此推导

共享容量仍有带宽和依赖成本。本库讨论 CPU/GPU 竞争、内存压力、标量同步以及请求状态隔离。

## 阅读范围与出处身份

共享内存模型与 CPU/GPU 数组执行。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
