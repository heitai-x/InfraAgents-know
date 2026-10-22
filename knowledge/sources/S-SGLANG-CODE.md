# S-SGLANG-CODE：SGLang — scheduler and radix cache source locations

## 本地完整原文

[完整原文阅读](../../references/S-SGLANG-CODE/reader.html) · [全文文本](../../references/S-SGLANG-CODE/fulltext.txt) · [原始文件](../../references/S-SGLANG-CODE/repository.tar.gz)

固定源码提交：dd83b54611897a5f80f4df59e69756bd2fb4b8ab；归档范围：entire_repository。

资料形态：按已核对范围整理的本地非代码释义。原文版本：Git dd83b54611897a5f80f4df59e69756bd2fb4b8ab。

## 来源支持的机制与定义

实际核对提交 dd83b546 的 scheduler.py 与 radix_cache.py：Scheduler、get_next_batch_to_run、run_batch，以及 RadixCache 的 match_prefix、insert、evict。定位与文件 SHA 已保存到本项目证据。

## InfraAgents 如何据此推导

源码入口用于连接前缀匹配、调度选择、执行与回收责任；本库的通用状态模型独立于这些方法名。

## 阅读范围与出处身份

只核对机制定位与职责，不复制上游实现。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
