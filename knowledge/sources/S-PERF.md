# S-PERF：Linux perf-stat manual

## 本地完整原文

[完整原文阅读](../../references/S-PERF/reader.html) · [全文文本](../../references/S-PERF/fulltext.txt) · [原始文件](../../references/S-PERF/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：manual snapshot 2026-09-20。

## 来源支持的机制与定义

perf stat 统计指定执行范围内的 CPU 性能事件，事件类型、计数对象和运行区间决定结果含义。

## InfraAgents 如何据此推导

本库把 cycles、instructions、cache 等可用计数与线程/算子范围对应；低 IPC 可能来自访存、依赖或调度，需要其他观测区分。

## 阅读范围与出处身份

CPU 性能事件统计、计数及运行区间。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
