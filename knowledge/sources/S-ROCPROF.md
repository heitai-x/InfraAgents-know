# S-ROCPROF：ROCprofiler-SDK：rocprofv3

## 本地完整原文

[完整原文阅读](../../references/S-ROCPROF/reader.html) · [全文文本](../../references/S-ROCPROF/fulltext.txt) · [原始文件](../../references/S-ROCPROF/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：SDK 1.3.5 documentation snapshot。

## 来源支持的机制与定义

ROCprofiler-SDK 的 rocprofv3 区分 API/activity tracing 与 kernel counter collection，记录进程、agent、queue、dispatch 等身份。计数器可用集合和并行采集能力依 GPU 而定。

## InfraAgents 如何据此推导

本库用系统 trace 定位执行实例，再用计数器解释 CU、访存和寄存器瓶颈；多遍采集与正常并发路径分别解释。

## 阅读范围与出处身份

API/kernel tracing、设备计数器、输出归属与多遍采集。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
