# S-NCU：Nsight Compute — Profiling Guide

## 本地完整原文

[完整原文阅读](../../references/S-NCU/reader.html) · [全文文本](../../references/S-NCU/fulltext.txt) · [原始文件](../../references/S-NCU/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：online snapshot 2026-09-20。

## 来源支持的机制与定义

Nsight Compute 提供 kernel 级硬件计数器和派生指标，某些采集需要 replay。计数器能力、采集模式及恢复条件影响所观察的执行。

## InfraAgents 如何据此推导

本库先由时间线确定代表 kernel，再用计数器解释资源限制；replay 下 cache、并发和状态环境与正常服务可能不同。

## 阅读范围与出处身份

计数器、replay、采集扰动与指标限制。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
