# S-NSYS：Nsight Systems — Post-Collection Analysis Guide

## 本地完整原文

[完整原文阅读](../../references/S-NSYS/reader.html) · [全文文本](../../references/S-NSYS/fulltext.txt) · [原始文件](../../references/S-NSYS/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：online snapshot 2026-09-20。

## 来源支持的机制与定义

Nsight Systems 的原始事件与导出可用于查看 host API、设备 kernel、复制和关联信息。事件属于具体进程、线程、设备和时间窗口。

## InfraAgents 如何据此推导

本库从请求关联到批次，再到 API/kernel；分别统计累计区间与区间并集。缺表、空表、窗口外与无法关联保持不同意义。

## 阅读范围与出处身份

API/kernel 关联、队列统计与原始导出解释。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
