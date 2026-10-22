# S-FI：FlashInfer：Attention kernels

## 本地完整原文

[完整原文阅读](../../references/S-FI/reader.html) · [全文文本](../../references/S-FI/fulltext.txt) · [原始文件](../../references/S-FI/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：0.6.18 documentation snapshot。

## 来源支持的机制与定义

已读 FlashInfer 0.6.18 接口说明区分 query 分段、KV 页表及末页长度。一些 wrapper 的 plan 安排计算/工作区，run 执行 attention；相关 plan 不能直接捕获入图。

## InfraAgents 如何据此推导

本库据此表达独立的 Q/KV 几何，推导跨请求源缓存、plan key、固定 metadata 地址和可变有效长度之间的关系。

## 阅读范围与出处身份

分页与变长接口、plan/run、图模式缓冲区。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
