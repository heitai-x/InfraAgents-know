# S-VLLM-HYBRID：vLLM：Hybrid KV Cache Manager

## 本地完整原文

[完整原文阅读](../../references/S-VLLM-HYBRID/reader.html) · [全文文本](../../references/S-VLLM-HYBRID/fulltext.txt) · [原始文件](../../references/S-VLLM-HYBRID/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：0.18.0 documentation；正文标注基于 commit 458e74。

## 来源支持的机制与定义

已读 0.18 设计文档注明其基于提交 458e74，区分 KV hidden bytes、token block size 和物理 page size，讨论层组统一池以及各组前缀命中的交集。

## InfraAgents 如何据此推导

本库将公共恢复位置推广为各状态族可恢复集合的交集，并讨论 LCM、联合预留与淘汰。文档中的具体版本功能限制不被外推到 0.24。

## 阅读范围与出处身份

层组布局、统一池与恢复位置交集；其中版本功能限制不外推到当前版本。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
