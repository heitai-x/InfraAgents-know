# S-NCCL-COLLECTIVE：NCCL — Collective Operations

## 本地完整原文

[完整原文阅读](../../references/S-NCCL-COLLECTIVE/reader.html) · [全文文本](../../references/S-NCCL-COLLECTIVE/fulltext.txt) · [原始文件](../../references/S-NCCL-COLLECTIVE/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：NCCL 2.31.2 documentation。

## 来源支持的机制与定义

集合通信操作由 communicator、参与 ranks、数据类型、数量、归约方式或 root 等共同定义。参与者必须执行相互匹配的逻辑操作。

## InfraAgents 如何据此推导

本库由矩阵分片推导 gather/reduce 的必要性，再讨论动态批、零 token rank、失败传播与共同提交点。

## 阅读范围与出处身份

参与 rank 和集合操作参数匹配。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
