# S-TRTLLM：TensorRT-LLM Architecture Overview

## 本地完整原文

[完整原文阅读](../../references/S-TRTLLM/reader.html) · [全文文本](../../references/S-TRTLLM/fulltext.txt) · [原始文件](../../references/S-TRTLLM/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：online 2026-09-20。

## 来源支持的机制与定义

已读 TensorRT-LLM 架构页将入口、PyExecutor、scheduler、model engine 与 sampler 连接成后台迭代，并介绍 graph padding 和 overlap 思路。

## InfraAgents 如何据此推导

本库用两代 batch 关系解释 host/设备重叠，分析 padding、在途步与停止处理的成本，不迁移网页中的性能数字。

## 阅读范围与出处身份

运行时与执行器、模型后端和请求管理架构。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
