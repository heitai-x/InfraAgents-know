# S-FI-SAMPLE：FlashInfer：Sampling

## 本地完整原文

[完整原文阅读](../../references/S-FI-SAMPLE/reader.html) · [全文文本](../../references/S-FI-SAMPLE/fulltext.txt) · [原始文件](../../references/S-FI-SAMPLE/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：0.6.18 documentation snapshot。

## 来源支持的机制与定义

FlashInfer sampling 接口提供从概率或 logits 的按行采样，以及 top-k、top-p 等相关操作；调用的输入表示和参数决定实际概率处理。

## InfraAgents 如何据此推导

本库先确定 logits 行到请求/位置的映射，再定义变换顺序和随机流；语法 mask 与截断后应使用相容的归一化分布。

## 阅读范围与出处身份

按行采样、top-k/top-p 与概率接口。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
