# S-GQA：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

## 本地完整原文

[完整原文阅读](../../references/S-GQA/reader.html) · [全文文本](../../references/S-GQA/fulltext.txt) · [原始文件](../../references/S-GQA/original.pdf)

完整 PDF：7 页，版本 2305.13245v3。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2023 / v3。

## 来源支持的机制与定义

GQA 让一组 query heads 共享 K/V heads；MQA 是只保留一组 K/V 的特殊形态。输出 query head 数与 KV head 数需要分别描述。

## InfraAgents 如何据此推导

由 2LNHkvdb 计算普通同构 KV 容量。减少 Hkv 不同于等比例减少 Hq 对应的 attention 算术；内核是否复用读取及 TP 是否复制 KV 需另行分析。

## 阅读范围与出处身份

query heads 与 KV heads 的结构区别。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
