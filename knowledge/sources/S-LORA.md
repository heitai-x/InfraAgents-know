# S-LORA：LoRA: Low-Rank Adaptation of Large Language Models

## 本地完整原文

[完整原文阅读](../../references/S-LORA/reader.html) · [全文文本](../../references/S-LORA/fulltext.txt) · [原始文件](../../references/S-LORA/original.pdf)

完整 PDF：26 页，版本 2106.09685v2。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2021 paper。

## 来源支持的机制与定义

LoRA 用低秩增量 ΔW＝sBA 表达权重适配；固定 adapter 可在条件允许时合并，多 adapter 可以共享基座并分别计算增量分支。

## InfraAgents 如何据此推导

本库按 rank 推导参数量和分组执行成本，分析 adapter 驻留、版本并存、图桶和受适配影响的缓存身份。

## 阅读范围与出处身份

低秩增量权重与合并关系。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
