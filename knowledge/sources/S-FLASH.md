# S-FLASH：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

## 本地完整原文

[完整原文阅读](../../references/S-FLASH/reader.html) · [全文文本](../../references/S-FLASH/fulltext.txt) · [原始文件](../../references/S-FLASH/original.pdf)

完整 PDF：34 页，版本 2205.14135v2。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2022 / v2。

## 来源支持的机制与定义

FlashAttention 通过片上分块和在线 softmax，避免将完整 attention 分数与概率矩阵写入设备主存；目标函数仍是精确 attention 的数学形式。

## InfraAgents 如何据此推导

对一块保留最大值 m、分母 ℓ 与未归一化加权和 u。合并另一块时按新的公共最大值重缩放 ℓ 和 u，最后计算 u/ℓ。本库据此解释 split-K、片上空间、IO 与浮点重排。

## 阅读范围与出处身份

IO-aware attention 的机制和精确计算目标。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
