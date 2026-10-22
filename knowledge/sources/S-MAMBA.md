# S-MAMBA：Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## 本地完整原文

[完整原文阅读](../../references/S-MAMBA/reader.html) · [全文文本](../../references/S-MAMBA/fulltext.txt) · [原始文件](../../references/S-MAMBA/original.pdf)

完整 PDF：36 页，版本 2312.00752v2。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2024 / v2。

## 来源支持的机制与定义

Mamba 使用输入相关的选择性状态空间机制。递归生成保留有限大小的状态，长序列处理可以采用专门的扫描/分块执行方式。

## InfraAgents 如何据此推导

递归状态通常原地推进，不能只缩短 KV 长度来回滚。本库据此推导前缀检查点、投机快照、卷积状态与混合模型的共同恢复位置。

## 阅读范围与出处身份

选择性状态空间与递归状态的结构差异。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
