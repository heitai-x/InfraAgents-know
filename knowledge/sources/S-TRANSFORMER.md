# S-TRANSFORMER：Attention Is All You Need

## 本地完整原文

[完整原文阅读](../../references/S-TRANSFORMER/reader.html) · [全文文本](../../references/S-TRANSFORMER/fulltext.txt) · [原始文件](../../references/S-TRANSFORMER/original.pdf)

完整 PDF：15 页，版本 1706.03762v7。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2017 paper / v7。

## 来源支持的机制与定义

Transformer 的 attention 用 query 与 keys 的分数生成归一化权重，再对 values 加权；encoder-decoder attention 的 Q 来自 decoder，K/V 来自 encoder 表示。

## InfraAgents 如何据此推导

当 E 与投影权重固定时，K＝EWk、V＝EWv 不依赖 decoder 步数，因而可以缓存。每步 QK 与加权求和仍存在；具体 mask 和位置规则由模型定义。

## 阅读范围与出处身份

注意力和 encoder-decoder 结构的基础来源。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
