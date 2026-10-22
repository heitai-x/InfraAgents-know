# S-DIT：Scalable Diffusion Models with Transformers

## 本地完整原文

[完整原文阅读](../../references/S-DIT/reader.html) · [全文文本](../../references/S-DIT/fulltext.txt) · [原始文件](../../references/S-DIT/original.pdf)

完整 PDF：25 页，版本 2212.09748v2。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2022 paper。

## 来源支持的机制与定义

DiT 使用 Transformer 处理潜空间 patch，去噪计算的 token 几何由 latent 网格和 patch 化决定。

## InfraAgents 如何据此推导

本库由网格推导 attention 工作与峰值空间，分别计条件编码、各次去噪、调度更新和最终解码。

## 阅读范围与出处身份

潜空间 patch 与 Transformer 去噪结构。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
