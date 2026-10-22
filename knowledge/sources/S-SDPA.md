# S-SDPA：PyTorch：scaled_dot_product_attention

## 本地完整原文

[完整原文阅读](../../references/S-SDPA/reader.html) · [全文文本](../../references/S-SDPA/fulltext.txt) · [原始文件](../../references/S-SDPA/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：main snapshot 2026-09-20。

## 来源支持的机制与定义

已核对 SDPA 文档：布尔 mask 的 true 表示参与 attention；dropout_p 大于零时执行 dropout；非方形 is_causal 采用文档定义的左上对齐。Q、K/V heads 与通道形状有明确关系。

## InfraAgents 如何据此推导

本库把 mask 含义、对齐、dropout 和 GQA 作为独立语义契约，先建立函数等价，再比较后端成本。

## 阅读范围与出处身份

mask 语义、dropout、GQA 与后端选择。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
