# S-SDPA：PyTorch：scaled_dot_product_attention

## 本地完整原文

[完整原文](../../references/S-SDPA/reader.html) · [全文文本](../../references/S-SDPA/fulltext.txt)

固定版本：Git 647b6a2ff9a83a886464932dfad78781ceb61cb6 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

已核对 SDPA 文档：布尔 mask 的 true 表示参与 attention；dropout_p 大于零时执行 dropout；非方形 is_causal 采用文档定义的左上对齐。Q、K/V heads 与通道形状有明确关系。

## InfraAgents 如何据此推导

本库把 mask 含义、对齐、dropout 和 GQA 作为独立语义契约，先建立函数等价，再比较后端成本。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
