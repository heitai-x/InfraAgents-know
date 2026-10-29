# S-MLA：DeepSeek-V2：MLA 与解耦位置编码

## 本地完整原文

[完整原文](../../references/S-MLA/reader.html) · [全文文本](../../references/S-MLA/fulltext.txt)

固定版本：2405.04434v5。来源版本日期：2024-06-19；整理核对日期：2026-10-22。

## 来源支持的机制与定义

DeepSeek-V2 的 MLA 以低维 latent 联合表示内容 K/V，通过线性投影的吸收减少历史展开；位置相关旋转不能任意与投影交换，因此采用解耦 RoPE 分量。

## InfraAgents 如何据此推导

本库用 qᵀWk c＝((Wk)ᵀq)ᵀc 与线性值归约解释吸收，按 L(d_c＋d_R)b 计算缓存，并区分 prefill 与 decode 的执行路径。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
