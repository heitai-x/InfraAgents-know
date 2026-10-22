# S-DIT：Scalable Diffusion Models with Transformers

## 本地完整原文

[完整原文](../../references/S-DIT/reader.html) · [全文文本](../../references/S-DIT/fulltext.txt)

固定版本：2212.09748v2。来源版本日期：2023-03-02；整理核对日期：2026-10-22。

## 来源支持的机制与定义

DiT 使用 Transformer 处理潜空间 patch，去噪计算的 token 几何由 latent 网格和 patch 化决定。

## InfraAgents 如何据此推导

本库由网格推导 attention 工作与峰值空间，分别计条件编码、各次去噪、调度更新和最终解码。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
