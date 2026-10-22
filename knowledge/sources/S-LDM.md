# S-LDM：High-Resolution Image Synthesis with Latent Diffusion Models

## 本地完整原文

[完整原文](../../references/S-LDM/reader.html) · [全文文本](../../references/S-LDM/fulltext.txt)

固定版本：2112.10752v2。来源版本日期：2022-04-13；整理核对日期：2026-10-22。

## 来源支持的机制与定义

Latent Diffusion 把去噪计算放在编码器形成的潜空间，通过条件机制引导生成，并最终解码到像素空间。

## InfraAgents 如何据此推导

本库区分固定文本条件与逐步变化的 latent，解释哪些 context 可以复用，哪些 self-attention 状态会随去噪步改变。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
