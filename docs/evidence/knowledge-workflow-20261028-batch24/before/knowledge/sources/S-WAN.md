# S-WAN：Wan: Open and Advanced Large-Scale Video Generative Models

## 本地完整原文

[完整原文](../../references/S-WAN/reader.html) · [全文文本](../../references/S-WAN/fulltext.txt)

固定版本：2503.20314v2。来源版本日期：2025-04-19；整理核对日期：2026-10-22。

## 来源支持的机制与定义

Wan 的视频生成报告涉及时空建模和视频潜空间/解码结构；输出帧与压缩潜表示具有不同的时间和空间粒度。

## InfraAgents 如何据此推导

本库按 T′、H′、W′ 推导 token 数、时空 attention 与 VAE 分块成本，区分精确计算分块和改变时间上下文的分段生成。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
