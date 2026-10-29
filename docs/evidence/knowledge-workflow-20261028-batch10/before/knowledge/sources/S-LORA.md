# S-LORA：LoRA: Low-Rank Adaptation of Large Language Models

## 本地完整原文

[完整原文](../../references/S-LORA/reader.html) · [全文文本](../../references/S-LORA/fulltext.txt)

固定版本：2106.09685v2。来源版本日期：2021-10-16；整理核对日期：2026-10-22。

## 来源支持的机制与定义

LoRA 用低秩增量 ΔW＝sBA 表达权重适配；固定 adapter 可在条件允许时合并，多 adapter 可以共享基座并分别计算增量分支。

## InfraAgents 如何据此推导

本库按 rank 推导参数量和分组执行成本，分析 adapter 驻留、版本并存、图桶和受适配影响的缓存身份。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
