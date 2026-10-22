# S-AITER：ROCm AITER 官方仓库

## 本地完整原文

[完整原文](../../references/S-AITER/reader.html) · [全文文本](../../references/S-AITER/fulltext.txt)

固定版本：Git d6e73f96141bcdb61c2cc7ed1b09d874dea8ecf8 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

AITER 是 AMD ROCm 的算子实现项目，提供面向相应硬件的 attention、矩阵或相关算子路径。项目覆盖不代表每种形状和设备都有相同实现。

## InfraAgents 如何据此推导

本库将模型语义、dtype、shape、layout、架构与具体 dispatch 取交集，确定可用路径及是否需要转换。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
