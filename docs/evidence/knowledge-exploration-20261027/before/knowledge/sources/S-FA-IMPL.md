# S-FA-IMPL：FlashAttention 官方实现说明

## 本地完整原文

[完整原文](../../references/S-FA-IMPL/reader.html) · [全文文本](../../references/S-FA-IMPL/fulltext.txt)

固定版本：Git c5f6ff4905b924d20b5fc2fe74551638cd1421f4 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

FlashAttention 的版本说明记录：2.1 将非等长 Q/K 的 causal 对齐由左上改成右下。示例 Q 长 2、K 长 5 时，右下对齐使两行分别可见前 4、5 个位置。

## InfraAgents 如何据此推导

本库以绝对位置推导期望 mask，再比较 backend 接口，不能只把同名 causal 布尔值直接替换。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
