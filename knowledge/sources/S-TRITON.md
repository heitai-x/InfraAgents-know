# S-TRITON：Triton：Fused Attention

## 本地完整原文

[完整原文](../../references/S-TRITON/reader.html) · [全文文本](../../references/S-TRITON/fulltext.txt)

固定版本：Git 042b2cdcb41bb15821682cb6ba4bb15b1f15abb3 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

Triton 的 fused attention 示例展示块级张量程序、在线 softmax 归约、tile 划分与相应配置选择。逻辑 tile 和实际线程/寄存器布局有映射关系。

## InfraAgents 如何据此推导

本库从复用、寄存器与片上存储活跃集合推导融合与 tile 权衡，并把 autotune 的冷成本与形状变体纳入服务模型。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
