# S-TORCH-DYNAMIC：PyTorch — Dynamic Shapes

## 本地完整原文

[完整原文](../../references/S-TORCH-DYNAMIC/reader.html) · [全文文本](../../references/S-TORCH-DYNAMIC/fulltext.txt)

固定版本：Git 647b6a2ff9a83a886464932dfad78781ceb61cb6 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

编译会基于 shape、stride、dtype、常量等条件特化，并用 guard 确定专门程序是否可继续使用；部分维度可以符号化，变化也可能触发重编译。

## InfraAgents 如何据此推导

本库从运行分布推导分桶、padding 和变体成本，分别解释编译程序与 CUDA Graph 的 key，避免把动态 shape 支持等同于任意图重放。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
