# S-MLX-MEM：MLX：Unified Memory

## 本地完整原文

[完整原文](../../references/S-MLX-MEM/reader.html) · [全文文本](../../references/S-MLX-MEM/fulltext.txt)

固定版本：Git e8ebdebeeb655feaa85a51f6b24ece5b6d5518d1 (2026-04-28)。来源版本日期：2026-04-28；整理核对日期：2026-10-22。

## 来源支持的机制与定义

MLX 利用 Apple Silicon 的统一内存，CPU/GPU 可以对共享内存中的数组执行操作，设备选择发生在操作执行层。

## InfraAgents 如何据此推导

共享容量仍有带宽和依赖成本。本库讨论 CPU/GPU 竞争、内存压力、标量同步以及请求状态隔离。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
