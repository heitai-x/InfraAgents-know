# S-METAL-TOOLS：MLX — Metal Debugger（固定历史文档）

## 本地完整原文

[完整原文](../../references/S-METAL-TOOLS/reader.html) · [全文文本](../../references/S-METAL-TOOLS/fulltext.txt)

固定版本：Git e8ebdebeeb655feaa85a51f6b24ece5b6d5518d1 (2026-04-28)。来源版本日期：2026-04-28；整理核对日期：2026-10-22。

## 来源支持的机制与定义

MLX 的固定 Metal Debugger 文档说明 GPU capture、Metal 对象标签，以及在 Xcode 中打开与重放 trace、查看操作依赖。它不单独证明完整 Instruments 时间线或所有硬件计数器的覆盖。

## InfraAgents 如何据此推导

由求值、GPU capture 和操作依赖还原实际执行边界；未被归档资料覆盖的平台工具能力保持未确认。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
