# S-ROCM：HIP：Hardware implementation

## 本地完整原文

[完整原文](../../references/S-ROCM/reader.html) · [全文文本](../../references/S-ROCM/fulltext.txt)

固定版本：Git 94e81d8098bcb999ffb4381ba7054267e5d7cb14 (2026-04-16)。来源版本日期：2026-04-16；整理核对日期：2026-10-22。

## 来源支持的机制与定义

HIP 硬件说明介绍 CU、SIMD、向量寄存器、LDS 与矩阵运算资源。寄存器使用影响驻留和隐藏延迟能力，不同 AMD 架构的资源参数不同。

## InfraAgents 如何据此推导

本库重新推导 tile、wave 和数据布局，而不继承 CUDA 硬编码常量；同一模型函数可选择不同平台 kernel。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
