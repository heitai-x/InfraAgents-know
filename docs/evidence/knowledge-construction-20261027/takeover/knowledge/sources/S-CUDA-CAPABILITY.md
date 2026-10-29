# S-CUDA-CAPABILITY：CUDA Programming Guide — Compute Capabilities

## 本地完整原文

[完整原文](../../references/S-CUDA-CAPABILITY/reader.html) · [全文文本](../../references/S-CUDA-CAPABILITY/fulltext.txt)

固定版本：CUDA 12.8.0。来源版本日期：2025-01-27；整理核对日期：2026-10-22。

## 来源支持的机制与定义

CUDA 的硬件功能与数值指令能力按 compute capability 和设备代际区分；一种 dtype 可被表示并不意味着所有算子形态都有高效实现。

## InfraAgents 如何据此推导

本库分别分析存储、乘法、累加和输出精度，以及 tile、对齐、打包格式与真实 dispatch。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
