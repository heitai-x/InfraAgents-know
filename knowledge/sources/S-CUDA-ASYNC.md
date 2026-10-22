# S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution

## 本地完整原文

[完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

固定版本：CUDA 12.8.0。来源版本日期：2025-01-27；整理核对日期：2026-10-22。

## 来源支持的机制与定义

CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。

## InfraAgents 如何据此推导

本库据此解释负的“设备开始减 API 结束”、双缓冲时序、传输源的生命周期以及可真正隐藏的 host 工作。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
