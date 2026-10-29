# S-ROCPROF：ROCprofiler-SDK：rocprofv3

## 本地完整原文

[完整原文](../../references/S-ROCPROF/reader.html) · [全文文本](../../references/S-ROCPROF/fulltext.txt)

固定版本：Git 95541731741b6821a0301b9c6a62922507b3835e (2026-02-16)。来源版本日期：2026-02-16；整理核对日期：2026-10-22。

## 来源支持的机制与定义

ROCprofiler-SDK 的 rocprofv3 区分 API/activity tracing 与 kernel counter collection，记录进程、agent、queue、dispatch 等身份。计数器可用集合和并行采集能力依 GPU 而定。

## InfraAgents 如何据此推导

本库用系统 trace 定位执行实例，再用计数器解释 CU、访存和寄存器瓶颈；多遍采集与正常并发路径分别解释。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
