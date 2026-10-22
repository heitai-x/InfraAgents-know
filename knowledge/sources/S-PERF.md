# S-PERF：Linux perf-stat manual

## 本地完整原文

[完整原文](../../references/S-PERF/reader.html) · [全文文本](../../references/S-PERF/fulltext.txt)

固定版本：Git e027c218c482c6a0ae1948129ccda3b0a2033368 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

perf stat 统计指定执行范围内的 CPU 性能事件，事件类型、计数对象和运行区间决定结果含义。

## InfraAgents 如何据此推导

本库把 cycles、instructions、cache 等可用计数与线程/算子范围对应；低 IPC 可能来自访存、依赖或调度，需要其他观测区分。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
