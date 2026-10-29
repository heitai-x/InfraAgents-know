# S-NCU：Nsight Compute — Profiling Guide

## 本地完整原文

[完整原文](../../references/S-NCU/reader.html) · [全文文本](../../references/S-NCU/fulltext.txt)

固定版本：Nsight Compute 2025.1。来源版本日期：2025-01-27；整理核对日期：2026-10-22。

## 来源支持的机制与定义

Nsight Compute 提供 kernel 级硬件计数器和派生指标，某些采集需要 replay。计数器能力、采集模式及恢复条件影响所观察的执行。

## InfraAgents 如何据此推导

本库先由时间线确定代表 kernel，再用计数器解释资源限制；replay 下 cache、并发和状态环境与正常服务可能不同。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
