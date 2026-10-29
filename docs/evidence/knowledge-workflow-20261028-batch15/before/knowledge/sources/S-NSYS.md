# S-NSYS：Nsight Systems 2024.6 — User Guide

## 本地完整原文

[完整原文](../../references/S-NSYS/reader.html) · [全文文本](../../references/S-NSYS/fulltext.txt)

固定版本：Nsight Systems 2024.6.2。来源版本日期：2025-01-27；整理核对日期：2026-10-22。

## 来源支持的机制与定义

Nsight Systems 2024.6 用户指南描述 host API、设备 kernel、复制、范围和相关事件的采集与查看。进程、线程、设备、采集范围和时间窗口共同限定观测含义。

## InfraAgents 如何据此推导

本库从请求关联到批次，再到 API/kernel；分别统计累计区间与区间并集。缺表、空表、窗口外与无法关联保持不同意义。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
