# S-DISTSERVE：DistServe

## 本地完整原文

[完整原文](../../references/S-DISTSERVE/reader.html) · [全文文本](../../references/S-DISTSERVE/fulltext.txt)

固定版本：2401.09670v3。来源版本日期：2024-06-06；整理核对日期：2026-10-22。

## 来源支持的机制与定义

DistServe 将 prefill 与 decode 的资源分配和并行选择解耦，并把阶段间状态通信与链路条件纳入服务目标。

## InfraAgents 如何据此推导

本库将总路径拆成两侧排队、计算、KV 转换与传输，说明分离收益必须超过新增通信和接管成本。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
