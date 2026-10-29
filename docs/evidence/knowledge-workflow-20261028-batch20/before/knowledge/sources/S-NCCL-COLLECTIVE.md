# S-NCCL-COLLECTIVE：NCCL — Collective Operations

## 本地完整原文

[完整原文](../../references/S-NCCL-COLLECTIVE/reader.html) · [全文文本](../../references/S-NCCL-COLLECTIVE/fulltext.txt)

固定版本：NCCL 2.25.1。来源版本日期：2025-01-27；整理核对日期：2026-10-22。

## 来源支持的机制与定义

集合通信操作由 communicator、参与 ranks、数据类型、数量、归约方式或 root 等共同定义。参与者必须执行相互匹配的逻辑操作。

## InfraAgents 如何据此推导

本库由矩阵分片推导 gather/reduce 的必要性，再讨论动态批、零 token rank、失败传播与共同提交点。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
