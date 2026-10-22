# S-SARATHI：SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills

## 本地完整原文

[完整原文](../../references/S-SARATHI/reader.html) · [全文文本](../../references/S-SARATHI/fulltext.txt)

固定版本：2308.16369v1。来源版本日期：2023-08-31；整理核对日期：2026-10-22。

## 来源支持的机制与定义

SARATHI 将长 prefill 切块，并在合适的批处理中与 decode 组织在一起，以减少两类工作之间的干扰并利用计算资源。

## InfraAgents 如何据此推导

本库从 q 个新 query、h 个历史 token 推导 attention 对数 qh＋q(q＋1)/2；固定 token 块不等于固定时间片，块长需结合目标 ITL、launch 和权重复用选择。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
